import os
import numpy as np
import time
import pandas as pd
import torch
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from tabpfn_extensions.many_class import ManyClassClassifier
from codecarbon import EmissionsTracker
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
    confusion_matrix
)
import math
from .GPUMonitor import GPUMemoryMonitor


TABPFN_VERSIONS = {
    "2.5": ModelVersion.V2_5,
    "3.5": ModelVersion.V3_5,
}

MAX_FEATURES_PER_ESTIMATOR = {
    "2.5": 500,
    "3.5": 768,
}


def build_classifier(tabpfn_version: str, SEED: int, run: int, DEVICE: str, n_features: int):
    """Constrói o classificador de acordo com tabpfn_version."""

    if tabpfn_version in TABPFN_VERSIONS:
        version = TABPFN_VERSIONS[tabpfn_version]
        
        max_feats_per_est = MAX_FEATURES_PER_ESTIMATOR[tabpfn_version]
        n_est = max(8, math.ceil(n_features / max_feats_per_est))

        estimator = TabPFNClassifier.create_default_for_version(
            version,
            ignore_pretraining_limits=True,
            device=DEVICE,
            random_state=SEED+run,
            n_estimators=n_est,
        )
        # ManyClassClassifier decompõe problemas com muitas classes em vários subproblemas menores.
        return ManyClassClassifier(estimator=estimator)

    raise ValueError(
        f"Modelo desconhecido: '{tabpfn_version}'. "
        f"Opções válidas: {list(TABPFN_VERSIONS)}"
    )


def _gpu_mb() -> float:
    return torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0


def train_test_model(
    X_train,
    X_test,
    y_train,
    SEED,
    run,
    DEVICE,
    dataset_name,
    tabpfn_version,
    emissions_dir=None,
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
 
    classifier = build_classifier(tabpfn_version, SEED, run, DEVICE, n_features=X_train.shape[1])
 
    # --- Warm-up: força o carregamento real dos pesos na GPU ---
    # create_default_for_version() só monta a config; o TabPFN baixa e
    # aloca os pesos na GPU dentro do primeiro .fit(). Sem isso, medir
    # vram_model_mb logo após build_classifier() dá 0 (ou um valor
    # contaminado por runs anteriores no mesmo processo).
    # Usamos um dataset sintético minúsculo, então o custo de "dado" aqui
    # é desprezível — o que sobra é essencially só o peso do modelo.
    dummy_X = np.random.randn(4, 2).astype(np.float32)
    dummy_y = np.array([0, 1, 0, 1])
    classifier.fit(dummy_X, dummy_y)
    vram_model_mb = _gpu_mb()

    classifier = build_classifier(tabpfn_version, SEED, run, DEVICE, n_features=X_train.shape[1])
 
    # Reseta o pico ANTES do fit real, senão o pico do warm-up (que inclui
    # o próprio carregamento do checkpoint) contamina vram_peak_fit_mb.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
 
    monitor = GPUMemoryMonitor(interval=0.005)  # Amostra a cada 5ms
 
    # 2. Executa Fit real (Carregamento de contexto ICL do dataset de verdade)
    tracker_train = EmissionsTracker(
        project_name=f"{dataset_name}_run{run + 1}_train",
        output_dir=emissions_dir,
        output_file="emissions_train.csv",
        log_level="error",
    )
 
    start_train = time.time()
    tracker_train.start()
    try:
        classifier.fit(X_train, y_train)
    finally:
        tracker_train.stop()
    train_time = time.time() - start_train
 
    # 3. Executa Predict (Forward Pass ICL)
    tracker_test = EmissionsTracker(
        project_name=f"{dataset_name}_run{run + 1}_test",
        output_dir=emissions_dir,
        output_file="emissions_test.csv",
        log_level="error",
    )
 
    start_test = time.time()
    tracker_test.start()
    monitor.start()
    try:
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)
    finally:
        tracker_test.stop()
        vram_peak_predict_mb = monitor.stop()
    test_time = time.time() - start_test
 
    gpu_metrics = {
        "vram_model_mb": vram_model_mb,
        "vram_data_mb": vram_peak_predict_mb - vram_model_mb,
        "vram_total_mb": vram_peak_predict_mb,
    }
 
    return y_pred, y_proba, train_time, test_time, gpu_metrics


def get_results(y_test, y_pred, y_proba, dataset_name, run):
    # Calculando matriz de confusão para Sensitivity e Specificity
    cm = confusion_matrix(y_test, y_pred)
    
    # Tratamento para binário vs multiclasse
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # Média macro por classe em problemas multiclasse
        sensitivity = np.mean(np.diag(cm) / np.maximum(1, cm.sum(axis=1)))
        
        specificities = []
        for i in range(len(cm)):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(np.delete(cm[:, i], i))
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificities.append(spec)
        specificity = np.mean(specificities)

    # AUROC (suporta binário e multiclasse ovr)
    try:
        if len(np.unique(y_test)) == 2:
            # Se for binário, pega a probabilidade da classe positiva
            prob_input = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            auroc = roc_auc_score(y_test, prob_input)
        else:
            auroc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        auroc = np.nan

    # Log Loss
    try:
        logloss = log_loss(y_test, y_proba)
    except Exception:
        logloss = np.nan

    return {
        "dataset": dataset_name,
        "run": run,
        "accuracy": accuracy_score(y_test, y_pred),
        "auroc": auroc,
        "balacc": balanced_accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "logloss": logloss,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }

def store_results(results, csv_path):
    df_row = pd.DataFrame([results])
    if os.path.exists(csv_path):
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, index=False)


def print_results(results):
    print("\n           ===== Results =====")
    print(f"           Accuracy   : {results['accuracy']:.8f}")
    print(f"           AUROC      : {results['auroc']:.8f}")
    print(f"           BalAcc     : {results['balacc']:.8f}")
    print(f"           F1-score   : {results['f1']:.8f}")
    print(f"           Log Loss   : {results['logloss']:.8f}")
    print(f"           Sensitivity: {results['sensitivity']:.8f}")
    print(f"           Specificity: {results['specificity']:.8f}")
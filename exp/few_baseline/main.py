"""Few-shot baselines (config `model`: ridge | 1nn) instead of TabPFN.

Same protocol as exp/few/main.py: same cached features (redux -> multirocket, uea -> tsfresh), same pre-selected support
sets (instance_selection/instances), same resamples and same test sets. Only the classifier changes:
  ridge: the aeon MultiRocketClassifier head, StandardScaler(with_mean=False) -> RidgeClassifierCV(logspace(-3, 3, 10))
  1nn:   StandardScaler -> 1-nearest-neighbour (Euclidean)
The scaler is fitted on the support set only. No CodeCarbon: times are measured around fit/predict only.
Support sets with a single class get a constant prediction (that class) and are logged in single_class_*.csv.
Outputs: <output_dir>/<model>/<selection>/K_<k>/{results_*.csv, single_class_*.csv, skipped_*.csv, errors_*.csv, time/}
"""
import os


import sys
import gc
import glob
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.Parser import Parser
from utils.utils_data import get_data_extracted
from utils.utils_train import store_results, get_results, print_results
from utils.utils_train_baseline import MODELS, train_test_model_baseline

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Feature usada em cada benchmark (univariado -> multirocket, multivariado -> tsfresh)
FEATURE_BY_BENCHMARK = {
    "redux": "multirocket",
    "uea": "tsfresh",
}


class SelectionMismatchError(Exception):
    """As instâncias salvas não batem com o X_train carregado (seed/resample diferente)."""


# --------------------------------------------------------------------------- #
# Utilitários
# --------------------------------------------------------------------------- #
def append_to_csv(df: pd.DataFrame, csv_path: str):
    """Faz append em CSV criando o cabeçalho apenas se o arquivo não existir."""
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)


def add_meta(results, meta: dict):
    """Anexa metadados few-shot ao `results` (funciona para dict ou DataFrame)."""
    if isinstance(results, dict):
        results.update(meta)
    elif isinstance(results, pd.DataFrame):
        for key, value in meta.items():
            results[key] = value
    return results


def as_list(value):
    """Aceita um valor único ou uma lista no config."""
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def load_done_keys(time_dir: str):
    """Pares (dataset, run) já concluídos, lidos dos logs de tempo (escritos só após o sucesso)."""
    done = set()
    for path in glob.glob(os.path.join(time_dir, "time_*.csv")):
        try:
            df = pd.read_csv(path, usecols=["dataset", "run"])
        except Exception:
            continue
        done.update((str(d), int(r)) for d, r in zip(df["dataset"], df["run"]))
    return done


def load_selection(npz_path: str):
    """Lê o .npz de seleção. Retorna (indices, X_sel, y_sel) ou (None, None, None) se o seletor pulou."""
    with np.load(npz_path, allow_pickle=True) as data:
        idx = data["selected_indices"]
        if idx.shape == () and idx.item() is None:
            return None, None, None
        return np.asarray(idx).astype(int), data["X_selected"], data["y_selected"]


def validate_selection(idx, X_sel, y_sel, X_train, y_train):
    """Confere se os índices são válidos e se reproduzem exatamente os dados salvos."""
    n = len(X_train)
    if idx.ndim != 1 or len(idx) == 0:
        raise ValueError(f"selected_indices inválido (shape={idx.shape}).")
    if idx.min() < 0 or idx.max() >= n:
        raise ValueError(f"selected_indices fora do intervalo [0, {n - 1}].")
    if len(np.unique(idx)) != len(idx):
        raise ValueError("selected_indices contém duplicatas.")

    X_chk = np.asarray(X_train)[idx]
    y_chk = np.asarray(y_train)[idx]
    same_X = X_chk.shape == np.asarray(X_sel).shape and np.allclose(
        X_chk, X_sel, equal_nan=True
    )
    same_y = np.array_equal(y_chk, np.asarray(y_sel))
    if not (same_X and same_y):
        raise SelectionMismatchError(
            "As instâncias salvas diferem de X_train[idx]/y_train[idx]. "
            "Provável seed/resample diferente do usado na geração das instâncias."
        )


# --------------------------------------------------------------------------- #
# Uma combinação (run, dataset, seleção, K)
# --------------------------------------------------------------------------- #
def run_experiment(
    *,
    run,
    seed,
    dataset_name,
    feature_name,
    model,
    selection,
    k_shot,
    benchmark,
    process_timestamp,
    instances_dir,
    output_dir,
):
    resample_seed = seed + run
    k_tag = f"K_{k_shot}"

    # <output_dir>/<modelo>/<seleção>/K_<k>/time
    model_output_dir = os.path.join(output_dir, model, selection, k_tag)
    time_dir = os.path.join(model_output_dir, "time")
    for d in (model_output_dir, time_dir):
        os.makedirs(d, exist_ok=True)

    csv_path = os.path.join(model_output_dir, f"results_{benchmark}_{process_timestamp}.csv")
    time_csv_path = os.path.join(time_dir, f"time_{process_timestamp}.csv")
    single_class_csv_path = os.path.join(model_output_dir, f"single_class_{process_timestamp}.csv")
    errors_csv_path = os.path.join(model_output_dir, f"errors_{process_timestamp}.csv")
    skipped_csv_path = os.path.join(model_output_dir, f"skipped_{process_timestamp}.csv")

    # Retomada: pula o que já foi concluído em execuções anteriores
    if (dataset_name, resample_seed) in load_done_keys(time_dir):
        print(f"            [skip] {dataset_name} | {selection} | {k_tag} | run {resample_seed} já concluído.")
        return

    # Arquivo com as instâncias pré-selecionadas
    npz_path = os.path.join(
        instances_dir,
        selection,
        k_tag,
        f"run_{run}",
        f"{feature_name}_{dataset_name}_k{k_shot}_run{run}.npz",
    )

    # Features pré-extraídas
    bench_aux = "mv_core" if benchmark == "uea" else "redux"
    feature_cache_dir = os.path.join(
        BASE_DIR, "features", "feat", feature_name, bench_aux, dataset_name
    )

    base_log = {
        "dataset": dataset_name,
        "run": resample_seed,
        "feature": feature_name,
        "model": model,
        "selection": selection,
        "k_shot": k_shot,
    }

    try:
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Arquivo de instâncias não encontrado: {npz_path}")

        start_total_time = time.perf_counter()

        # Dados do resample (treino completo = pool de seleção, teste completo)
        X_train, y_train, X_test, y_test = get_data_extracted(
            feature_cache_dir, dataset_name, resample_seed
        )
        n_pool = len(X_train)

        # Carrega e valida a seleção
        idx, X_sel, y_sel = load_selection(npz_path)
        if idx is None:
            print(f"            [skip] seletor não gerou instâncias para {dataset_name} ({selection}, {k_tag}).")
            append_to_csv(
                pd.DataFrame([{**base_log, "reason": "skipped_by_selector"}]),
                skipped_csv_path,
            )
            return

        validate_selection(idx, X_sel, y_sel, X_train, y_train)

        X_sup = np.asarray(X_train)[idx]
        y_sup = np.asarray(y_train)[idx]
        coverage = len(np.unique(y_sup)) / len(np.unique(np.asarray(y_train)))

        (
            y_pred,
            y_proba,
            train_time,
            predict_time,
            predict_proba_time,
            test_time,
            constant_fallback,
        ) = train_test_model_baseline(model, X_sup, X_test, y_sup)

        if constant_fallback:
            # Support set with a single class: no head can be trained, every test series gets that class
            print(f"            [single class] predição constante: {dataset_name} | {selection} | {k_tag} | run {resample_seed}.")
            append_to_csv(pd.DataFrame([base_log]), single_class_csv_path)

        total_time = time.perf_counter() - start_total_time

        all_classes = np.unique(np.concatenate([np.asarray(y_train), np.asarray(y_test)]))
        seen_classes = np.unique(y_sup)

        if y_proba.shape[1] != len(seen_classes):
            raise ValueError("[error] y_proba não tem uma coluna por classe vista no support set.")

        if len(seen_classes) < len(all_classes):
            padded = np.zeros((y_proba.shape[0], len(all_classes)))
            col = {c: i for i, c in enumerate(all_classes)}
            for j, c in enumerate(seen_classes):
                padded[:, col[c]] = y_proba[:, j]
            y_proba = padded

        # Evita log(0): piso pequeno e renormalização
        eps = 1e-15
        y_proba = np.clip(y_proba, eps, 1.0 - eps)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

        results = get_results(
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            dataset_name=dataset_name,
            run=resample_seed,
        )
        results["auroc"] = np.nan      # no predict_proba: one-hot "probabilities" make
        results["logloss"] = np.nan    # AUROC and log-loss meaningless, so they are not reported
        results = add_meta(
            results,
            {
                "model": model,
                "single_class_support": constant_fallback,
                "selection": selection,
                "k_shot": k_shot,
                "n_support": len(X_sup),
                "n_pool": n_pool,
                "class_coverage": coverage,
            },
        )
        print_results(results)
        store_results(results, csv_path)

        # Log de tempo (marca a combinação como concluída)
        append_to_csv(
            pd.DataFrame([{
                "dataset": dataset_name,
                "run": resample_seed,
                "selection": selection,
                "k_shot": k_shot,
                "train_time": train_time,
                "predict_time": predict_time,
                "predict_proba_time": predict_proba_time,
                "test_time": test_time,
                "total_execution_time": total_time,
            }]),
            time_csv_path,
        )

    except SelectionMismatchError as e:
        # Erro sistemático (seed errado): todas as próximas combinações seriam inválidas. Aborta.
        print("FATAL!!!")
        print(f"[FATAL] {dataset_name} | {selection} | {k_tag} | run {resample_seed}: {e}")
        append_to_csv(
            pd.DataFrame([{**base_log, "error_type": type(e).__name__, "error_msg": str(e)}]),
            errors_csv_path,
        )
        raise

    except Exception as e:
        print("ERROR!!!")
        print(
            f"[ERROR] {feature_name} | {selection} | {k_tag} on dataset "
            f"{dataset_name} (run {resample_seed}) failed:"
        )
        print(type(e).__name__, e)
        append_to_csv(
            pd.DataFrame([{**base_log, "error_type": type(e).__name__, "error_msg": str(e)}]),
            errors_csv_path,
        )

    finally:
        gc.collect()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    # Load config.yaml.
    parser = Parser()
    config = parser.parse()

    # Extract config parameters.
    NUM_RUNS = int(config.get("runs", 3))
    BENCHMARK = config.get("benchmark", "uea")
    DATASETS = config.get("datasets", [])
    SEED = config.get("seed", 0)
    MODEL = config.get("model", "ridge")
    OUTPUT_DIR = config.get("output_dir", os.path.join(BASE_DIR, "outputs", "few_baseline"))
    SPECIFIC_RESAMPLES = config.get("specific_resamples", None)
    K_VALUES = as_list(config.get("k", [1, 3, 5, 10]))
    SELECTIONS = as_list(
        config.get("selection", ["random", "stratified_random", "kmedoids", "stratified_kmedoids"])
    )
    INSTANCES_DIR = config.get(
        "instances_dir", os.path.join(BASE_DIR, "instance_selection", "instances")
    )

    if BENCHMARK not in FEATURE_BY_BENCHMARK:
        raise ValueError(
            f"Benchmark '{BENCHMARK}' desconhecido. Opções: {list(FEATURE_BY_BENCHMARK)}"
        )
    FEATURE = FEATURE_BY_BENCHMARK[BENCHMARK]
    if MODEL not in MODELS:
        raise ValueError(f"Modelo '{MODEL}' desconhecido. Opções: {list(MODELS)}")

    # TIMESTAMP ÚNICO deste processo
    process_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determina quais runs serão executadas
    if SPECIFIC_RESAMPLES is not None and len(SPECIFIC_RESAMPLES) > 0:
        run_ids = SPECIFIC_RESAMPLES
        print(f"Modo re-run ativado. Executando resamples específicos: {run_ids}")
    else:
        run_ids = list(range(NUM_RUNS))

    print(f"Modelo: {MODEL} | Benchmark: {BENCHMARK} | feature: {FEATURE} | K:{K_VALUES} | seleções: {SELECTIONS}")

    for run in run_ids:
        print(f"\n === Started run {SEED + run} === ")

        for dataset_name in DATASETS:
            print(f"\n    --- Executing dataset {dataset_name} --- ")

            for selection in SELECTIONS:
                for k_shot in K_VALUES:
                    print(f"\n        -> selection={selection} | K={k_shot}")
                    run_experiment(
                        run=run,
                        seed=SEED,
                        dataset_name=dataset_name,
                        feature_name=FEATURE,
                        model=MODEL,
                        selection=selection,
                        k_shot=k_shot,
                        benchmark=BENCHMARK,
                        process_timestamp=process_timestamp,
                        instances_dir=INSTANCES_DIR,
                        output_dir=OUTPUT_DIR,
                    )

    print("\nDONE!")


if __name__ == "__main__":
    main()
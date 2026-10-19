import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CODECARBON_LOG_LEVEL"] = "error"

import sys
import gc
import glob
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.Parser import Parser
from utils.utils_data import get_data_extracted
from utils.utils_train import store_results, train_test_model, get_results, print_results

warnings.filterwarnings("ignore", category=RuntimeWarning)

# 3.5 version
warnings.filterwarnings(
    "ignore",
    message=r"Running .* estimators covers at most .* features .* the remaining features are never sampled.*",
    category=UserWarning,
)

# 3.5 version
warnings.filterwarnings(
    "ignore",
    message=r"Auto-scaling n_estimators from .* capped at MAX_AUTO_SCALED_N_ESTIMATORS=.*",
    category=UserWarning,
)

# 2.5 version
warnings.filterwarnings(
    "ignore",
    message=r"Auto-scaling n_estimators from .* to .* so every feature is included.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r"The number of unique classes is greater than 50% of the number of samples.*",
    category=UserWarning,
    module=r"tabpfn\.validation",
)

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
    selection,
    k_shot,
    benchmark,
    device,
    tabpfn_version,
    process_timestamp,
    instances_dir,
):
    resample_seed = seed + run
    k_tag = f"K_{k_shot}"

    # outputs/few/<feature>/<seleção>/K_<k>/{emissions,time,gpu}
    model_output_dir = os.path.join(
        BASE_DIR, "outputs", "few", selection, k_tag
    )
    emissions_dir = os.path.join(model_output_dir, "emissions")
    time_dir = os.path.join(model_output_dir, "time")
    gpu_dir = os.path.join(model_output_dir, "gpu")
    for d in (model_output_dir, emissions_dir, time_dir, gpu_dir):
        os.makedirs(d, exist_ok=True)

    csv_path = os.path.join(model_output_dir, f"results_{benchmark}_{process_timestamp}.csv")
    time_csv_path = os.path.join(time_dir, f"time_{process_timestamp}.csv")
    gpu_csv_path = os.path.join(gpu_dir, f"gpu_{process_timestamp}.csv")
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
            gpu_metrics,
        ) = train_test_model(
            X_sup,
            X_test,
            y_sup,
            seed,
            run,
            device,
            dataset_name,
            tabpfn_version,
            process_timestamp=process_timestamp,
            emissions_dir=emissions_dir,
        )

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
        results = add_meta(
            results,
            {
                "selection": selection,
                "k_shot": k_shot,
                "n_support": len(X_sup),
                "n_pool": n_pool,
                "class_coverage": coverage,
            },
        )
        print_results(results)
        store_results(results, csv_path)

        # Log de GPU (antes do de tempo: o de tempo marca a combinação como concluída)
        append_to_csv(
            pd.DataFrame([{
                "dataset": dataset_name,
                "run": resample_seed,
                "selection": selection,
                "k_shot": k_shot,
                "n_train_samples": len(X_sup),
                "n_features": X_sup.shape[1],
                "vram_model_mb": gpu_metrics["vram_model_mb"],
                "vram_data_mb": gpu_metrics["vram_data_mb"],
                "vram_total_mb": gpu_metrics["vram_total_mb"],
            }]),
            gpu_csv_path,
        )

        # Log de tempo
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
        # Limpeza explícita da RAM e da VRAM ao fim de cada combinação
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
    DEVICE = config.get("device", "cuda")
    TABPFN_VERSION = str(config.get("tabpfn_version", "3.5"))
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

    # TIMESTAMP ÚNICO deste processo
    process_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determina quais runs serão executadas
    if SPECIFIC_RESAMPLES is not None and len(SPECIFIC_RESAMPLES) > 0:
        run_ids = SPECIFIC_RESAMPLES
        print(f"Modo re-run ativado. Executando resamples específicos: {run_ids}")
    else:
        run_ids = list(range(NUM_RUNS))

    print(f"Benchmark: {BENCHMARK} | feature: {FEATURE} | K: {K_VALUES} | seleções: {SELECTIONS}")

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
                        selection=selection,
                        k_shot=k_shot,
                        benchmark=BENCHMARK,
                        device=DEVICE,
                        tabpfn_version=TABPFN_VERSION,
                        process_timestamp=process_timestamp,
                        instances_dir=INSTANCES_DIR,
                    )

    print("\nDONE!")


if __name__ == "__main__":
    main()
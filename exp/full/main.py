import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CODECARBON_LOG_LEVEL"] = "error"

import sys
import gc
import torch
import time
import warnings
import pandas as pd
from datetime import datetime
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

def append_to_csv(df: pd.DataFrame, csv_path: str):
    """Função utilitária para fazer append em CSV criando o cabeçalho apenas se o arquivo não existir."""
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)


def main():
    # Load config.yaml.
    parser = Parser()
    config = parser.parse()

    # Extract config parameters.
    NUM_RUNS = int(config.get("runs", 1))
    BENCHMARK = config.get("benchmark", "redux")
    DATASETS = config.get("datasets", [])
    FEATURES = config.get("feature", [])
    SEED = config.get("seed", 0)
    DEVICE = config.get("device", "cuda")
    TABPFN_VERSION = str(config.get("tabpfn_version", "3.5"))
    SPECIFIC_RESAMPLES = config.get("specific_resamples", None)

    # Gera o TIMESTAMP ÚNICO deste processo
    process_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determina quais runs serão executadas
    if SPECIFIC_RESAMPLES is not None and len(SPECIFIC_RESAMPLES) > 0:
        run_ids = SPECIFIC_RESAMPLES
        print(f"Modo re-run ativado. Executando resamples específicos: {run_ids}")
    else:
        run_ids = list(range(NUM_RUNS))

    for run in run_ids:
        print(f"\n === Started run {SEED+run} === ")

        for dataset_name in DATASETS:
            print(f"\n    --- Executing dataset {dataset_name} --- ")

            for feature_name in FEATURES:
                print(f"\n        -> Running feature {feature_name}  ")

                # Define a estrutura de diretórios base e subpastas para cada tipo de log
                model_output_dir = os.path.join(BASE_DIR, "outputs", TABPFN_VERSION, feature_name)
                emissions_dir = os.path.join(model_output_dir, "emissions")
                time_dir = os.path.join(model_output_dir, "time")
                gpu_dir = os.path.join(model_output_dir, "gpu")

                os.makedirs(model_output_dir, exist_ok=True)
                os.makedirs(emissions_dir, exist_ok=True)
                os.makedirs(time_dir, exist_ok=True)
                os.makedirs(gpu_dir, exist_ok=True)

                # CSV de resultados principais ÚNICO por processo
                csv_path = os.path.join(model_output_dir, f"results_{process_timestamp}.csv")

                # Arquivos CSV únicos DESTE PROCESSO para time e gpu
                time_csv_path = os.path.join(time_dir, f"time_{process_timestamp}.csv")
                gpu_csv_path = os.path.join(gpu_dir, f"gpu_{process_timestamp}.csv")

                # Folder com as características pré-extraídas
                feature_cache_dir = os.path.join(
                    BASE_DIR, "features", "feat", feature_name, BENCHMARK, dataset_name
                )

                try:
                    start_total_time = time.perf_counter()

                    # Pegando os dados pré-extraídos
                    X_train, y_train, X_test, y_test = get_data_extracted(
                        feature_cache_dir, dataset_name, SEED+run
                    )

                    y_pred, y_proba, train_time, predict_time, predict_proba_time, test_time, gpu_metrics = train_test_model(
                        X_train,
                        X_test,
                        y_train,
                        SEED,
                        run,
                        DEVICE,
                        dataset_name,
                        TABPFN_VERSION,
                        process_timestamp=process_timestamp,
                        emissions_dir=emissions_dir,
                    )

                    end_total_time = time.perf_counter()
                    total_time = end_total_time - start_total_time

                    results = get_results(
                        y_test=y_test,
                        y_pred=y_pred,
                        y_proba=y_proba,
                        dataset_name=dataset_name,
                        run=SEED+run,
                    )
                    print_results(results)
                    
                    # Salva os resultados no CSV do próprio processo
                    store_results(results, csv_path)

                    # Append no log de tempo do processo
                    df_time = pd.DataFrame([{
                        "dataset": dataset_name,
                        "run": SEED + run,
                        "train_time": train_time,
                        "predict_time": predict_time,
                        "predict_proba_time": predict_proba_time,
                        "test_time": test_time,
                        "total_execution_time": total_time,
                    }])
                    append_to_csv(df_time, time_csv_path)

                    # Append no log de GPU do processo
                    df_gpu = pd.DataFrame([{
                        "dataset": dataset_name,
                        "run": SEED+run,
                        "n_train_samples": len(X_train),
                        "n_features": X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0]),
                        "vram_model_mb": gpu_metrics["vram_model_mb"],
                        "vram_data_mb": gpu_metrics["vram_data_mb"],
                        "vram_total_mb": gpu_metrics["vram_total_mb"],
                    }])
                    append_to_csv(df_gpu, gpu_csv_path)

                except Exception as e:
                    print("ERROR!!!")
                    print(
                        f"[ERROR] {feature_name} on dataset {dataset_name} failed:"
                    )
                    print(type(e).__name__, e)

                finally:
                    # Limpeza explícita da RAM e da VRAM ao fim de cada combinação
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    print("\nDONE!")


if __name__ == "__main__":
    main()
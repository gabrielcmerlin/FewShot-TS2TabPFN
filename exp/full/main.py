import os
import sys
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

def main():
    # Load config.yaml.
    parser = Parser()
    config = parser.parse()
    FILENAME = "results.csv"

    # Extract config parameters.
    NUM_RUNS = int(config.get("runs", 1))
    BENCHMARK = config.get("benchmark", "redux")
    DATASETS = config.get("datasets", [])
    FEATURES = config.get("feature", [])
    SEED = config.get("seed", 0)
    DEVICE = config.get("device", "cuda")
    TABPFN_VERSION = str(config.get("tabpfn_version", "3.5"))

    for run in range(NUM_RUNS):
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

                # Definindo onde salvar o CSV de resultados principais
                csv_path = os.path.join(model_output_dir, FILENAME)

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

                    y_pred, y_proba, train_time, test_time, gpu_metrics = train_test_model(
                        X_train,
                        X_test,
                        y_train,
                        SEED,
                        run,
                        DEVICE,
                        dataset_name,
                        TABPFN_VERSION,
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
                    store_results(results, csv_path)

                    # Gera o timestamp único no formato AAAAMMDD_HHMMSS_microseconds
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # Log de tempo em pasta e arquivo específicos
                    time_record = [{
                        "dataset": dataset_name,
                        "run": SEED+run,
                        "train_time": train_time,
                        "test_time": test_time,
                        "total_execution_time": total_time,
                    }]
                    
                    df_time = pd.DataFrame(time_record)
                    time_csv_path = os.path.join(time_dir, f"time_{timestamp}.csv")
                    df_time.to_csv(time_csv_path, index=False)

                    # Log de GPU em pasta e arquivo específicos
                    gpu_record = [{
                        "dataset": dataset_name,
                        "run": SEED+run,
                        "n_train_samples": len(X_train),
                        "n_features": X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0]),
                        "vram_model_mb": gpu_metrics["vram_model_mb"],
                        "vram_data_mb": gpu_metrics["vram_data_mb"],
                        "vram_total_mb": gpu_metrics["vram_total_mb"],
                    }]

                    df_gpu = pd.DataFrame(gpu_record)
                    gpu_csv_path = os.path.join(gpu_dir, f"gpu_{timestamp}.csv")
                    df_gpu.to_csv(gpu_csv_path, index=False)

                except Exception as e:
                    print("ERROR!!!")
                    print(
                        f"[ERROR] {feature_name} on dataset {dataset_name} failed:"
                    )
                    print(type(e).__name__, e)

    print("\nDONE!")


if __name__ == "__main__":
    main()
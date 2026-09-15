import csv
import os
import sys
import time
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
from aeon.transformations.collection.feature_based import Catch22
from codecarbon import EmissionsTracker
from utils.Parser import Parser
from utils.utils_data import get_data
from datetime import datetime

warnings.filterwarnings("ignore", category=RuntimeWarning)

def main():
    parser = Parser()
    config = parser.parse()

    benchmark = config.get("benchmark") or config.get("task")
    datasets_list = config.get("datasets", [])

    if not benchmark:
        raise ValueError("[ERRO] A chave 'benchmark' (ou 'task') não foi encontrada no arquivo de configuração.")

    # Base paths
    data_path = os.path.join(BASE_DIR, "data", benchmark)
    output_base_dir = os.path.join(BASE_DIR, "features", benchmark)
    logs_dir = os.path.join(BASE_DIR, "features", "logs")

    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    time_log_file = os.path.join(logs_dir, f"time_{run_id}.csv")

    if not datasets_list and os.path.exists(data_path):
        datasets_list = [
            d for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        ]

    print(f"Benchmark: {benchmark} | Datasets encontrados: {len(datasets_list)}")

    for dataset_name in datasets_list:
        print(f"\n=== Extraindo Catch22 para {dataset_name} ===")

        # Create a specific directory for each dataset: features/<benchmark>/<dataset_name>/
        dataset_dir = os.path.join(output_base_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # Initialize CodeCarbon per dataset
        tracker = EmissionsTracker(
            project_name=f"catch22_{benchmark}_{dataset_name}",
            output_dir=logs_dir,
            output_file=f"emissions_{run_id}.csv",  # Appends individual runs to the same CSV
            log_level="critical",
        )

        tracker.start()
        try:
            X_train, y_train, X_test, y_test = get_data(
                data_path,
                dataset_name,
                resample_id=0
            )

            n_train = len(X_train)
            X_full = np.concatenate([X_train, X_test], axis=0)
            y_full = np.concatenate([y_train, y_test], axis=0)

            transformer = Catch22(n_jobs=16)

            start_time = time.time()
            transformer.fit(X_train)
            X_full_feat = transformer.transform(X_full)
            duration = time.time() - start_time

            if not isinstance(X_full_feat, np.ndarray):
                X_full_feat = X_full_feat.to_numpy()

            # Save arrays into features/<benchmark>/<dataset_name>/
            np.save(os.path.join(dataset_dir, f"{dataset_name}_X.npy"), X_full_feat)
            np.save(os.path.join(dataset_dir, f"{dataset_name}_y.npy"), y_full)
            np.save(os.path.join(dataset_dir, f"{dataset_name}_ntrain.npy"), np.array([n_train]))

            # Log runtime to CSV
            file_exists = os.path.isfile(time_log_file)
            with open(time_log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["dataset", "n_samples", "n_features", "time_seconds"])
                writer.writerow([
                    dataset_name,
                    len(X_full),
                    X_full_feat.shape[1],
                    round(duration, 4),
                ])

            print(f"Sucesso: {dataset_name}")

        except Exception as e:
            print(f"[ERRO] Falha no dataset {dataset_name}: {type(e).__name__} - {e}")

        finally:
            emissions = tracker.stop()
            print(f"[CodeCarbon] Emissões ({dataset_name}): {emissions:.6f} kg CO2eq")


if __name__ == "__main__":
    main()
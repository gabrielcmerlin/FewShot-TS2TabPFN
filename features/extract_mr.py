import csv
import os
import sys
import time
import warnings
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from aeon.transformations.collection.convolution_based import MultiRocket
from codecarbon import EmissionsTracker
from utils.Parser import Parser
from utils.utils_data import feature_stem, get_data

warnings.filterwarnings("ignore", category=RuntimeWarning)


def find_actual_dataset_name(dataset_folder_path):
    """
    Escaneia a pasta do dataset e identifica o prefixo real do arquivo _TRAIN.ts.
    Resolve variações como 'ShakeGestureWiimoteZ_eq_TRAIN.ts' ou 'Dataset_nmv_TRAIN.ts'.
    """
    if not os.path.exists(dataset_folder_path):
        return None

    for f in os.listdir(dataset_folder_path):
        if f.endswith("_TRAIN.ts"):
            # Remove o sufixo '_TRAIN.ts' para obter o nome exato esperado por get_data
            return f[:-9]
    return None


def main():
    parser = Parser()
    config = parser.parse()

    benchmark = config.get("benchmark") or config.get("task")
    datasets_list = config.get("datasets", [])
    SEED = int(config.get("seed", 0))
    NUM_RUNS = int(config.get("runs", 1))
    SPECIFIC_RESAMPLES = config.get("specific_resamples", None)
    # O número de threads do transform é o n_jobs do MultiRocket: o aeon ignora numba.set_num_threads
    # e o limita ao número de núcleos disponíveis.
    N_JOBS = int(config.get("n_jobs", 16))

    if not benchmark:
        raise ValueError(
            "[ERRO] A chave 'benchmark' (ou 'task') não foi encontrada no arquivo de configuração."
        )

    # Mesma convenção dos experimentos: reamostragem = seed + run (0 = divisão oficial)
    if SPECIFIC_RESAMPLES is not None and len(SPECIFIC_RESAMPLES) > 0:
        run_ids = list(SPECIFIC_RESAMPLES)
    else:
        run_ids = list(range(NUM_RUNS))
    resample_ids = [SEED + run for run in run_ids]

    # Base paths para o MultiROCKET
    data_path = os.path.join(BASE_DIR, "data", benchmark)
    output_base_dir = os.path.join(BASE_DIR, "features", "feat", "multirocket", benchmark)
    logs_dir = os.path.join(BASE_DIR, "features", "logs", "multirocket")

    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    time_dir = os.path.join(logs_dir, "time")
    os.makedirs(time_dir, exist_ok=True)
    time_log_file = os.path.join(time_dir, f"time_{run_id}.csv")

    if not datasets_list and os.path.exists(data_path):
        datasets_list = [
            d
            for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        ]

    print(
        f"Benchmark: {benchmark} | Datasets encontrados: {len(datasets_list)} | "
        f"Reamostragens: {resample_ids} | n_jobs: {N_JOBS}"
    )

    for dataset_name in datasets_list:
        print(f"\n=== Extraindo MultiROCKET para {dataset_name} ===")

        target_folder = os.path.join(data_path, dataset_name)

        # Resolve o nome exato considerando sufixos (_eq, _nmv, etc.)
        actual_name = find_actual_dataset_name(target_folder)
        if not actual_name:
            actual_name = dataset_name

        dataset_dir = os.path.join(output_base_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        emission_dir = os.path.join(logs_dir, "emission")
        os.makedirs(emission_dir, exist_ok=True)

        # Uma tabela por reamostragem: o MultiROCKET é ajustado apenas com o treino daquela reamostragem
        for resample_id in resample_ids:
            stem = feature_stem(dataset_name, resample_id)
            print(f"  -- reamostragem {resample_id}")

            tracker = EmissionsTracker(
                project_name=f"multirocket_{benchmark}_{stem}",
                output_dir=emission_dir,
                output_file=f"emissions_{run_id}.csv",
                log_level="critical",
            )

            tracker.start()
            try:
                # Carrega os dados já divididos (mesma divisão estratificada dos experimentos)
                X_train, y_train, X_test, y_test = get_data(
                    data_path, actual_name, dataset_name, resample_id=resample_id
                )

                n_train = len(X_train)
                X_full = np.concatenate([X_train, X_test], axis=0)
                y_full = np.concatenate([y_train, y_test], axis=0)

                print("Ajustando e transformando com MultiROCKET...")
                start_time = time.time()

                transformer = MultiRocket(n_jobs=N_JOBS, random_state=resample_id)
                transformer.fit(X_train)
                X_full_feat = transformer.transform(X_full)

                duration = time.time() - start_time

                if not isinstance(X_full_feat, np.ndarray):
                    X_full_feat = X_full_feat.to_numpy()

                # Salva os arrays no diretório de destino ([treino; teste], com n_train linhas de treino)
                np.save(os.path.join(dataset_dir, f"{stem}_X.npy"), X_full_feat)
                np.save(os.path.join(dataset_dir, f"{stem}_y.npy"), y_full)
                np.save(
                    os.path.join(dataset_dir, f"{stem}_ntrain.npy"),
                    np.array([n_train]),
                )

                file_exists = os.path.isfile(time_log_file)
                with open(time_log_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(
                            ["dataset", "n_samples", "n_features", "time_seconds", "resample"]
                        )
                    writer.writerow(
                        [
                            dataset_name,
                            len(X_full),
                            X_full_feat.shape[1],
                            round(duration, 4),
                            resample_id,
                        ]
                    )

                print(f"Sucesso: {dataset_name} (reamostragem {resample_id})")

            except Exception as e:
                print("ERROR!!!")
                print(
                    f"[ERRO] Falha no dataset {dataset_name} (reamostragem {resample_id}): "
                    f"{type(e).__name__} - {e}"
                )

            finally:
                emissions = tracker.stop()
                print(
                    f"[CodeCarbon] Emissões ({dataset_name}, reamostragem {resample_id}): {emissions:.6f} kg CO2eq"
                )

    print("DONE!")


if __name__ == "__main__":
    main()

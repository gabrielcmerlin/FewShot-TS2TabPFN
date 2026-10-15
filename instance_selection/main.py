import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.Parser import Parser
from utils.utils_data import get_data_extracted
from utils.utils_selection import (
    random_selection,
    stratified_random_selection,
    kmedoids_selection,
    stratified_kmedoids_selection
)

def main():
    # Load config.yaml.
    parser = Parser()
    config = parser.parse()

    # Extract config parameters.
    SEED = config.get("seed", 0)
    RUNS = config.get("runs", 30)
    K_VALUES = config.get("k", [1, 3, 5, 10])
    FEATURE = config.get("feature", None)
    DATASETS = config.get("datasets", None)
    BENCHMARK = config.get("benchmark", None)
    SELECTION = config.get("selection", ['random', 'stratified_random', 'kmedoids', 'stratified_kmedoids'])

    # Cria o diretório de destino se não existir
    output_dir = os.path.join(BASE_DIR, "instance_selection", "instances")
    os.makedirs(output_dir, exist_ok=True)

    for selection_method in SELECTION:
        print(f"Selection method: {selection_method}")
        
        for run in range(RUNS):
            print(f"Run {run + 1}/{RUNS}")

            for dataset_name in DATASETS:
                print(f"Dataset: {dataset_name}")

                for K in K_VALUES:
                    print(f"K: {K}")

                    feature_cache_dir = os.path.join(
                        BASE_DIR, "features", "feat", FEATURE, BENCHMARK, dataset_name
                    )

                    # Pegando os dados pré-extraídos
                    X_train, y_train, _, _ = get_data_extracted(
                        feature_cache_dir, dataset_name, SEED + run
                    )

                    # --- SELEÇÃO DE INSTÂNCIAS ---
                    if selection_method == 'stratified_random':
                        selected_indices = stratified_random_selection(y_train, K, SEED + run)
                    elif selection_method == 'random':
                        selected_indices = random_selection(X_train, y_train, K, SEED + run)
                    elif selection_method == 'kmedoids':
                        selected_indices = kmedoids_selection(X_train, y_train, K, SEED + run)
                    elif selection_method == 'stratified_kmedoids':
                        selected_indices = stratified_kmedoids_selection(X_train, y_train, K, SEED + run)
                    else:
                        print(f"Unknown selection method: {selection_method}. Skipping...")
                        continue

                    if selected_indices is None:
                        X_train_selected = None
                        y_train_selected = None
                        print("Selection skipped. No instances saved.\n")
                    else:
                        X_train_selected = X_train[selected_indices]
                        y_train_selected = y_train[selected_indices]
                        print(f"Sampled {len(X_train_selected)} / {len(X_train)} training instances.")

                    # --- SALVANDO OS DADOS ---
                    filename = f"{FEATURE}_{dataset_name}_k{K}_run{run}.npz"
                    save_dir = os.path.join(
                        output_dir,
                        selection_method,
                        f"K_{K}",
                        f"run_{run}"
                    )
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, filename)

                    np.savez(
                        save_path,
                        selected_indices=selected_indices,
                        X_selected=X_train_selected,
                        y_selected=y_train_selected
                    )
                    
                    print(f"Saved: {save_path}\n")

if __name__ == "__main__":
    main()
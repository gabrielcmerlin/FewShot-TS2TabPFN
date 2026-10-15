import numpy as np
from sklearn.preprocessing import StandardScaler
import kmedoids

def fast_pairwise_euclidean(X):
    """Calcula matriz de distâncias euclidianas aos pares de forma vetorizada."""
    X_64 = X.astype(np.float64)
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_64)

    # ||a||^2
    norms = np.sum(X_norm**2, axis=1, keepdims=True)
    # ||a||^2 + ||b||^2 - 2 * <a, b>
    distances_sq = norms + norms.T - 2 * np.dot(X_norm, X_norm.T)
    np.maximum(distances_sq, 0, out=distances_sq)
    D = np.sqrt(distances_sq, out=distances_sq)
    np.fill_diagonal(D, 0)
    return D

def random_selection(X, y, K, seed_val):
    rng = np.random.default_rng(seed_val)
    num_classes = len(np.unique(y))
    print(f'K = {K}, Number of classes = {num_classes}')
    sample_size = K * num_classes  # Sample K instances per class

    if sample_size > len(X):
        print(f"Sample size {sample_size} is larger than training set size {len(X)}. Skipping...")
        selected_indices = None
    else:
        selected_indices = rng.choice(len(X), size=sample_size, replace=False)

    return selected_indices

def stratified_random_selection(y, K, seed_val):
    rng = np.random.default_rng(seed_val)
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    print(f'K = {K}, Number of classes = {len(unique_classes)}')

    # Verifica se alguma classe tem menos do que K elementos disponíveis
    if np.any(class_counts < K):
        print(f"Skipping: Some classes have fewer than K={K} instances.")
        return None

    selected_indices = []
    for c in unique_classes:
        # Encontra os índices de todas as instâncias da classe 'c'
        class_indices = np.where(y == c)[0]
        # Seleciona K índices aleatórios sem reposição para essa classe
        chosen_indices = rng.choice(class_indices, size=K, replace=False)
        selected_indices.extend(chosen_indices)

    return np.array(selected_indices)

def kmedoids_selection(X, y, K, seed_val):
    """K-Medoids Global (Unstratified) sobre todo o dataset X."""
    num_classes = len(np.unique(y))
    total_medoids = K * num_classes
    print(f'KMedoids (Global) - Total medoids: {total_medoids}')

    if total_medoids > len(X):
        print(f"Sample size {total_medoids} is larger than dataset size {len(X)}. Skipping...")
        return None

    D = fast_pairwise_euclidean(X)
    res = kmedoids.fasterpam(D, total_medoids, random_state=seed_val)
    return np.array(res.medoids)


def stratified_kmedoids_selection(X, y, K, seed_val):
    """K-Medoids por Classe (Estratificado) - K medoids para cada classe."""
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f'KMedoids (Class-wise) - K = {K}, Number of classes = {len(unique_classes)}')

    if np.any(class_counts < K):
        print(f"Skipping: Some classes have fewer than K={K} instances.")
        return None

    D = fast_pairwise_euclidean(X)
    selected_indices = []

    for c in unique_classes:
        idx_c = np.where(y == c)[0]
        D_c = D[np.ix_(idx_c, idx_c)]

        res_c = kmedoids.fasterpam(D_c, K, random_state=seed_val)
        selected_indices.extend(idx_c[res_c.medoids])

    return np.array(selected_indices)
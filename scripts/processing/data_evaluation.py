import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans, DBSCAN

def evaluate_kmeans(X, labels, model):
    """
    Évalue la qualité des clusters générés par K-Means.

    Paramètres :
    ------------
    X : ndarray (n_samples, n_features)
        Données utilisées pour le clustering.
    labels : ndarray (n_samples,)
        Labels des clusters assignés par K-Means.
    model : KMeans
        Modèle K-Means entraîné.

    Retourne :
    ----------
    dict :
        Un dictionnaire contenant les métriques suivantes :
        - 'silhouette_score' : Mesure la séparation des clusters (entre -1 et 1).
        - 'davies_bouldin_score' : Plus faible est la valeur, mieux c'est (cohésion intra-cluster vs séparation).
        - 'calinski_harabasz_score' : Plus élevé, mieux c'est (variance inter vs intra-cluster).
        - 'inertia' : Somme des distances intra-cluster (plus faible est mieux).
        - 'cluster_stability' : Variance des tailles des clusters (plus faible indique une répartition plus homogène).
    """
    metrics = {}

    # Vérifier qu'on a au moins 2 clusters pour calculer certaines métriques
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        metrics["silhouette_score"] = silhouette_score(X, labels)
        metrics["davies_bouldin_score"] = davies_bouldin_score(X, labels)
        metrics["calinski_harabasz_score"] = calinski_harabasz_score(X, labels)

    # Inertie (dispersion intra-cluster)
    if hasattr(model, "inertia_"):
        metrics["inertia"] = model.inertia_

    # Stabilité des clusters (variance du nombre de points par cluster)
    cluster_sizes = np.bincount(labels)
    metrics["cluster_stability"] = np.std(cluster_sizes)

    return metrics

def evaluate_dbscan(X, labels):
    """
    Évalue la qualité des clusters générés par DBSCAN.

    Paramètres :
    ------------
    X : ndarray (n_samples, n_features)
        Données utilisées pour le clustering.
    labels : ndarray (n_samples,)
        Labels des clusters assignés par DBSCAN (-1 pour le bruit).

    Retourne :
    ----------
    dict :
        Un dictionnaire contenant les métriques suivantes :
        - 'silhouette_score' : Mesure la séparation des clusters (si plus d’un cluster détecté).
        - 'noise_ratio' : Ratio de points considérés comme bruit (entre 0 et 1).
        - 'cluster_stability' : Variance des tailles des clusters (plus faible est mieux).
    """
    metrics = {}

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)  # -1 = bruit
    noise_ratio = np.sum(labels == -1) / len(labels)  # Taux de bruit

    # Calcul du score de silhouette uniquement si plus d'un cluster
    if n_clusters > 1:
        metrics["silhouette_score"] = silhouette_score(X, labels)

    metrics["noise_ratio"] = noise_ratio

    # Stabilité des clusters (variance du nombre de points par cluster hors bruit)
    cluster_sizes = [np.sum(labels == label) for label in unique_labels if label != -1]
    if cluster_sizes:
        metrics["cluster_stability"] = np.std(cluster_sizes)

    return metrics

def evaluate_clusters(X, labels, model=None):
    """
    Applique les métriques d'évaluation appropriées en fonction du type de clustering utilisé.

    Paramètres :
    ------------
    X : ndarray (n_samples, n_features)
        Données utilisées pour le clustering.
    labels : ndarray (n_samples,)
        Labels des clusters assignés par un algorithme de clustering.
    model : object, optionnel
        Modèle de clustering (KMeans ou DBSCAN).

    Retourne :
    ----------
    dict :
        Un dictionnaire contenant les métriques pertinentes en fonction du modèle.
    
    Erreurs :
    ----------
    ValueError :
        Si le modèle fourni n'est ni un KMeans ni un DBSCAN.
    """
    if isinstance(model, KMeans):
        return evaluate_kmeans(X, labels, model)
    elif isinstance(model, DBSCAN):
        return evaluate_dbscan(X, labels)
    else:
        raise ValueError("Modèle non reconnu. Utilisez KMeans ou DBSCAN.")

# Exemple d'utilisation :
# metrics = evaluate_clusters(X, labels, model)
# print(metrics)

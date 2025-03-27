import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from wordcloud import WordCloud

# === STEP 1: Placeholder Data ===
# Simule des données avec 4 clusters
X, y_placeholder = make_blobs(n_samples=500, centers=4, n_features=5, random_state=42)

# Générer une colonne de texte factice (par exemple pour simuler des mots-clés)
random_words = ["data", "model", "analysis", "python", "cluster", "feature", "vector", "distance", "label", "score"]
def generate_fake_text():
    return ' '.join(np.random.choice(random_words, size=np.random.randint(5, 10)))

texts = [generate_fake_text() for _ in range(len(X))]

# === STEP 2: Clustering Placeholder (remplace avec ton algo plus tard) ===
# On suppose que tu as un array "labels" des clusters prédits
labels = y_placeholder  # <- à remplacer par tes vrais labels plus tard

# === STEP 3: Evaluation des Clusters ===
def evaluate_clustering(X, labels):
    scores = {}
    scores['Silhouette Score'] = silhouette_score(X, labels)
    scores['Davies-Bouldin Score'] = davies_bouldin_score(X, labels)
    scores['Calinski-Harabasz Score'] = calinski_harabasz_score(X, labels)
    return scores

scores = evaluate_clustering(X, labels)
print("\n\n=== Cluster Evaluation ===")
for k, v in scores.items():
    print(f"{k}: {v:.3f}")

# === STEP 4: Analyse par Cluster ===
def cluster_summary(X, labels):
    df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
    df['Cluster'] = labels
    summary = df.groupby('Cluster').agg(['mean', 'std', 'count'])
    return summary

summary = cluster_summary(X, labels)
print("\n\n=== Cluster Summary ===")
print(summary)

# === STEP 5: Visualisation 2D ===
def plot_clusters_2D(X, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='tab10')
    plt.title("Clusters (PCA 2D projection)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_clusters_2D(X, labels)

# === STEP 6: Nuages de Mots par Cluster ===
def plot_wordclouds_by_cluster(df, text_col='text', cluster_col='Cluster'):
    unique_clusters = sorted(df[cluster_col].unique())
    n_clusters = len(unique_clusters)
    fig, axes = plt.subplots(1, n_clusters, figsize=(6*n_clusters, 6))
    if n_clusters == 1:
        axes = [axes]
    for i, cluster in enumerate(unique_clusters):
        text = ' '.join(df[df[cluster_col] == cluster][text_col].values)
        wordcloud = WordCloud(background_color='white', max_words=100).generate(text)
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].axis('off')
        axes[i].set_title(f"Cluster {cluster}")
    plt.tight_layout()
    plt.show()

# Créer un DataFrame complet pour wordcloud
df_text = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
df_text['Cluster'] = labels
df_text['text'] = texts

plot_wordclouds_by_cluster(df_text)

try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from scipy.cluster.hierarchy import dendrogram, linkage
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.mixture import GaussianMixture
    import abc
    # import DBSCAN
    import seaborn as sns
    from sklearn.decomposition import PCA, TruncatedSVD
    from typing import Union
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.manifold import TSNE, Isomap
except ImportError:
    print("Please install the required packages (scikit-learn, scipy, matplotlib)")
    exit(1)

from ..utilities.data_utils import load_data, log_action

class DataClustering(abc.ABC):
    def __init__(self, df, num_clusters) -> None:
        self.df = df
        self.num_clusters = num_clusters
        self.kmeans_model = self.km_model
        self.hierarchical_model = self.hierarchical_model
        self.gmm_model = self.gmm_model

    @classmethod
    def from_csv(cls, file_path, num_clusters) -> 'DataClustering':
        return cls(load_data(file_path), num_clusters)

    @property
    def km_model(self) -> KMeans:
        return self.kmeans_model
    
    @km_model.setter
    def km_model(self, random_state : int = 42, n_init : int = 10) -> KMeans:
        return KMeans(n_clusters=self.num_clusters, random_state=random_state, n_init=n_init)
    
    @property
    def hierarchical_model(self) -> AgglomerativeClustering:
        return self.hierarchical_model
    
    @hierarchical_model.setter
    def hierarchical_model(self, n_clusters: int = None) -> AgglomerativeClustering:
        n_clusters = self.num_clusters if n_clusters is None else n_clusters

        return AgglomerativeClustering(n_clusters=n_clusters)
    
    @property
    def gmm_model(self) -> GaussianMixture:
        return self.gmm_model
    
    @gmm_model.setter
    def gmm_model(self, n_components: int = None, random_state: int = 42) -> GaussianMixture:
        n_components = self.num_clusters if n_components is None else n_components

        return GaussianMixture(n_components=n_components, random_state=random_state)
    
    def _hierarchical_clustering(self) -> np.ndarray:
        labels = self.hierarchical_model.fit_predict(self.df)

        num_unique_labels = len(set(labels))
        if num_unique_labels > 1 and num_unique_labels < len(self.df):
            score = silhouette_score(self.df, labels)
            print(f"Hierarchical Clustering Silhouette Score: {score}")
        else:
            print("Silhouette Score non calculable (nombre de clusters incorrect).")

        return labels

    def _train_model(self, model, *args, **kwargs) -> np.ndarray:
        model.fit(self.df, *args, **kwargs)
        labels = model.predict(self.df) if hasattr(model, 'predict') else model.labels_

        if 1 < len(set(labels)) < len(self.df):
            score = silhouette_score(self.df, labels)
            print(f"{model.__class__.__name__} Silhouette Score: {score}")
        else: print("Silhouette Score non calculable (nombre de clusters incorrect).")

        return labels
    
    @abc.abstractmethod
    def _plot_elbow_curve(self,
                          distortions: list,
                          xlabel : str = "Number of clusters",
                          ylabel : str = "Distortion") -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def _run_clustering(self) -> None:
        raise NotImplementedError
    
class PreProcessing(DataClustering):
    def __init__(self, scaler: Union[StandardScaler, MinMaxScaler], 
                 reducer: Union[PCA, TSNE, TruncatedSVD, Isomap]) -> None:
        self.scaler = scaler
        self.reducer = reducer

    def _processing_matrix_data(self) -> list:
        col100g = [col for col in self.df.columns if "_100g" in col]
        data_100g = self.df[col100g]

        corr_matrix = data_100g.corr().abs()

        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]
        data_100g = data_100g.drop(columns=to_drop)

        return data_100g
    
class GMMFeature(DataClustering):
    __init__ = DataClustering.__init__
    
    def _train_gmm(self, *args, **kwargs) -> np.ndarray:
        return self._train_model(self.gmm_model, *args, **kwargs)
    
    @log_action("🔍 Optimize GMM")
    def _optimize_gmm(self, max_clusters: int = 10, best_score: float = -1, best_k: int = 2) -> int:
        max_clusters = min(max_clusters , len(self.df) - 1) 

        for k in range(2 , max_clusters +1 ):
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(self.df)
            labels = gmm.predict(self.df)

            num_unique_labels = len(set(labels))
            if num_unique_labels > 1 and num_unique_labels < len(self.df):
                score = silhouette_score(self.df, labels)
                if score > best_score: 
                    best_score = score
                    best_k = k
            self.num_clusters = best_k
            print(f"Optimal number of clusters for GMM: {best_k}")
            return best_k
        
# class DBSCANFeature(DataClustering):
#     __init__ = DataClustering.__init__

#     def _train_dbscan(self, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
#         dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#         return self._train_model(dbscan)
    
#     @log_action("🔍 Optimize DBSCAN")
#     def _optimize_dbscan(self, eps_range: list = None, min_samples_range: list = None) -> tuple:
#         if eps_range is None:
#             eps_range = np.linspace(0.1, 2.0, 10)
#         if min_samples_range is None:
#             min_samples_range = range(2, 10)

#         best_score, best_eps, best_min_samples = -1, None, None

#         for eps in eps_range:
#             for min_samples in min_samples_range:
#                 dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#                 labels = dbscan.fit_predict(self.df)

#                 num_unique_labels = len(set(labels)) - (1 if -1 in labels else 0)
#                 if num_unique_labels > 1 and num_unique_labels < len(self.df):
#                     score = silhouette_score(self.df, labels)
#                     if score > best_score:
#                         best_score, best_eps, best_min_samples = score, eps, min_samples

#         print(f"Optimal DBSCAN parameters -> eps: {best_eps}, min_samples: {best_min_samples}")
#         return best_eps, best_min_samples

class KmeansFeature(PreProcessing):
    def __init__(self, df, num_clusters, scaler, reducer):
        super().__init__(df, num_clusters)
        self.preprocessing = PreProcessing(scaler, reducer)

    def _train_kmeans(self, *args, **kwargs) -> np.ndarray:
        return self._train_model(self.kmeans_model, *args, **kwargs)
    
    @log_action("🔍 Optimize KMeans")
    def _optimize_kmeans(self, max_clusters: int = 10, distortions : list = []) -> int:

        for _ in range(1, max_clusters + 1):
            self.kmeans_model.fit(self.df)
            distortions.append(self.kmeans_model.inertia_)
            
        optimal_clusters = 3 if len(distortions) < 3 else np.argmin(np.gradient(np.gradient(distortions)))
        print(f"Optimal number of clusters (approx): {optimal_clusters}")
        return optimal_clusters

    def _quantile_result(self, data) -> pd.DataFrame:
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return lower_bound, upper_bound

    @log_action("🔍 KMeans Clustering")
    def kmeans(self, n_clusters : int = 5, random_state : int = 42) -> pd.DataFrame:
        data_100g = self._processing_matrix_data()
        self.data100 = data_100g

        lower_bound, upper_bound = self._quantile_result(self.data100)

        self.data100 = self.data100[~((self.data100 < lower_bound) | (self.data100 > upper_bound)).any(axis=1)]

        scaled_data_100g = self.scaler.fit_transform(self.data100)

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

        self.data100.loc[:, "cluster"] = kmeans.fit_predict(scaled_data_100g)

        print("Inertie du modèle K-Means : ", kmeans.inertia_)

        print(f"Cluster labels: {sorted(set(self.data100['cluster']))}")
        print(f"Distribution des points par cluster: \n{self.data100['cluster'].value_counts()}")

        return self.data100
    
    def find_clusters_elbow(self, max_clusters: int = 10) -> None:
        inertias = []
        scaled_data_100g = self.scaler.fit_transform(self.data100)

        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(scaled_data_100g)
            inertias.append(kmeans.inertia_)

        plt.figure(figsize=(8, 6))

        plt.plot(range(1, max_clusters + 1), inertias, marker='o', linestyle='--', color='b')
        plt.title('Elbow Method for Optimal Clusters')

        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    def plot_kmeans_clusters(self, n_components : int = 5) -> None:

        lower_bound, upper_bound = self._quantile_result(self.data100)

        self.data100 = self.data100[~((self.data100 < lower_bound) | (self.data100 > upper_bound)).any(axis=1)]

        scaled_data_100g = self.scaler.fit_transform(self.data100)

        pca = self.reducer(n_components=n_components)
        pca_scores = pca.fit_transform(scaled_data_100g)

        self.data100['pca1'] = pca_scores[:, 0]
        self.data100['pca2'] = pca_scores[:, 1]

        plt.figure(figsize=(8, 6))

        sns.scatterplot(x=self.data100['pca1'], y=self.data100['pca2'], hue=self.data100["cluster"], palette="Set1", alpha=0.7)

        plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)

        plt.xlabel("Composante principale 1")
        plt.ylabel("Composante principale 2")

        plt.title(f"Nuage de points des données après ACP et K-Means")

        plt.show()   

    @log_action("🔍 Run Clustering")
    def _run_clustering(self) -> None:
        self.kmeans()
        self.find_clusters_elbow()
        self.plot_kmeans_clusters()

class DataFeaturing(KmeansFeature, GMMFeature):
    pass
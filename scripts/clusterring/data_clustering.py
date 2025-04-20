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
        if not col100g:
            raise ValueError("Aucune colonne avec '_100g' trouvée. Vérifiez vos données.")
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
    def __init__(self, file_path, num_clusters=10, scaler=StandardScaler(), reducer=PCA):
        # Charge les données et filtre les colonnes numériques avec _100g
        df = load_data(file_path)
        numeric_cols = [col for col in df.columns 
                       if '_100g' in col and pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_cols:
            raise ValueError("Aucune colonne numérique avec '_100g' trouvée")
            
        self.original_df = df.copy()
        self.df = df[numeric_cols].dropna()
        self.num_clusters = num_clusters
        self.scaler = scaler
        self.reducer = reducer
        self.kmeans_model = KMeans(n_clusters=num_clusters)
        self.data100 = None  # Sera initialisé dans _processing_matrix_data

    def _processing_matrix_data(self) -> pd.DataFrame:
        """Prétraitement des données et suppression des colonnes corrélées"""
        if self.df.empty:
            raise ValueError("Le DataFrame est vide après le filtrage initial")
        
        # Suppression des colonnes avec trop de valeurs manquantes
        data = self.df.dropna(axis=1, thresh=0.7*len(self.df))
        
        if data.empty:
            raise ValueError("Toutes les colonnes ont été supprimées lors du dropna")
            
        # Suppression des colonnes trop corrélées
        corr_matrix = data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.8)]
        data = data.drop(columns=to_drop)
        
        return data

    def _train_kmeans(self, *args, **kwargs) -> np.ndarray:
        """Entraîne le modèle KMeans"""
        return self._train_model(self.kmeans_model, *args, **kwargs)
    
    def _quantile_result(self, data: pd.DataFrame) -> tuple:
        """Calcule les bornes pour la détection des outliers"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        return (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    @log_action("🔍 KMeans Clustering")
    def kmeans(self, n_clusters: int = 5, random_state: int = 42) -> pd.DataFrame:
        """Exécute le clustering KMeans complet"""
        try:
            self.data100 = self._processing_matrix_data()
            print(f"Données après prétraitement : {self.data100.shape}")
            
            if self.data100.empty:
                print("⚠️ Aucune donnée valide pour le clustering")
                return pd.DataFrame()
                
            # Suppression des outliers
            lower, upper = self._quantile_result(self.data100)
            mask = ~((self.data100 < lower) | (self.data100 > upper)).any(axis=1)
            self.data100 = self.data100[mask]
            
            if self.data100.empty:
                print("⚠️ Toutes les données ont été filtrées comme outliers")
                return pd.DataFrame()
                
            # Normalisation
            scaled_data = self.scaler.fit_transform(self.data100)
            
            # Clustering
            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state)
            self.data100['cluster'] = self.kmeans_model.fit_predict(scaled_data)
            
            print(f"Résultats du clustering :\n{self.data100['cluster'].value_counts()}")
            return self.data100
            
        except Exception as e:
            print(f"❌ Erreur lors du clustering : {str(e)}")
            return pd.DataFrame()

    def find_clusters_elbow(self, max_clusters: int = 10) -> None:
        """Méthode du coude pour déterminer le nombre optimal de clusters"""
        if self.data100 is None or self.data100.empty:
            print("⚠️ Aucune donnée disponible pour la méthode du coude")
            return
            
        scaled_data = self.scaler.fit_transform(self.data100)
        inertias = []
        
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)
            
        self._plot_elbow_curve(inertias)

    def plot_kmeans_clusters(self, n_components: int = 2) -> None:
        """Visualisation des clusters après réduction de dimension"""
        if self.data100 is None or self.data100.empty:
            print("⚠️ Aucune donnée disponible pour la visualisation")
            return
            
        scaled_data = self.scaler.fit_transform(self.data100)
        
        # Réduction de dimension
        reduced_data = self.reducer(n_components=n_components).fit_transform(scaled_data)
        
        # Création du plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                             c=self.data100['cluster'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title("Visualisation des clusters après réduction de dimension")
        plt.xlabel("Composante 1")
        plt.ylabel("Composante 2")
        plt.show()

    def _plot_elbow_curve(self, distortions: list, 
                         xlabel: str = "Nombre de clusters",
                         ylabel: str = "Inertie") -> None:
        """Trace la courbe du coude"""
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(distortions)+1), distortions, 'bx-')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Méthode du Coude pour K-Means')
        plt.grid(True)
        plt.show()

    @log_action("🔍 Run Clustering")
    def _run_clustering(self) -> None:
        """Exécute le pipeline complet de clustering"""
        try:
            # Étape 1: Clustering KMeans
            clustered_data = self.kmeans()
            
            if clustered_data.empty:
                return
                
            # Étape 2: Méthode du coude
            self.find_clusters_elbow()
            
            # Étape 3: Visualisation
            self.plot_kmeans_clusters()
            
        except Exception as e:
            print(f"❌ Erreur dans le pipeline de clustering : {str(e)}")

class DataFeaturing(KmeansFeature, GMMFeature):
    pass
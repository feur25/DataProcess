# Projet : Clustering et Analyse des Données OpenFoodFacts

## Description du projet

Ce projet a pour objectif d’analyser les données du dataset OpenFoodFacts en appliquant des méthodes de clustering afin d’identifier des groupes de produits similaires. L’analyse comprend le prétraitement des données, l’application de modèles de clustering (K-Means et DBSCAN) et l’évaluation des résultats avec différentes métriques.

## Pipeline du projet

Le projet suit plusieurs étapes, détaillées ci-dessous :

1. **Prétraitement des données**
   - Chargement des données
   - Sélection des colonnes pertinentes
   - Nettoyage des valeurs aberrantes
   - Normalisation des données
   - Sélection des outliers

2. **Encodage et Scaling**
   - Encodage des données catégorielles (**One-Hot Encoding, TF-IDF, Hashing Encoding, Count Encoding**)
   - Mise à l’échelle des données (**Min-Max, StandardScaler, RobustScaler**)

3. **Réduction de dimension**
   - Application de techniques comme **PCA, t-SNE, UMAP** pour réduire la complexité des données

4. **Application du clustering**
   - Clustering avec **K-Means** et **DBSCAN**
   - Comparaison des clusters formés

5. **Évaluation des résultats**
   - Visualisation des clusters
   - Calcul des métriques :
     - **Score de silhouette**
     - **Davies-Bouldin Index**
     - **Calinski-Harabasz Score**
     - **Inertie (pour K-Means)**
     - **Taux de bruit (pour DBSCAN)**
     - **Stabilité des clusters**

6. **Interprétation des clusters**
   - Analyse des groupes identifiés
   - Sélection des clusters les plus pertinents

## Répartition des tâches

Étape | Tâches | Responsable :

0 Pipeline | Définition d'une pipeline pour le projet https://miro.com/app/board/uXjVIL1M2KA=/?share_link_id=706493141722 | Quentin, Valentin, Tom
1 Prétraitement des données | Nettoyage, normalisation, sélection des variables | Tout le monde (code récupéré des tâches d'un peu tout le monde)
2 Encodage et Scaling | Transformation des variables catégorielles et normalisation | Quentin, Tom & Jules (code des tâches de surtout Quentin)
3 Réduction de dimension | Implémentation de PCA, t-SNE, UMAP | Quentin, Zineb et Valentin (code des tâches de surtout Quentin)
4 Clustering | Application de K-Means et DBSCAN | Zineb & Valentin
5 Évaluation des résultats | Calcul des métriques et visualisation des clusters | Jules & Tom
6 Interprétation des clusters | Analyse des résultats et documentation | Jules & Tom

## Technologies utilisées

- **Python (pandas, numpy, scikit-learn, seaborn, matplotlib)**
- **Jupyter Notebook**
- **Scikit-learn** pour les algorithmes de clustering et les métriques d’évaluation
- **Matplotlib/Seaborn** pour la visualisation des résultats

## Steps to start working on the project

### 0. Initialize conda

with powershell (for widows users) : 

```zsh
conda init powershell
```

or bash-like shells (linux & macOS users):

```zsh
conda init
```
### 1. Create a new virtual environment and activate it 
with pyenv virtualenv:
```bash
	pyenv virtualenv clustering_OFF && pyenv activate clustering_OFF
```
or conda:
```bash
	conda clustering_OFF && conda activate clustering_OFF
```
### 2. Install required librairies 

```bash
  	pip install -r requirements
```

While you work on the project, don't forget to update the`requirements.txt`

### 3. Start coding
You can look at `notebooks/project_starter.ipynb` for your first steps : 
- load data set 
#   D a t a P r o c e s s  
 #   D a t a P r o c e s s  
 #   D a t a P r o c e s s  
 
import argparse
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .utilities.data_processing import ApplyFunctionThread

from .encoder.data_encoder import FeatureEncoder
from .scaler.data_scaling import FeatureScaler
from .visualization.univariate_visualization import AdvancedVisualization
from .cleanner.data_missing_values import AdvancedDataFrameProcessor
from .processing.dimensionality_reduction_pca import DimensionalityReduction
from .cleanner.data_outliers import OutlierDetection
from .clusterring.data_clustering import KmeansFeature

__all__ = [
    'DataFrameProcessor',
    'ApplyFunctionThread',
    'FeatureEncoder',
    'FeatureScaler',
    'AdvancedVisualization',
    'OutlierDetection',
    'DimensionalityReduction',
    'KmeansFeature'
]

__version__ = '1.0.0'
__author__ = 'Feurking'

def pandas_display() -> str:
    return """⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⡗⠀⠀⠉⠑⢢⣴⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡶⢢⠀⠀⠀⠀⠀⠀⠀⠀⠻⠋⠀⠀⠀⠀⠀⢿⣿⠿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠳⢝⢧⡀⠀⠀⠀⠀⠀⢠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣤⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⣄⠙⠄⠀⠀⠀⢀⠇⢠⡄⠀⠀⣄⡀⠀⠀⠀⣴⣄⠘⡀⠀⠀⠀⠀⠀⠀⠀⣀⣤⡴⠶⢚⣫⣵⡶⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⣿⡅⣤⣀⠃⠀⠘⠀⢿⠇⠀⠸⣿⣿⡄⠀⠀⣿⣿⣷⣷⣄⠀⠀⠀⠀⣀⣸⣿⡀⠸⣿⣿⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⠛⠁⣹⣿⣷⣦⣤⣶⡄⢀⣀⠀⠙⠋⠁⣠⣾⣿⣿⣿⣿⣿⣿⣾⣿⣿⣿⣿⣿⡟⢰⣿⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢷⣾⣿⣿⣿⣿⣿⣿⣿⣌⣋⣀⣀⣠⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣈⣯⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠿⠿⣿⣿⣿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⡿⠟⠿⣿⣿⠿⠛⠁⠀⠀⠀⠀⠈⢿⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡗⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⣦⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣿⣿⣿⣿⣦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⣀⣠⣾⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣿⣿⣿⣿⡿⠛⠒⠚⠀⠀⠙⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢼⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⢿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠛⠛⠛⠋⠉⠀⠀⠀⠀⠀⠀⠀⠈⠙⠻⠿⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀Feurking⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀"""

def main(description: str, parser: argparse.ArgumentParser) -> argparse.Namespace:
    print(description, " - Version:", __version__)


    parser.add_argument("--function", type=int, default=0, required=True, help="0 for PCA, 1 for Outliers, 2 for Visualization, 3 for Scaling, 4 for Encoding, 5 for Data Cleaning, 6 for Clustering")
    parser.add_argument("--file_path", type=str, required=True, help="Chemin vers le fichier CSV")
    parser.add_argument("--limit", type=int, default=1000, help="Limite de lignes à traiter")

    parser.add_argument("--output_path", type=str, default="cleaned_data.csv", help="Chemin du fichier de sortie")
    parser.add_argument("--missing_threshold", type=float, default=0.5, help="Seuil de valeurs manquantes pour supprimer une colonne")
    parser.add_argument("--pattern_col", type=str, default="serving_size", help="Colonne pour extraire un motif particulier")
    parser.add_argument("--pattern", type=str, default="(\\d+)", help="Motif regex à extraire")
    parser.add_argument("--irrelevant_cols", type=str, nargs='+', default=["id", "unwanted_col"], help="Colonnes non pertinentes")
    parser.add_argument("--method", type=str, nargs='+', default=["col1", "col2"], help="Colonnes à encoder")
    parser.add_argument("--target_col", type=str, help="Colonne cible pour certains encodages")
    parser.add_argument("--option", type=int, default="0", help="Méthode d'imputation")
    parser.add_argument("--value", type=int, default="0", help="Valeur pour l'imputation")

    return parser.parse_args()

def get_all_columns(df) -> list:
    return df.columns.tolist()

def pars_args_data_pca(args: list) -> None:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--file_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--target_col", type=str, required=True, help="Target column for PCA")
    parser.add_argument("--method", type=str, required=True, help="Dimensionality reduction method")

    args = parser.parse_args(args)

    if not args.function == 0: return

    method = ""

    if isinstance(args.method, list):
        method = args.method[0]
    else: method = args.method

    processor = DimensionalityReduction(args.file_path, args.target_col if not args.target_col else "price", method=method)

    processor.scale_data()
    processor.optimize_n_components()
    processor.apply_reduction()
    processor.explained_variance()

    selected_features = processor.select_features()
    print(f"Most influential features per principal component ({method}): {selected_features}")

def pars_args_data_outliers(args: list) -> None:
    parser = argparse.ArgumentParser()
    args = main("Data Outliers Script", parser)

    if not args.function == 1: return

    processor = OutlierDetection(args.file_path)

    if not args.value:
        processor.run_outlier_analysis() if not args.method else processor.run_outlier_analysis(strategy=args.method[0])

    if args.method and args.value:
        processor.handle_outliers(col=args.method[0], strategy=args.method[1], value=args.value)

    print(f"✅ Cleaned data saved to {args.output_path}")

def pars_args_data_visualization(args: list) -> None:
    parser = argparse.ArgumentParser()
    args = main("Data Univariate Visualization Script", parser)

    if not args.function == 2: return

    processor = AdvancedVisualization(args.file_path)

    processor.run_univariate_analysis()

    print(f"✅ Cleaned data saved to {args.output_path}")

def pars_args_data_scaling(args: list) -> None:
    parser = argparse.ArgumentParser()
    args = main("Data Scaling Script", parser)

    if not args.function == 3: return

    scaler = FeatureScaler.from_csv(args.file_path, limit=1000)

    scaled_df = scaler.scale_based_on_distribution(args.option) if args.option else scaler.scale_based_on_distribution()

    scaled_df.to_csv(args.output_path, index=False, encoding='utf-8')

    print(f"✅ Scaled data saved to {args.output_path}")


def pars_args_data_encoder(args: list) -> None:
    parser = argparse.ArgumentParser()
    args = main("Data Encoding Script", parser)

    if not args.function == 4: return

    encoder = FeatureEncoder.from_csv(args.file_path, args.target_col, limit=1000)

    encoder.encoding_selected(args.method[0])

    encoder.save_encoded_df(args.output_path)

    print(f"✅ Encoded data saved to {args.output_path}")

def pars_args_data_frame_processor(args: list) -> None:
    parser = argparse.ArgumentParser()
    args = main("Data Cleaning Script", parser)

    if not args.function == 5: return

    processor = AdvancedDataFrameProcessor(args.file_path, args.limit)

    processor.impute_missing_values(method='linear_regression')

    output_path = args.output_path
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "cleaned_data.csv")

    processor.df.to_csv(output_path, index=False)

    print(f"✅ Cleaned data saved to {args.output_path}")

def pars_args_data_clustering(args: list) -> None:
    parser = argparse.ArgumentParser()
    args = main("Data Clustering Script", parser)

    if not args.function == 6: 
        return

    processor = KmeansFeature(args.file_path, 10)  # Passe juste le file_path maintenant
    processor._run_clustering()
    print(f"✅ Cleaned data saved to {args.output_path}")
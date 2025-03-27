"""
This module processes command-line arguments and invokes the appropriate function.
"""

import sys
# pars_args_data_pca, pars_args_data_outliers, pars_args_data_visualization, pars_args_data_scaling, pars_args_data_encoder, pars_args_data_clustering
from . import pars_args_data_frame_processor

if __name__ == "__main__":
    # pars_args_data_pca(sys.argv[1:])
    # pars_args_data_outliers(sys.argv[1:])
    # pars_args_data_visualization(sys.argv[1:])
    # pars_args_data_scaling(sys.argv[1:])
    # pars_args_data_encoder(sys.argv[1:])
    pars_args_data_frame_processor(sys.argv[1:])
    # pars_args_data_clustering(sys.argv[1:])
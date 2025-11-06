import argparse
import torch
import json
import time
import os
import numpy as np
from run_eval_ffm_stock import eval, get_model_api  
from fincast.data_tools.TSdataset import find_files_with_suffix


def average_results(results_list):
    keys = results_list[0].keys()
    avg_result = {}
    for key in keys:
        if isinstance(results_list[0][key], (int, float)):
            avg_result[key] = np.mean([r[key] for r in results_list])
    return avg_result


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for multiple datasets")
    parser.add_argument('--dataset_dir', required=True,
                        help='directory for dataset name')
    #dataset loading mode
    parser.add_argument('--data_mode', type=int, default=1, help="stock : 1, stock_test_norm_val_test : 0")
    parser.add_argument("--train_pct", type=float, default=0.7, help="for datamode 0")
    parser.add_argument("--test_pct", type=float, default=0.2, help="for datamode 0")
    #model
    parser.add_argument('--forecast_mode', type=str, default='mean', help=" Literal[mean, median], either mean output or 0.5 in the quantile head region")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_experts', type=int, required=True)
    parser.add_argument('--gating_top_n', type=int, required=True)
    parser.add_argument('--load_from_compile', action='store_true')
    #forecast arguments
    parser.add_argument('--context_len', type=int, default=128)
    parser.add_argument('--horizon_len', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--datetime_col', type=str, default="date")
    parser.add_argument('--ts_cols', nargs='*', default=None,
                    help="List of time-series feature column names.")
    parser.add_argument('--normalize', action='store_true',
                    help="Apply normalization if set.")
    parser.add_argument('--num_cov_cols', nargs='*', default=None,
                    help="List of numerical covariate column names.")
    parser.add_argument('--cat_cov_cols', nargs='*', default=None,
                    help="List of categorical covariate column names.")
    #result
    parser.add_argument('--result_dir', type=str, default="./results/long_horizon")
    parser.add_argument('--logging', type=int, default=0)
    parser.add_argument('--logging_name', type=str, default="multi_eval")
    parser.add_argument('--run_id', default=None, help="run id for each run")



    args = parser.parse_args()

    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.set_float32_matmul_precision('high')
    # torch.cuda.set_per_process_memory_fraction(0.3, device=0)  #memory reserve ratio cap for each process

    print(f"TF32 Allow Matmul : {torch.backends.cuda.matmul.allow_tf32}")
    print(f"TF32 Allow Convolution : {torch.backends.cudnn.allow_tf32}")
    print(f"CUDNN Benchmark Enabled : {torch.backends.cudnn.benchmark}")
    print(f"CUDA Version : {torch.version.cuda}")
    print(f"Torch Compile Active : {torch._dynamo.config.verbose}")
    print(f"Float32 matmul precision mode: {torch.get_float32_matmul_precision()}")
    

    #init model for inference
    model = get_model_api(args)

    results_all = []

    dataset_list = find_files_with_suffix(args.dataset_dir, suffix='.csv')

    for dataset in dataset_list:
        print(f"\n🧪 Running evaluation for dataset: {dataset}")
        config = argparse.Namespace(**vars(args))  # Shallow copy
        config.dataset = dataset

        if config.run_id is None:
            config.run_id = int(time.time())

        eval(config, model)



if __name__ == "__main__":
    main()

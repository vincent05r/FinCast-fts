# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Eval pipeline."""

import json
import os
import sys
import time
from absl import flags, app #type: ignore
import numpy as np
import pandas as pd

from fincast import ffm
from fincast.ffm import data_loader, FFmHparams

import torch
import tqdm

import argparse
import logging
from fincast.tools.utils import log_model_statistics, make_logging_file 
from fincast.tools.model_utils import get_model_FFM, plot_predictions 

from Freq_map_eval import Freq_map_dict
from fincast.data_tools.TSdataset import freq_reader

FLAGS = flags.FLAGS



QUANTILES = list(np.arange(1, 10) / 10.0)
EPS = 1e-7


def get_forecasts(model, past, freq):
  """Get forecasts. add for median and quantiile output supports"""
 
  
  lfreq = [freq] * past.shape[0]
  out, full_output = model.forecast(list(past), lfreq)


  return out


def _mse(y_pred, y_true):
  """mse loss."""
  return np.square(y_pred - y_true)


def _mae(y_pred, y_true):
  """mae loss."""
  return np.abs(y_pred - y_true)


def _smape(y_pred, y_true):
  """_smape loss."""
  abs_diff = np.abs(y_pred - y_true)
  abs_val = (np.abs(y_true) + np.abs(y_pred)) / 2
  abs_val = np.where(abs_val > EPS, abs_val, 1.0)
  abs_diff = np.where(abs_val > EPS, abs_diff, 0.0)
  return abs_diff / abs_val


def get_model_api(config):
  model_path = config.model_path

  ffm_hparams = FFmHparams(
      backend="gpu",
      per_core_batch_size=32,
      horizon_len=config.horizon_len,  #variable
      context_len=config.context_len,  # Context length can be anything up to 2048 in multiples of 32
      use_positional_embedding=False,
      num_experts=config.num_experts,
      gating_top_n=config.gating_top_n,
      load_from_compile=config.load_from_compile,
      point_forecast_mode=config.forecast_mode,
  )


  model_actual, ffm_config, ffm_api = get_model_FFM(model_path, ffm_hparams)

  ffm_api.model_eval_mode()

  log_model_statistics(model_actual)

  return ffm_api


def eval(config, model):
  """Eval pipeline."""

  dataset = os.path.basename(config.dataset)
  data_path = config.dataset
  fm = Freq_map_dict()
  freq = data_path.split('_')[-1].split('.')[0]
  if freq == '1wk':
     freq = '1w'
  print("Freq str : {}, Forecasting mode : {}".format(freq, config.forecast_mode))
  int_freq = freq_reader(data_path, fm.universal_map, mode=1)
  boundaries_2 = pd.read_csv(open(data_path, "r")).shape[0]

  data_df = pd.read_csv(open(data_path, "r"))

  if config.ts_cols is not None:
    raise NotImplementedError("todo")
    # ts_cols = DATA_DICT[dataset]["ts_cols"]
    # num_cov_cols = DATA_DICT[dataset]["num_cov_cols"]
    # cat_cov_cols = DATA_DICT[dataset]["cat_cov_cols"]
  else:
    ts_cols = [col for col in data_df.columns if col != config.datetime_col]
    num_cov_cols = None
    cat_cov_cols = None
  batch_size = min(config.batch_size, len(ts_cols))
  if config.data_mode == 0:
    total_length = pd.read_csv(open(data_path, "r")).shape[0]
    train_boundaries = int(total_length * config.train_pct)
    test_boundaries = int(total_length * (1-config.test_pct))
    
    dtl = data_loader.TimeSeriesdata(
        data_path=data_path,
        datetime_col=config.datetime_col,
        num_cov_cols=num_cov_cols,
        cat_cov_cols=cat_cov_cols,
        ts_cols=np.array(ts_cols),
        train_range=[0, train_boundaries],
        val_range=[train_boundaries, test_boundaries],
        test_range=[test_boundaries, total_length],
        hist_len=config.context_len,
        pred_len=config.horizon_len,
        batch_size=batch_size,
        freq=freq,
        normalize=config.normalize,
        epoch_len=None,
        holiday=False,
        permute=False,
    )

    eval_itr = dtl.tf_dataset(mode="test",
                        shift=1).as_numpy_iterator()


  elif config.data_mode == 1:
    dtl = data_loader.TimeSeriesdata(
        data_path=data_path,
        datetime_col=config.datetime_col,
        num_cov_cols=num_cov_cols,
        cat_cov_cols=cat_cov_cols,
        ts_cols=np.array(ts_cols),
        train_range=[0, boundaries_2],
        val_range=[0, boundaries_2],
        test_range=[config.context_len, boundaries_2],
        hist_len=config.context_len,
        pred_len=config.horizon_len,
        batch_size=batch_size,
        freq=freq,
        normalize=config.normalize,
        epoch_len=None,
        holiday=False,
        permute=False,
    )
    
    eval_itr = dtl.tf_dataset(mode="test",
                            shift=config.horizon_len).as_numpy_iterator()
  


  model_path = config.model_path

  # model = timesfm.TimesFm(
  #     hparams=timesfm.TimesFmHparams(
  #         backend="gpu",
  #         per_core_batch_size=32,
  #         horizon_len=128,
  #         num_layers=50,
  #         context_len=_CONTEXT_LEN.value,
  #         use_positional_embedding=False,
  #     ),
  #     checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_path),
  # )

  #set up and get ffm model

  if config.logging == 1:

      log_dir = 'logs/longeval'

      new_log_filename = make_logging_file(log_dir, config.logging_name)

      logging.basicConfig(
      filename=new_log_filename,  # Logs will be written to this file
      level=logging.INFO,
      format="%(asctime)s - %(levelname)s - %(message)s",
      force=True
      )


  model = model


  smape_run_losses = []
  mse_run_losses = []
  mae_run_losses = []

  num_elements = 0
  abs_sum = 0
  start_time = time.time()

  for batch in tqdm.tqdm(eval_itr):
    past = batch[0]
    actuals = batch[3]
    forecasts = get_forecasts(model, past, int_freq)
    forecasts = forecasts[:, 0:actuals.shape[1]]
    mae_run_losses.append(_mae(forecasts, actuals).sum())
    mse_run_losses.append(_mse(forecasts, actuals).sum())
    smape_run_losses.append(_smape(forecasts, actuals).sum())
    num_elements += actuals.shape[0] * actuals.shape[1]
    abs_sum += np.abs(actuals).sum()


  class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        else:
            return super(NumpyEncoder, self).default(obj)

  if num_elements != 0:

    mse_val = np.sum(mse_run_losses) / num_elements
    
    result_dict = {
        "mse": mse_val,
        "smape": np.sum(smape_run_losses) / num_elements,
        "mae": np.sum(mae_run_losses) / num_elements,
        "wape": np.sum(mae_run_losses) / abs_sum,
        "nrmse": np.sqrt(mse_val) / (abs_sum / num_elements),
        "num_elements": num_elements,
        "abs_sum": abs_sum,
        "total_time": time.time() - start_time,
        "model_path": model_path,
        "dataset": dataset,
        "freq": freq,
        "pred_len": config.horizon_len,
        "context_len": config.context_len,
    }
    run_id = config.run_id
    result_sv_name = "{}_{}_h{}_id{}".format(os.path.basename(model_path).split('.')[0], dataset, config.horizon_len, run_id)
    save_path = os.path.join(config.result_dir, result_sv_name)
    print(f"Saving results to {save_path}", flush=True)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "results.json"), "w") as f:
      json.dump(result_dict, f, cls=NumpyEncoder)
    print(result_dict, flush=True)
    logging.info("Result dictionary: %s", result_dict)
  
  else:
    print("no data for this set")


if __name__ == "__main__":
  FLAGS = flags.FLAGS
  FLAGS(sys.argv, known_only=True)

  parser = argparse.ArgumentParser(description='FFM_longeval')

  #dataset loading mode
  parser.add_argument('--data_mode', type=int, default=1, help="stock : 1, stock_test_norm_val_test : 0")
  parser.add_argument("--train_pct", type=float, default=0.7, help="for datamode 0")
  parser.add_argument("--test_pct", type=float, default=0.2, help="for datamode 0")
  #data
  parser.add_argument('--dataset', type=str, default="etth1")
  parser.add_argument('--horizon_len', type=int, default=96)
  parser.add_argument('--context_len', type=int, default=128)
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

  #logging
  parser.add_argument('--logging', type=int, default=1, help='1 = logging, 0 = not logging')
  parser.add_argument('--logging_name', type=str, default='exp')
  parser.add_argument('--result_dir', type=str, default="./results/long_horizon")

  #model
  parser.add_argument('--forecast_mode', type=str, default='mean', help=" Literal[mean, median], either mean output or 0.5 in the quantile head region")
  parser.add_argument('--model_path', type=str, required=True)
  parser.add_argument('--num_experts', type=int, required=True)
  parser.add_argument('--gating_top_n', type=int, required=True)
  parser.add_argument('--load_from_compile', action='store_true', help='strip _orig_mod. in compile state dict, might have issue')


  config = parser.parse_args()

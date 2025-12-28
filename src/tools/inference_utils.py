import os
from os import path
from typing import List, Optional, Tuple, Union, Dict, Any
from types import SimpleNamespace

import logging
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


#modules
from ffm import FFM, FFmHparams

from ffm.pytorch_patched_decoder_MOE import PatchedTimeSeriesDecoder_MOE

from tools.utils import log_model_statistics, make_logging_file 
from tools.model_utils import get_model_FFM

from data_tools.Inference_dataset import *
from contextlib import nullcontext



def _slice_to_horizon(arr, H):
    """Ensure the time dimension is exactly horizon H."""
    if arr is None:
        return None
    if arr.ndim == 2:
        # [N, T]
        return arr[:, -H:] if arr.shape[1] != H else arr
    if arr.ndim == 3:
        # [N, T, D]
        return arr[:, -H:, :] if arr.shape[1] != H else arr
    raise ValueError(f"Unexpected array shape for horizon slicing: {arr.shape}")

def _pick_last_window_indices(mapping_df):
    """
    Return a numpy array of row indices that correspond to the LAST window for each series,
    robust for both sliding_windows=True/False.
    """
    if mapping_df is None or mapping_df.empty:
        return np.array([], dtype=int)
    if not {"series_idx", "window_end"}.issubset(mapping_df.columns):
        # Fallback: assume already 1 per series in order.
        return np.arange(len(mapping_df), dtype=int)
    # one index per series: the row where window_end is maximal
    idx = mapping_df.groupby("series_idx")["window_end"].idxmax().to_numpy()
    # keep original CSV column order if present
    return idx[np.argsort(mapping_df.loc[idx, "series_idx"].to_numpy())]

def _validate_quantile_requests(full_all, requested_qs):
    """
    Map requested quantile numbers [1..9] to depth indices in full_all[..., D],
    assuming convention D = 1 + Q (mean at depth 0, q1..qQ at 1..Q).
    Returns a list of valid (q, depth_index) pairs preserving request order.
    """
    if full_all is None or full_all.ndim != 3:
        return []
    D = full_all.shape[2]
    Q = D - 1
    out = []
    for q in requested_qs:
        if isinstance(q, (int, np.integer)) and 1 <= q <= Q:
            out.append((int(q), int(q)))  # depth index = q (0 is mean)
    return out

def _save_outputs_to_csv(mean_all, full_all, mapping_df, save_dir, prefix="fincast"):
    """
    Save mean and (if present) full outputs to CSV.
    - mean CSV columns: mean_t+1 ... mean_t+H
    - full CSV columns: mean_t+*, q1_t+*, ..., q9_t+* (only up to available Q)
    Rows are per series (series_name if available, else series_idx).
    Returns (path_mean_csv, path_full_csv or None).
    """
    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    # Determine row labels
    if mapping_df is not None and "series_name" in mapping_df.columns:
        row_names = mapping_df["series_name"].tolist()
    elif mapping_df is not None and "series_idx" in mapping_df.columns:
        row_names = [f"series_{i}" for i in mapping_df["series_idx"].tolist()]
    else:
        row_names = [f"series_{i}" for i in range(mean_all.shape[0])]

    # Mean file
    H = mean_all.shape[1]
    mean_cols = [f"mean_t+{k+1}" for k in range(H)]
    df_mean = pd.DataFrame(mean_all, columns=mean_cols, index=row_names)
    p_mean = os.path.join(save_dir, f"{prefix}_mean_{ts}.csv")
    df_mean.to_csv(p_mean)

    # Full file (if available)
    p_full = None
    if full_all is not None:
        D = full_all.shape[2]
        Q = D - 1
        stats = ["mean"] + [f"q{q}" for q in range(1, Q+1)]
        cols = [f"{s}_t+{k+1}" for s in stats for k in range(H)]
        flat = np.concatenate(
            [full_all[:, :, 0]] + [full_all[:, :, d] for d in range(1, D)],
            axis=1  # [N, H] concat over stats -> [N, H*(1+Q)]
        )
        df_full = pd.DataFrame(flat, columns=cols, index=row_names)
        p_full = os.path.join(save_dir, f"{prefix}_full_{ts}.csv")
        df_full.to_csv(p_full)

    return p_mean, p_full

def plot_last_outputs(
    fincast_inference,
    mean_all,          # preds from run_inference (np.ndarray [N, H’])
    mapping_df,        # mapping from run_inference (pd.DataFrame)
    full_all=None,     # full_outputs from run_inference (np.ndarray [N, H’, D]) or None
    config=None
):
    """
    For each selected column, plot the corresponding input context (last L points)
    and the last forecast outputs (mean + optional quantiles).
    Only one plot per column (the LAST window), even if sliding windows were used.
    """
    ds = fincast_inference.inference_dataset
    L  = ds.L
    H  = int(getattr(config, "horizon_len", mean_all.shape[1]))
    # Slice to pure horizon if model returned on-context forecasts
    mean_all = _slice_to_horizon(mean_all, H)
    full_all = _slice_to_horizon(full_all, H)

    # pick last-per-series indices
    pick_idx = _pick_last_window_indices(mapping_df)
    if pick_idx.size == 0:
        print("Nothing to plot: mapping is empty.")
        return

    mean_sel = mean_all[pick_idx, :]
    full_sel = None if full_all is None else full_all[pick_idx, :, :]

    # Requested quantiles (e.g., [1,3,5,9]) -> depth indices
    req_qs = getattr(config, "plt_quantiles", [])
    qmap   = _validate_quantile_requests(full_sel, req_qs)  # list of (q, depth_index)

    # Iterate series in the same order as dataset’s selected columns
    for row_i, i in enumerate(pick_idx):
        meta  = mapping_df.iloc[i].to_dict() if mapping_df is not None else {}
        sidx  = meta.get("series_idx", row_i)
        sname = meta.get("series_name", f"series_{sidx}")

        # context (corresponding input)
        ctx = ds.series_arrays[sidx][-L:]  # last L input values

        # outputs
        y_mean = mean_sel[row_i, :]  # [H]

        # --- Plot ---
        plt.figure(figsize=(10, 4))
        x_ctx = np.arange(L)
        x_fut = np.arange(L, L + H)

        plt.plot(x_ctx, ctx, label="context", linewidth=1.5)
        plt.plot(x_fut, y_mean, label="forecast (mean)", linewidth=2)

        # Optional quantiles
        if qmap:
            for (q, d) in qmap:
                yq = full_sel[row_i, :, d]  # [H]
                plt.plot(x_fut, yq, linestyle="--", linewidth=1.3, label=f"q{q}")

        plt.title(f"{sname}  |  L={L}, H={H}")
        plt.xlabel("Time (relative index)")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    # Optional: save numeric outputs
    if getattr(config, "save_output", False):
        save_dir = getattr(config, "save_output_path", ".")
        mean_csv, full_csv = _save_outputs_to_csv(mean_sel, full_sel, mapping_df.iloc[pick_idx], save_dir, prefix="fincast")
        print(f"[saved] mean -> {mean_csv}")
        if full_csv is not None:
            print(f"[saved] full  -> {full_csv}")

















class FinCast_Inference:


    def __init__(
            self,
            config: SimpleNamespace,
            use_df: bool = False,
            ):
        
        self.config = config

        self.inference_freq = freq_reader_inference(config.data_frequency)

        if use_df:
            self.inference_dataset = TimeSeriesDataset_SinglePandasDF_Inference(
                df=self.config.df,
                context_length=config.context_len,
                freq_type=self.inference_freq,
                columns=config.columns_target,
                first_c_date=True,
                series_norm=config.series_norm,
                dropna=getattr(config, "dropna", True),
                sliding_windows=config.all_data,
                return_meta=True,
                df_name="in_memory_df",
            )

        else:
            self.inference_dataset = TimeSeriesDataset_SingleCSV_Inference(
                csv_path=self.config.data_path,
                context_length=self.config.context_len,
                freq_type=self.inference_freq,
                columns=self.config.columns_target,
                first_c_date=True,
                series_norm=self.config.series_norm,
                dropna=getattr(self.config, "dropna", True),
                sliding_windows=self.config.all_data,
                return_meta=True,                                   # CHANGE: enable mapping output
                )
        



        if self.config.model_version.lower() == "v1":
            self.config.num_experts = 4
            self.config.gating_top_n = 2
            self.config.load_from_compile = True

        self.model_api = get_model_api(self.config)


    
    # -------- Internal: robust DataLoader, safe with num_workers=0 --------
    def _make_inference_loader(
        self,
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,
    ) -> DataLoader:
        dl_kwargs = dict(
            dataset=self.inference_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
            collate_fn=collate_with_optional_meta,
        )
        # === CHANGE: only add prefetch_factor if workers>0 (PyTorch restriction) ===
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(**dl_kwargs)


    # -------- Integrated inference runner (last-window or sliding) --------
    # Uses the dataset and model kept in this instance, returns preds + mapping DataFrame
    def run_inference(
        self,
        num_workers: int = 2,
    ):
        """
        Runs inference over `self.inference_dataset` and returns:
          mean_all : np.ndarray of shape [N, H']        # point (mean) forecasts
          mapping_df : pandas.DataFrame or None         # per-row metadata (if return_meta=True)
          full_all : np.ndarray of shape [N, H', D]     # full outputs (mean + quantiles), D = 1 + Q
        Notes:
          - If your decoder uses return_forecast_on_context=True, H' = (context_len - patch_len) + horizon_len.
          - If you need horizon-only later, slice [-self.config.horizon_len:] on axis=1.
        """
        ds = self.inference_dataset
        if not getattr(ds, "return_meta", False):
            raise ValueError("The dataset must be constructed with return_meta=True.")

        loader = self._make_inference_loader(
            batch_size=self.config.batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        model = self.model_api

        out_collector: List[np.ndarray] = []
        full_collector: List[np.ndarray] = []
        mapping_rows: List[Dict[str, Any]] = []

        def _to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            elif isinstance(x, np.ndarray):
                return x
            else:
                return np.asarray(x)

        out_T = None  # expected H' for mean
        full_T = None  # expected H' for full
        full_D = None  # expected D = 1 + Q

        with torch.inference_mode():
            for x_ctx, x_pad, freq, _x_fut, meta in loader:
                preds = get_forecasts_f(model, x_ctx, freq=freq)

                # Unpack (mean, full)
                if isinstance(preds, tuple) and len(preds) == 2:
                    out_pred, full_pred = preds
                else:
                    # Backward-compat: if get_forecasts_f returns only mean
                    out_pred, full_pred = preds, None

                out_np = _to_numpy(out_pred)
                if out_np.ndim == 1:
                    # Rare: [H'] -> [B=1, H']
                    out_np = out_np[None, :]
                if out_np.ndim != 2:
                    raise ValueError(f"Expected mean_pred to be [B, H'], got {out_np.shape}")

                # Shape consistency check across batches (mean)
                if out_T is None:
                    out_T = out_np.shape[1]
                elif out_np.shape[1] != out_T:
                    raise ValueError(
                        f"Inconsistent time length for mean across batches: "
                        f"got {out_np.shape[1]} vs expected {out_T}. "
                        f"Check that context_len/patch_len/horizon_len are constant, "
                        f"or slice to horizon-only before collecting."
                    )

                out_collector.append(out_np)

                # Full output (mean+quantiles)
                if full_pred is not None:
                    full_np = _to_numpy(full_pred)
                    if full_np.ndim == 2:
                        # Rare: [H', D] -> [B=1, H', D]
                        full_np = full_np[None, :, :]
                    if full_np.ndim != 3:
                        raise ValueError(f"Expected full_pred to be [B, H', D], got {full_np.shape}")

                    # Shape consistency check across batches (full)
                    if full_T is None:
                        full_T = full_np.shape[1]
                        full_D = full_np.shape[2]
                    else:
                        if full_np.shape[1] != full_T:
                            raise ValueError(
                                f"Inconsistent time length for full across batches: "
                                f"got {full_np.shape[1]} vs expected {full_T}. "
                                f"Align return_forecast_on_context or slice consistently."
                            )
                        if full_np.shape[2] != full_D:
                            raise ValueError(
                                f"Inconsistent output depth D (1+Q) across batches: "
                                f"got {full_np.shape[2]} vs expected {full_D}. "
                                f"Ensure quantile set is fixed."
                            )

                    full_collector.append(full_np)

                # Meta aligns with batch dim
                mapping_rows.extend(meta)

        out_all = (
            np.concatenate(out_collector, axis=0)
            if out_collector else np.empty((0, 0), dtype=np.float32)
        )
        full_all = (
            np.concatenate(full_collector, axis=0)
            if full_collector else None
        )
        mapping_df = pd.DataFrame(mapping_rows) if mapping_rows else None

        return out_all, mapping_df, full_all











def collate_with_optional_meta(batch: List[Tuple]):
    """
    Works for both sample formats:
      - (x_ctx, x_pad, freq, x_fut)
      - (x_ctx, x_pad, freq, x_fut, meta)

    Returns:
      x_ctx: [B, L]          # squeezed (was [B, L, 1])
      x_pad: [B, L] or [B, L+H] depending on dataset; squeezed if 3D
      freq : [B, 1]          # int64
      x_fut: [B, 0] or [B, H] (squeezed if 3D)
      meta : list[dict] | None
    """
    has_meta = (len(batch[0]) == 5)

    x_ctx = torch.stack([b[0] for b in batch], dim=0)
    x_pad = torch.stack([b[1] for b in batch], dim=0)
    freq  = torch.stack([b[2] for b in batch], dim=0)
    x_fut = torch.stack([b[3] for b in batch], dim=0)
    meta  = [b[4] for b in batch] if has_meta else None

    # Helper: squeeze trailing singleton channel if present
    def _squeeze_last(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.size(-1) == 1:
            return x.squeeze(-1)  # [B, L, 1] -> [B, L]
        return x

    x_ctx = _squeeze_last(x_ctx)
    x_pad = _squeeze_last(x_pad)
    x_fut = _squeeze_last(x_fut)

    # Normalise freq to [B, 1] int64
    if freq.dim() == 1:
        freq = freq.view(-1, 1)
    elif freq.dim() == 2 and freq.size(1) != 1:
        # Conservative fallback if dataset returns extra dims
        freq = freq[:, :1]
    freq = freq.long()

    # Quick invariants to catch shape regressions early
    assert x_ctx.dim() == 2, f"x_ctx must be [B,L], got {tuple(x_ctx.shape)}"
    assert freq.dim() == 2 and freq.size(1) == 1, f"freq must be [B,1], got {tuple(freq.shape)}"

    return x_ctx, x_pad, freq, x_fut, meta































def freq_reader_inference(freq: str) -> int:

    freq = str.upper(freq)
    if freq.endswith("MS"):
        return 1
    elif freq.endswith(("H", "T", "MIN", "D", "B", "U", "S")):
        return 0
    elif freq.endswith(("W", "M")):
        return 1
    elif freq.endswith(("Y", "Q", "A")):
        return 2
    else:
        print("Invalid frequency, default to fast freq: 0")
        return 0






def get_forecasts_f(model, past, freq):
    """Get forecasts. add for median and quantiile output supports"""


    lfreq = [freq] * past.shape[0]
    out, full_output = model.forecast(list(past), lfreq)


    return out, full_output





def get_model_api(config):
  model_path = config.model_path

  ffm_hparams = FFmHparams(
      backend=config.backend,
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




























## legacy code



def plot_predictions_multi(
    model: FFM,
    val_dataset: Dataset,
    number_img: int = 5,
    model_version: str = 'exp',
    save_dir: Optional[str] = None,
    save_name: Optional[str] = "predictions",
    moe_n: int = 0,
    moe_tk: int = 0,
    quantile: int = 0,
    output_length: int = 128,
) -> None:
    """
    Plot model predictions against ground truth for a batch of validation data.

    Args:
        quantile: 0 is the mean, 1 to 9 for 9 quantile.
        model: Trained TimesFM model
        val_dataset: Validation dataset
        save_path: Path to save the plot
    """

    model.eval()
    device = next(model.parameters()).device

    for plt_i in range(number_img):
        x_context, x_padding, freq, x_future = val_dataset[plt_i]
        x_context = x_context.unsqueeze(0)  # Add batch dimension
        x_padding = x_padding.unsqueeze(0)
        freq = freq.unsqueeze(0)
        x_future = x_future.unsqueeze(0)

        x_context = x_context.to(device)
        x_padding = x_padding.to(device)
        freq = freq.to(device)
        x_future = x_future.to(device)

        with torch.no_grad():
            predictions, total_aux_loss = model(x_context, x_padding.float(), freq)
            predictions_mean = predictions[..., quantile]  # [B, N, horizon_len]
            last_patch_pred = predictions_mean[:, -1, :]  # [B, horizon_len]

        context_vals = x_context[0].cpu().numpy()
        future_vals = x_future[0, 0:output_length].cpu().numpy()
        pred_vals = last_patch_pred[0, 0:output_length].cpu().numpy()

        context_len = len(context_vals)
        horizon_len = len(future_vals)

        plt.figure(figsize=(12, 6))

        if quantile == 0:
            quantile_str = "mean"
        else:
            quantile_str = quantile

        plt.plot(range(context_len),
                context_vals,
                label="Historical Data",
                color="blue",
                linewidth=2)

        plt.plot(
            range(context_len, context_len + horizon_len),
            future_vals,
            label="Ground Truth",
            color="green",
            linestyle="--",
            linewidth=2,
        )

        plt.plot(range(context_len, context_len + horizon_len),
                pred_vals,
                label="Prediction",
                color="red",
                linewidth=2)

        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title("FFM_{}_M{}_T{} Predictions vs Ground Truth\ndistribution on {}".format(model_version, moe_n, moe_tk, quantile_str))
        plt.legend()
        plt.grid(True)

        plt.show()

        if save_dir != None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, '{}_{}.jpg'.format(save_name, plt_i))
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

        plt.close()



def plot_predictions_multi_distribution(
    model: FFM,
    val_dataset: Dataset,
    number_img: int = 5,
    model_version: str = 'exp',
    save_dir: Optional[str] = None,
    save_name: Optional[str] = "predictions",
    moe_n: int = 0,
    moe_tk: int = 0,
    quantiles: Optional[List[int]] = None,  # Now optional list
    output_length: int = 128,
    output_number: bool = False,
) -> None:
    """
    Plot model predictions (mean + optional quantiles) against ground truth.

    Args:
        model: Trained FFM model
        val_dataset: Validation dataset
        number_img: Number of plots to generate
        model_version: For title metadata
        save_dir: If specified, saves plots
        save_name: Base name for saved files
        moe_n, moe_tk: MoE config for title
        quantiles: Optional list of quantile indices (1–9). Mean (index 0) is always plotted.
        output_length: Forecast horizon length
    """
    model.eval()
    device = next(model.parameters()).device

    if quantiles is None:
        quantiles = []

    for plt_i in range(number_img):
        x_context, x_padding, freq, x_future = val_dataset[plt_i]
        x_context = x_context.unsqueeze(0).to(device)
        x_padding = x_padding.unsqueeze(0).to(device)
        freq = freq.unsqueeze(0).to(device)
        x_future = x_future.unsqueeze(0).to(device)

        with torch.no_grad():
            predictions, _ = model(x_context, x_padding.float(), freq)
            # predictions: [B, N, horizon_len, Q]
            last_step_preds = predictions[:, -1, :, :]  # [B, horizon_len, Q]

            # Extract mean forecast (index 0)
            mean_pred = last_step_preds[0, :output_length, 0].cpu().numpy()

            context_vals = x_context[0].cpu().numpy()
            future_vals = x_future[0, :output_length].cpu().numpy()
            context_len = len(context_vals)
            horizon_len = len(future_vals)

            if output_number:
                print('output values')
                print(context_vals)
                print(future_vals)

            plt.figure(figsize=(12, 6))

            # Historical input
            plt.plot(range(context_len), context_vals, label="Historical", color="blue", linewidth=2)

            # Ground truth
            plt.plot(
                range(context_len, context_len + horizon_len),
                future_vals,
                label="Ground Truth",
                color="green",
                linestyle="--",
                linewidth=2,
            )

            # Mean forecast
            plt.plot(
                range(context_len, context_len + horizon_len),
                mean_pred,
                label="Prediction (mean)",
                color="red",
                linewidth=2,
            )

            # Optional quantile forecasts
            for q in quantiles:
                if q <= 0:
                    continue  # Skip mean index if redundantly included
                quantile_pred = last_step_preds[0, :output_length, q].cpu().numpy()
                plt.plot(
                    range(context_len, context_len + horizon_len),
                    quantile_pred,
                    label=f"Prediction (q{q})",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                )

            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.title(f"FFM_{model_version}_M{moe_n}_T{moe_tk} Forecast\nQuantiles={quantiles}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'{save_name}_{plt_i}.jpg')
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")

            plt.show()
            plt.close()



def plot_predictions_multi_distribution_v2(
    model: FFM,
    val_dataset: Dataset,
    number_img: int = 5,
    model_version: str = 'exp',
    save_dir: Optional[str] = None,
    save_name: Optional[str] = "predictions",
    moe_n: int = 0,
    moe_tk: int = 0,
    quantiles: Optional[List[int]] = None,  # Now optional list
    output_length: int = 128,
    output_number: bool = False,
) -> None:
    """
    Plot model predictions (mean + optional quantiles) against ground truth.

    Args:
        model: Trained FFM model
        val_dataset: Validation dataset
        number_img: Number of plots to generate
        model_version: For title metadata
        save_dir: If specified, saves plots
        save_name: Base name for saved files
        moe_n, moe_tk: MoE config for title
        quantiles: Optional list of quantile indices (1–9). Mean (index 0) is always plotted.
        output_length: Forecast horizon length
    """
    model.eval()
    device = next(model.parameters()).device

    if quantiles is None:
        quantiles = []

    for plt_i in range(number_img):
        x_context, x_padding, freq, x_future = val_dataset[plt_i]
        x_context = x_context.unsqueeze(0).to(device)
        x_padding = x_padding.unsqueeze(0).to(device)
        freq = freq.unsqueeze(0).to(device)
        x_future = x_future.unsqueeze(0).to(device)

        with torch.no_grad():
            predictions, _ = model(x_context, x_padding.float(), freq)
            # predictions: [B, N, horizon_len, Q]
            last_step_preds = predictions[:, -1, :, :]  # [B, horizon_len, Q]

            # Extract mean forecast (index 0)
            mean_pred = last_step_preds[0, :output_length, 0].cpu().numpy()

            context_vals = x_context[0].cpu().numpy()
            future_vals = x_future[0, :output_length].cpu().numpy()
            context_len = len(context_vals)
            horizon_len = len(future_vals)

            if output_number:
                print('output values')
                print(context_vals)
                print(future_vals)

            plt.figure(figsize=(12, 6))

            # Concatenated ground truth (historical + future)
            gt_full = np.concatenate([context_vals, future_vals], axis=0)
            plt.plot(
                range(context_len + horizon_len),
                gt_full,
                label="Ground Truth",
                color="blue",
                linewidth=2,
            )


            # Mean forecast
            plt.plot(
                range(context_len, context_len + horizon_len),
                mean_pred,
                label="Forecast (point)",
                color="red",
                linewidth=3,
            )

            # Optional quantile forecasts
            for q in quantiles:
                if q <= 0:
                    continue  # Skip mean index if redundantly included
                quantile_pred = last_step_preds[0, :output_length, q].cpu().numpy()
                plt.plot(
                    range(context_len, context_len + horizon_len),
                    quantile_pred,
                    label=f"Quantile (Q{q})",
                    linestyle="--",
                    linewidth=2.5,
                    alpha=0.8,
                )

            #plt.xlabel("Time Step")
            #plt.ylabel("Value")
            #plt.title(f"FFM_{model_version}_M{moe_n}_T{moe_tk} Forecast\nQuantiles={quantiles}")
            plt.legend(fontsize=15)
            plt.grid(True)
            plt.tight_layout()

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'{save_name}_{plt_i}.jpg')
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")

            plt.show()
            plt.close()
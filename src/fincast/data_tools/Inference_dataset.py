from typing import List, Optional, Tuple, Union, Dict, Any


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader





class TimeSeriesDataset_SingleCSV_Inference(Dataset):
    """
    Inference-only dataset for a SINGLE CSV.

    Output signature (training-compatible by default):
        (x_context, x_padding, freq, x_future)

    Shapes/dtypes:
        x_context: [L, 1], float32
        x_padding: [L, 1], float32 (zeros; no masking at inference)
        freq:      [1],   int64
        x_future:  [0, 1], float32 (empty; no ground truth at inference)

    Modes:
      - sliding_windows=False: one sample per selected column (the LAST L values).
      - sliding_windows=True:  stride=1 windows across the series for each column.

    NEW (for plotting / mapping):
      - return_meta (bool): if True, __getitem__ returns a 5th item: a dict with
        fields that identify which column/window the sample comes from.
        Default False to preserve training-time compatibility.
    """

    def __init__(
        self,
        csv_path: str,
        context_length: int,
        freq_type: int,
        columns: Optional[List[Union[int, str]]] = None,
        first_c_date: bool = True,
        series_norm: bool = False,
        dropna: bool = True,
        sliding_windows: bool = False,
        return_meta: bool = False,  # NEW
    ):
        super().__init__()
        if context_length <= 0:
            raise ValueError("context_length must be positive.")
        self.csv_path = csv_path
        self.L = int(context_length)
        self.freq_type = int(freq_type)
        self.first_c_date = bool(first_c_date)
        self.series_norm = bool(series_norm)
        self.dropna = bool(dropna)
        self.sliding_windows = bool(sliding_windows)
        self.return_meta = bool(return_meta)

        # ---- Load CSV
        df = pd.read_csv(csv_path)

        # ---- Resolve columns to use
        if columns is None:
            start_idx = 1 if self.first_c_date else 0
            use_cols = list(df.columns[start_idx:])
        else:
            use_cols = []
            ncols = len(df.columns)
            for c in columns:
                if isinstance(c, int):
                    if c < 0 or c >= ncols:
                        raise IndexError(
                            f"Column index {c} out of range [0, {ncols-1}] for CSV '{csv_path}'."
                        )
                    use_cols.append(df.columns[c])
                elif isinstance(c, str):
                    if c not in df.columns:
                        raise KeyError(f"Column '{c}' not found in CSV '{csv_path}'.")
                    use_cols.append(c)
                else:
                    raise TypeError("columns entries must be int indices or str names.")

        # Optional row-wise NaN drop (prevents misalignment)
        if self.dropna:
            df = df.dropna(axis=0).reset_index(drop=True)

        # ---- Build per-series numeric arrays + names/indices (for meta)
        self.series_arrays: List[np.ndarray] = []
        self.series_names: List[str] = []            # NEW
        self.series_col_indices: List[int] = []      # NEW

        for name in use_cols:
            col_idx = df.columns.get_loc(name)
            s = pd.to_numeric(df[name], errors="coerce")
            arr = s.to_numpy(dtype=np.float32)

            # Ensure numeric and clean
            if self.dropna:
                arr = arr[~np.isnan(arr)]
            else:
                if np.isnan(arr).any():
                    raise ValueError(
                        f"Column '{name}' contains NaNs; set dropna=True or clean prior to loading."
                    )

            if len(arr) < self.L:
                raise ValueError(
                    f"Column '{name}' too short: len={len(arr)} < context_length={self.L}."
                )

            if self.series_norm:
                mu = float(arr.mean())
                sigma = float(arr.std(ddof=0))
                if sigma == 0.0:
                    sigma = 1.0
                arr = (arr - mu) / sigma

            self.series_arrays.append(arr)
            self.series_names.append(str(name))
            self.series_col_indices.append(int(col_idx))

        # ---- Build indices for __getitem__
        # CHANGE (robustness): store index records only when sliding to keep memory light.
        self.index_records: List[Tuple[int, int]] = []  # (series_idx, start_idx)
        if self.sliding_windows:
            for sidx, arr in enumerate(self.series_arrays):
                n = len(arr)
                # stride=1 over full length
                for start in range(0, n - self.L + 1):
                    self.index_records.append((sidx, start))
            self.sample_lengths = [self.L] * len(self.index_records)
        else:
            self.sample_lengths = [self.L] * len(self.series_arrays)

    def __len__(self) -> int:
        return len(self.index_records) if self.sliding_windows else len(self.series_arrays)

    def get_length(self, idx: int) -> int:
        return self.sample_lengths[idx]

    def _make_meta(self, series_idx: int, window_start: int) -> Dict[str, Any]:
        """Build a lightweight metadata dict for plotting & traceability."""
        return {
            "csv_path": self.csv_path,
            "series_idx": int(series_idx),
            "series_name": self.series_names[series_idx],
            "column_idx": self.series_col_indices[series_idx],
            "context_length": int(self.L),
            "freq_type": int(self.freq_type),
            "window_start": int(window_start),
            "window_end": int(window_start + self.L - 1),
        }

    def __getitem__(self, idx: int):
        if self.sliding_windows:
            series_idx, start_idx = self.index_records[idx]
            series = self.series_arrays[series_idx]
            ctx_np = series[start_idx : start_idx + self.L]
            meta = self._make_meta(series_idx, start_idx)
        else:
            series_idx = idx
            series = self.series_arrays[series_idx]
            start_idx = len(series) - self.L
            ctx_np = series[-self.L:]
            meta = self._make_meta(series_idx, start_idx)

        # Tensors
        x_context = torch.as_tensor(ctx_np, dtype=torch.float32).unsqueeze(-1)  # [L,1]
        x_padding = torch.zeros((self.L, 1), dtype=torch.float32)               # [L,1]
        freq = torch.tensor([self.freq_type], dtype=torch.long)                 # [1]
        x_future = torch.empty((0, 1), dtype=torch.float32)                     # [0,1]

        # Optional meta as 5th return value (backwards-compatible)
        if self.return_meta:
            return x_context, x_padding, freq, x_future, meta
        return x_context, x_padding, freq, x_future




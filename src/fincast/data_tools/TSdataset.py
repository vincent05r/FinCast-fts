import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset
import random


import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
import os

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, List, Dict

from sklearn.preprocessing import StandardScaler


def freq_reader(file_path: str, freq_dict: Dict, mode: int = 0) -> int:
    '''out put freq int based on the freq map and file name
        mode 0 is direct conversion
        mode 1 is part of the name match
        '''

    if mode == 1:
        for suffix in sorted(freq_dict.keys(), key=lambda x: -len(x)):
            if file_path.endswith(suffix) and file_path[-len(suffix):] == suffix:
                return freq_dict[suffix]
        raise ValueError(f"No known suffix match found for file: {file_path}")

    elif mode == 0:
        basename = os.path.basename(file_path).split('.')[0].lower()
        if basename not in freq_dict:
            raise KeyError(f"freq_map does not have an entry for: {basename}")
        return freq_dict[basename]


class TimeSeriesDataset_MultiCSV_train_Production(Dataset):
    """
    Dataset for reading multiple CSV files (with multiple columns => multiple univariate series),
    generating sliding windows for training, and returning:
        (x_context, x_padding, freq, x_future)

    Key points to optimize memory usage:
      1. We store only the original time-series as NumPy arrays (self.all_series).
      2. For each valid window, we store only an index record of
         (series_idx, start_idx, context_length, freq_type).
      3. During __getitem__, we slice the appropriate segment on the fly.
    """

    def __init__(
        self,
        csv_paths: List[str],
        horizon_length: int,
        freq_map: Dict[str, int],
        freq_map_mode: int = 0,
        mask_ratio: float = 0.0,
        possible_context_lengths: Dict = None,
        first_c_date: bool = True,
        series_norm: bool = True,
        data_slice_interval: Dict = None,
        shuffle: bool = True,
        shuffle_seed: int = 5,
    ):
        """
        Args:
            csv_paths: List of CSV file paths.
            horizon_length: Number of future timesteps to predict.
            freq_map: Dictionary mapping each CSV base filename to an integer freq type.
            mask_ratio: Fraction of timesteps to mask in x_padding. Value in [0,1].
            possible_context_lengths: Dict map, different freq will have a diffrent list of training range.
            first_c_date: If True, skip the first column for every CSV (matching original code).
            series_norm: norm each series based on the stats of it self, z score norm.
            data_slice_interval: Dict, different slicing interval for different frequency
        """
        super().__init__()

        if not (0.0 <= mask_ratio <= 1.0):
            raise ValueError("mask_ratio must be in [0, 1].")

        # Store constructor args
        self.csv_paths = csv_paths
        self.horizon_length = horizon_length
        self.freq_map = freq_map
        self.freq_map_mode = freq_map_mode
        self.mask_ratio = mask_ratio
        self.first_c_date = first_c_date
        self.series_norm = series_norm
        self.shuffle = shuffle #ddp shuffle, same random seed
        self.shuffle_seed = shuffle_seed

        #slicing interval for different freq
        if data_slice_interval is None:
            data_slice_interval = {
                0 : 4,
                1 : 2,
                2 : 1,
            }
        self.data_slice_interval = data_slice_interval

        # 32..2048 in increments of 32
        if possible_context_lengths is None:
            possible_context_lengths = {
                0 : [512, 128],
                1 : [256, 128],
                2 : [64],
            }
        self.possible_context_lengths = possible_context_lengths

        # 1) Read CSVs and store each column as univariate series
        self.all_series = []  # Each element is (np_array_of_series, freq_type)
        self._read_csvs()

        # 2) Create indexing records for all possible windows:
        #    (series_idx, start_idx, context_length, freq_type)
        self.index_records = []
        self.sample_lengths = []  # Parallel array to store context lengths for each record
        self._prepare_index_records()

    def _read_csvs(self):
        """
        Load each CSV file, parse freq_type from freq_map,
        store each column (minus the first if self.first_c_date=True) as a univariate series.
        """
        for csv_file in self.csv_paths:
            freq_type = freq_reader(file_path=csv_file, freq_dict=self.freq_map, mode=self.freq_map_mode)

            # If self.first_c_date is True, skip the first column for every CSV
            if self.first_c_date:
                start_c_idx = 1
            else:
                start_c_idx = 0

            df = pd.read_csv(csv_file)

            if self.series_norm:
                scaler = StandardScaler()
                cols_to_normalize = df.columns[start_c_idx:]
                df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

            data = df.values  # shape: (num_rows, num_cols)

            # For each column from start_c_idx onward, store univariate series
            for col_idx in range(start_c_idx, data.shape[1]):
                col_data = data[:, col_idx].astype(np.float32)  # shape: [num_rows]
                self.all_series.append((col_data, freq_type))

    def _prepare_index_records(self):
        """
        For each univariate series in self.all_series,
        generate all valid windows for each context length in self.possible_context_lengths,
        and store an index record for slicing on the fly.
        """
        for series_idx, (series_data, freq_type) in enumerate(self.all_series):
            series_len = len(series_data)
            # For each possible context length
            current_context_length_list = self.possible_context_lengths[freq_type]
            for length in current_context_length_list:
                total_window = length + self.horizon_length
                if series_len < total_window:
                    continue

                # Slide over the series
                max_start = series_len - total_window
                slice_step = self.data_slice_interval[freq_type]
                for start_idx in range(0, max_start + 1, slice_step):
                    # Index record references the series array location + freq
                    self.index_records.append((series_idx, start_idx, length, freq_type))
                    self.sample_lengths.append(length)

        # Shuffle once so each epoch sees a random order
        indices = list(range(len(self.index_records)))

        if self.shuffle: #deterministic shuffle
            rng = random.Random(self.shuffle_seed)
            rng.shuffle(indices)

        self.index_records = [self.index_records[i] for i in indices]
        self.sample_lengths = [self.sample_lengths[i] for i in indices]

    def __len__(self) -> int:
        return len(self.index_records)

    def get_length(self, idx: int) -> int:
        """
        Return the 'context length' for the idx-th sample.
        Useful if you have a custom sampler that groups by length.
        """
        return self.sample_lengths[idx]

    def __getitem__(self, idx: int):
        """
        Return one sample: (x_context, x_padding, freq, x_future),
        where:
            x_context shape: [context_length, 1]
            x_padding shape: [context_length, 1]  # 1 => masked, 0 => unmasked
            freq shape: [1]
            x_future shape: [horizon_length, 1]
        """
        series_idx, start_idx, length, freq_type = self.index_records[idx]
        series_data, _ = self.all_series[series_idx]  # We already have freq_type in the record

        # Slice on the fly from the underlying NumPy array
        x_context_np = series_data[start_idx : start_idx + length]
        x_future_np  = series_data[start_idx + length : start_idx + length + self.horizon_length]

        # Convert to PyTorch tensors
        x_context = torch.tensor(x_context_np, dtype=torch.float32).unsqueeze(-1)  # shape: [length, 1]
        x_future  = torch.tensor(x_future_np, dtype=torch.float32).unsqueeze(-1)   # shape: [horizon_length, 1]

        # Create a mask of shape [length, 1], with random fraction = mask_ratio set to 1
        if self.mask_ratio > 0.0:
            mask = torch.rand(size=(length, 1))
            x_padding = (mask < self.mask_ratio).float()
        else:
            x_padding = torch.zeros((length, 1), dtype=torch.float32)

        # freq as a single integer
        freq = torch.tensor([freq_type], dtype=torch.long)

        return x_context, x_padding, freq, x_future





def find_files_with_suffix(directory: str, suffix: str) -> List[str]:
    """
    Recursively traverse the given directory to find all files ending with `suffix`.

    Args:
        directory (str): Path to the directory where the search should begin.
        suffix (str): File suffix (e.g., ".csv", ".txt") to filter by.

    Returns:
        List[str]: List of full file paths that match the given suffix.
    """
    matched_files = []
    for root, dirs, files in os.walk(directory):
        dirs.sort()   # Ensures deterministic traversal of subdirectories
        files.sort()  # Ensures deterministic file ordering
        for f in files:
            if f.endswith(suffix):
                matched_files.append(os.path.join(root, f))
    return matched_files


def csv_nparray(df_path, target: str) -> np.ndarray:
  df = pd.read_csv(df_path)
  uni_timeseries = df[target].values
  return uni_timeseries


# def prepare_dataset_inferencing(series: np.ndarray,
#                      context_length: int,
#                      horizon_length: int,
#                      freq_type: int = 0) -> Dataset:
#   """
#   Prepare Inference dataset from time series data.

#   Args:
#       series: Input time series data
#       context_length: Number of past timesteps to use
#       horizon_length: Number of future timesteps to predict
#       freq_type: Frequency type (0, 1, or 2)
#       train_split: Fraction of data to use for training

#   Returns: TimeSeriesDataset_MOE_M
      
#   """
#   return TimeSeriesDataset_MOE_M(series, context_length=context_length, horizon_length=horizon_length, freq_type=freq_type)

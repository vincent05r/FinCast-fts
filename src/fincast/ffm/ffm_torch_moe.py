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

import logging
from os import path
from typing import Any, Sequence

import numpy as np
import torch
from fincast.ffm import ffm_base
from fincast.ffm import pytorch_patched_decoder_MOE as ppd

_TOL = 1e-6


class FFmTorch(ffm_base.FFmBase):
  """forecast API for inference, modify from timesfm"""

  def __post_init__(self):

    self._model_config = ppd.FFMConfig(
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        hidden_size=self.model_dims,
        intermediate_size=self.model_dims,
        patch_len=self.input_patch_len,
        horizon_len=self.output_patch_len, #tricks here, this is done for inferencing. Since when training hl is not adjustable(last ff defined by it.) This is a hp for decode as well
        head_dim=self.model_dims // self.num_heads,
        quantiles=self.quantiles,
        use_positional_embedding=self.use_pos_emb,
        num_experts=self.num_experts, #moe part
        gating_top_n=self.gating_top_n,
        threshold_train=self.threshold_train,
        threshold_eval=self.threshold_eval,
    )
    self._model = None
    self.num_cores = 1
    self.global_batch_size = self.per_core_batch_size
    self._device = torch.device("cuda" if (
        torch.cuda.is_available() and self.backend == "gpu") else "cpu")
    self._median_index = -1




  def load_from_checkpoint_ffm(self, checkpoint: str) -> None:
    """Loads a checkpoint and compiles the decoder."""

    self._model = ppd.PatchedTimeSeriesDecoder_MOE(self._model_config)

    if self.hparams.load_from_compile: #load from compilled model

      # Load the state dict
      state_dict = torch.load(checkpoint, map_location='cpu', weights_only=True)
      # if keys contain `_orig_mod.`, strip them
      new_state_dict = {}
      for k, v in state_dict.items():
        new_k = k
        # Strip `_orig_mod.module.`
        if new_k.startswith('_orig_mod.module.'):
            new_k = new_k[len('_orig_mod.module.'):]
        # Strip `_orig_mod.`
        elif new_k.startswith('_orig_mod.'):
            new_k = new_k[len('_orig_mod.'):]
        # Strip `module.` (DDP)
        elif new_k.startswith('module.'):
            new_k = new_k[len('module.'):]
        new_state_dict[new_k] = v
      
      logging.info("Loading checkpoint from %s, strict = True", checkpoint)
      self._model.load_state_dict(new_state_dict, strict=True)

    else: #normal loading
      loaded_checkpoint = torch.load(checkpoint, weights_only=True)
      logging.info("Loading checkpoint from %s, strict = True", checkpoint)
      self._model.load_state_dict(loaded_checkpoint, strict=True)

    logging.info("Sending checkpoint to device %s", f"{self._device}")
    self._model.to(self._device)





  def model_eval_mode(self):
    '''set model to nn.module.eval()'''
    self._model.eval()



  def _forecast(
      self,
      inputs: Sequence[Any],
      freq: Sequence[int] | None = None,
      window_size: int | None = None,
      forecast_context_len: int | None = None,
      return_forecast_on_context: bool = False,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Forecasts on a list of time series.

    Args:
      inputs: list of time series forecast contexts. Each context time series
        should be in a format convertible to JTensor by `jnp.array`.
      freq: frequency of each context time series. 0 for high frequency
        (default), 1 for medium, and 2 for low. Notice this is different from
        the `freq` required by `forecast_on_df`.
      window_size: window size of trend + residual decomposition. If None then
        we do not do decomposition.
      forecast_context_len: optional max context length.
      return_forecast_on_context: True to return the forecast on the context
        when available, i.e. after the first input patch.

    Returns:
    A tuple for JTensors:
    - the mean forecast of size (# inputs, # forecast horizon),
    - the full forecast (mean + quantiles) of size
        (# inputs,  # forecast horizon, 1 + # quantiles).

    Raises:
    ValueError: If the checkpoint is not properly loaded.
    """
    if self._model is None:
      raise ValueError("Checkpoint is not properly loaded.")

    if forecast_context_len is None:
      forecast_context_len = self.context_len
    inputs = [np.array(ts)[-forecast_context_len:] for ts in inputs]

    if window_size is not None:
      new_inputs = []
      for ts in inputs:
        new_inputs.extend(ffm_base.moving_average(ts, window_size))
      inputs = new_inputs

    if freq is None:
      logging.info("No frequency provided via `freq`. Default to high (0).")
      freq = [0] * len(inputs)

    input_ts, input_padding, inp_freq, pmap_pad = self._preprocess(inputs, freq)

    with torch.no_grad():
      mean_outputs = []
      full_outputs = []
      for i in range(input_ts.shape[0] // self.global_batch_size):
        t_input_ts = torch.Tensor(input_ts[i * self.global_batch_size:(i + 1) *
                                           self.global_batch_size]).to(
                                               self._device)
        t_input_padding = torch.Tensor(
            input_padding[i * self.global_batch_size:(i + 1) *
                          self.global_batch_size]).to(self._device)
        t_inp_freq = torch.LongTensor(
            inp_freq[i * self.global_batch_size:(i + 1) *
                     self.global_batch_size, :]).to(self._device)

        mean_output, full_output = self._model.decode(
            input_ts=t_input_ts,
            paddings=t_input_padding,
            freq=t_inp_freq,
            horizon_len=self.horizon_len,
            output_patch_len=self.output_patch_len,
            # Returns forecasts on context for parity with the Jax version.
            return_forecast_on_context=True,
        )
        if not return_forecast_on_context:
          mean_output = mean_output[:, self._horizon_start:, ...]
          full_output = full_output[:, self._horizon_start:, ...]

        if self.backend == "gpu":
          mean_output = mean_output.cpu()
          full_output = full_output.cpu()
        mean_output = mean_output.detach().numpy()
        full_output = full_output.detach().numpy()
        mean_outputs.append(mean_output)
        full_outputs.append(full_output)

    mean_outputs = np.concatenate(mean_outputs, axis=0)
    full_outputs = np.concatenate(full_outputs, axis=0)

    if pmap_pad > 0:
      mean_outputs = mean_outputs[:-pmap_pad, ...]
      full_outputs = full_outputs[:-pmap_pad, ...]

    if window_size is not None:
      mean_outputs = mean_outputs[0::2, ...] + mean_outputs[1::2, ...]
      full_outputs = full_outputs[0::2, ...] + full_outputs[1::2, ...]

    return mean_outputs, full_outputs

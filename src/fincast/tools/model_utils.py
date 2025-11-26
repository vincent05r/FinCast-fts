import os
from os import path
from typing import Optional, Tuple

import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

#modules
from fincast.ffm import FFM, FFmHparams







def get_model_FFM(checkpoint : str, hparams : FFmHparams):

    device = "gpu" # Literal["cpu", "gpu", "tpu"] = "gpu"

    ffm = FFM(hparams=hparams, checkpoint=checkpoint, loading_mode=0)

    model = ffm._model

    return model, ffm._model_config, ffm





def plot_predictions(
    model: FFM,
    val_dataset: Dataset,
    save_path: Optional[str] = "predictions.png",
) -> None:
    """
    Plot model predictions against ground truth for a batch of validation data.

    Args:
        model: Trained TimesFM model
        val_dataset: Validation dataset
        save_path: Path to save the plot
    """

    model.eval()

    x_context, x_padding, freq, x_future = val_dataset[0]
    x_context = x_context.unsqueeze(0)  # Add batch dimension
    x_padding = x_padding.unsqueeze(0)
    freq = freq.unsqueeze(0)
    x_future = x_future.unsqueeze(0)

    device = next(model.parameters()).device
    x_context = x_context.to(device)
    x_padding = x_padding.to(device)
    freq = freq.to(device)
    x_future = x_future.to(device)

    with torch.no_grad():
        predictions, total_aux_loss = model(x_context, x_padding.float(), freq)
        predictions_mean = predictions[..., 0]  # [B, N, horizon_len]
        last_patch_pred = predictions_mean[:, -1, :]  # [B, horizon_len]

    context_vals = x_context[0].cpu().numpy()
    future_vals = x_future[0].cpu().numpy()
    pred_vals = last_patch_pred[0].cpu().numpy()

    context_len = len(context_vals)
    horizon_len = len(future_vals)

    plt.figure(figsize=(12, 6))

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
    plt.title("TimesFM Predictions vs Ground Truth")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.close()


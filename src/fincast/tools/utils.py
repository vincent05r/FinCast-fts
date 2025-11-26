import logging
import os
import re


from fincast.ffm.pytorch_patched_decoder_MOE import PatchedTimeSeriesDecoder_MOE



def log_config(config):
    logger = logging.getLogger(__name__)
    logger.info("training config -----------------------------")
    for attr, value in vars(config).items():
        logger.info(f"{attr}: {value}")
    logger.info("------------------------------------")

def make_logging_file(log_dir: str = 'logs/finetune', log_name: str = 'exp'):
        
    log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)  

    # Define a regex pattern to match log files, e.g., log31.log
    pattern = re.compile(r"{}(\d+)\.log".format(log_name))

    # List files in the log directory that match the pattern
    existing_logs = [f for f in os.listdir(log_dir) if pattern.match(f)]
    if existing_logs:
        # Extract numbers from the filenames and find the maximum
        log_max_num = max(int(pattern.match(f).group(1)) for f in existing_logs)
        new_log_num = log_max_num + 1
    else:
        new_log_num = 1

    new_log_filename = os.path.join(log_dir, f"{log_name}{new_log_num}.log")

    return new_log_filename




def log_model_statistics(model: PatchedTimeSeriesDecoder_MOE):
    """
    Logs key statistics of the given PatchedTimeSeriesDecoder_MOE model.

    Args:
        model (PatchedTimeSeriesDecoder_MOE): The decoder model instance.
    """
    # Number of layers taken from the model configuration
    logger = logging.getLogger(__name__)

    num_layers = model.config.num_layers

    # Collect total and trainable parameters
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            trainable_params += numel

    # Approximate model size in megabytes, assuming float32 (4 bytes per param)
    param_size_bytes = total_params * 4
    param_size_mb = param_size_bytes / (1024 ** 2)

    logger.info("\nModel Statistics:")
    logger.info("-----------------")
    logger.info(f"Model Class: {model.__class__.__name__}")
    logger.info(f"Number of Transformer Layers: {num_layers}")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Approx. Model Size (float32): {param_size_mb:.2f} MB")

    # Optionally, log additional config info:
    logger.info("\nAdditional Configuration:")
    logger.info(f"Hidden Size: {model.config.hidden_size}")
    logger.info(f"Intermediate Size: {model.config.intermediate_size}")
    logger.info(f"Number of Heads: {model.config.num_heads}")
    logger.info(f"Number of Experts (MoE): {model.config.num_experts}")
    logger.info(f"Top N gating: {model.config.gating_top_n}")
    logger.info(f"Patch Length: {model.config.patch_len}")
    logger.info(f"Horizon Length: {model.config.horizon_len}")
    logger.info(f"Use Positional Embedding: {model.config.use_positional_embedding}")
    logger.info("-----------------\n")


def print_model_statistics(model: PatchedTimeSeriesDecoder_MOE):
    """
    Prints key statistics of the given PatchedTimeSeriesDecoder_MOE model.

    Args:
        model (PatchedTimeSeriesDecoder_MOE): The decoder model instance.
    """

    num_layers = model.config.num_layers

    # Collect total and trainable parameters
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            trainable_params += numel

    # Approximate model size in megabytes, assuming float32 (4 bytes per param)
    param_size_bytes = total_params * 4
    param_size_mb = param_size_bytes / (1024 ** 2)

    print("\nModel Statistics:")
    print("-----------------")
    print(f"Model Class: {model.__class__.__name__}")
    print(f"Number of Transformer Layers: {num_layers}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Approx. Model Size (float32): {param_size_mb:.2f} MB")

    # Optionally, print additional config info:
    print("\nAdditional Configuration:")
    print(f"Hidden Size: {model.config.hidden_size}")
    print(f"Intermediate Size: {model.config.intermediate_size}")
    print(f"Number of Heads: {model.config.num_heads}")
    print(f"Number of Experts (MoE): {model.config.num_experts}")
    print(f"Top N gating: {model.config.gating_top_n}")
    print(f"Patch Length: {model.config.patch_len}")
    print(f"Horizon Length: {model.config.horizon_len}")
    print(f"Use Positional Embedding: {model.config.use_positional_embedding}")
    print("-----------------\n")

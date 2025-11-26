#!/bin/bash

# Exit immediately on error
set -e

ENV_NAME="fincast_v1"

# Ensure conda is available
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "⚙️ Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"

echo "📦 Installing FinCast with dependencies..."
pip install -e .
#pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118 older version, slow inferencing
pip install -U torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

echo "✅ All packages installed in Conda env '$ENV_NAME'."

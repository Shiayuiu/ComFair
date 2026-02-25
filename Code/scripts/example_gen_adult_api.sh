#!/bin/bash

# -------------------------------
# Configuration
# -------------------------------

# Dataset configuration
DATASET="adult"

# Model configuration
MODEL_TYPE="api"
MODEL_NAME="YOUR_MODEL_NAME"
MODEL_PATH="YOUR_MODEL_IDENTIFIER"
BASE_URL="https://your-api-base-url.com/v1"
API_KEY="YOUR_API_KEY"

# Runtime parameters
BATCH_SIZE=8
# Set DEBUG_N=10 to run only the first 10 samples for testing.
# Set DEBUG_N=0 to process the full dataset.
DEBUG_N=0

# -------------------------------
# Execution
# -------------------------------

echo "========================================================"
echo "Step 1: Model Inference (Generation)"
echo "========================================================"

python run_gen_classification.py \
    --dataset "$DATASET" \
    --model_type "$MODEL_TYPE" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --batch_size $BATCH_SIZE \
    --debug_n $DEBUG_N \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL"

if [ $? -eq 0 ]; then
    echo "========================================================"
    echo "Pipeline Finished Successfully!"
    echo "Check results in: data/results/predictions/"
    echo "========================================================"
else
    echo "Inference failed!"
    exit 1
fi
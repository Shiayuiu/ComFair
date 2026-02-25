#!/bin/bash

# --- Configuration ---

# Dataset path
DATASET="adult"

# Model parameters
MODEL_TYPE="api"
MODEL_NAME="Deepseek-V3.2-chat"
MODEL_PATH="deepseek-chat"
BASE_URL="https://api.anthropic.com/v1" # Replace with your API Base URL
API_KEY=

# Task parameters
BATCH_SIZE=8
DEBUG_N=0 # Set to 10 to run only the first 10 examples for testing; set to 0 to run all

# --- Execution ---
echo "========================================================"
echo "Step 1: Data Preprocessing"
echo "========================================================"
# Run data preparation script (if applicable)

echo "========================================================"
echo "Step 2: Model Inference (Generation)"
echo "========================================================"
# Run generation script
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
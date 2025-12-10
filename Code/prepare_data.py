# prepare_data.py
import os
from src.data.preprocess_adult import process_adult_data
from configs.adult import dataset_config

def main():
    # 1. 确保目录存在
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/results/predictions", exist_ok=True)
    os.makedirs("data/results/reports", exist_ok=True)

    print(">>> [Step 1] Initializing Data Preparation...")
    
    # 2. 检查原始文件是否存在
    if not os.path.exists(dataset_config.raw_data_path):
        raise FileNotFoundError(f"Raw data not found at: {dataset_config.raw_data_path}. Please place 'adult.csv' there.")

    # 3. 运行预处理
    print(f"Processing {dataset_config.dataset_name} dataset...")
    process_adult_data(
        input_path=dataset_config.raw_data_path,
        output_feature_path=dataset_config.processed_feature_path,
        output_meta_path=dataset_config.processed_metadata_path
    )
    print(">>> Data Preparation Complete.")

if __name__ == "__main__":
    main()
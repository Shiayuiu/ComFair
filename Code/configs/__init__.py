# code/configs/__init__.py
import importlib

def get_dataset_config(dataset_name: str):
    """
    根据数据集名称动态加载对应的配置模块。
    假设配置文件都位于 configs/ 目录下，命名为 [dataset_name].py，
    并且其中必须包含一个名为 dataset_config 的实例。
    """
    try:
        # 动态 import，例如 dataset_name="adult" -> import configs.adult
        module = importlib.import_module(f"configs.{dataset_name}")
        
        # 获取模块中的 dataset_config 对象
        if hasattr(module, "dataset_config"):
            return module.dataset_config
        else:
            raise AttributeError(f"Config module 'configs.{dataset_name}' has no attribute 'dataset_config'.")
            
    except ImportError:
        raise ImportError(f"Could not find config file for dataset: '{dataset_name}'. Make sure 'configs/{dataset_name}.py' exists.")
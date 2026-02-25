# code/configs/__init__.py
import importlib

def get_dataset_config(dataset_name: str):
    """
    Dynamically load the configuration module for the given dataset name.
    Assumes config files are located in the configs/ directory named [dataset_name].py
    and that each module contains an instance named `dataset_config`.
    """
    try:
        # Dynamic import, e.g., dataset_name="adult" -> import configs.adult
        module = importlib.import_module(f"configs.{dataset_name}")
        
        # Retrieve the dataset_config object from the module
        if hasattr(module, "dataset_config"):
            return module.dataset_config
        else:
            raise AttributeError(f"Config module 'configs.{dataset_name}' has no attribute 'dataset_config'.")
            
    except ImportError:
        raise ImportError(f"Could not find config file for dataset: '{dataset_name}'. Make sure 'configs/{dataset_name}.py' exists.")
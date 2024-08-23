import json
import logging
import os
import time
import torch

from models import TaggerCNN

def setup_logging(working_dir):
    log_file_path = os.path.join(working_dir, 'training.log')

    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def log_print(message):
    print(message)
    logging.info(message)
    
def generate_model_name(epochs: int, base_model: str | None = None) -> str:
    """
    Generate a unique model name based on current timestamp, base model (if any), classes, number of samples, and epochs.
    """
    result = f"{int(time.time())}"
    if base_model:
        result += f"_b={base_model}"
    
    result += f"_e={epochs}"
    
    return result

def get_model_by_name(device: torch.device, directory: str, name: str) -> tuple[torch.nn.Module, dict[int, str]]:
    """
    Load a model by its name from the specified directory and move it to the specified device.
    Assumes that the number of classes can be parsed from the model name.
    """
    for file in os.listdir(directory):
        if file.startswith(name):
            model_path = os.path.join(directory, file)
            break
    else:
        raise ValueError(f"no model starting with {name} found in {directory}!")
    
    index_to_tag_path = model_path.replace('.pth', '.json')

    if os.path.exists(index_to_tag_path):
        with open(index_to_tag_path, 'r') as f:
            index_to_tag = json.load(f)
            index_to_tag = {int(k): v for k, v in index_to_tag.items()}
    else:
        raise FileNotFoundError(f"no class mappings file found for the model {model_path}!")
    
    num_tags = len(index_to_tag)

    model = TaggerCNN(num_tags)

    model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    
    return model, index_to_tag
    

def get_model_by_latest(device: torch.device, directory: str|None=None) -> tuple[torch.nn.Module, dict[int, str]] | None:
    """
    Load a model whose model name is the latest time from the specified directory and move it to the specified device and return its class mapping.
    """
    if directory and os.path.exists(directory):
        model_files = [f for f in os.listdir(directory) if f.endswith('.pth')]
        if not model_files:
            raise ValueError(f"no model files found in {directory}!")

        latest_model = max(model_files, key=lambda x: int(x.split('_')[0]))
        print(f"latest model: {latest_model}")
        
        model_path = os.path.join(directory, latest_model)

        index_to_tag_path = model_path.replace('.pth', '.json')

        if os.path.exists(index_to_tag_path):
            with open(index_to_tag_path, 'r') as f:
                index_to_tag = json.load(f)
                index_to_tag = {int(k): v for k, v in index_to_tag.items()}
        else:
            raise FileNotFoundError(f"no class mappings file found for the model at {latest_model}!")

        num_tags = len(index_to_tag)

        model = TaggerCNN(num_tags)

        model.load_state_dict(torch.load(model_path, map_location=device))

        model = model.to(device)
        
        return model, index_to_tag
    
    else:
        return None
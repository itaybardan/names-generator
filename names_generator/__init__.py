import importlib.resources
import json
import logging
import torch
from dotmap import DotMap

import names_generator.resources

CONFIG = DotMap(json.loads(importlib.resources.read_text(names_generator.resources.__name__, 'config.json')))
logging.basicConfig(level=CONFIG.log.level, format=CONFIG.log.format)
CONFIG.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with importlib.resources.path(names_generator.resources.__name__, 'small_dataset') as file_path:
    CONFIG.dataset_root_folder = str(file_path)

with importlib.resources.path(names_generator.resources.__name__, 'model.pth') as model_path:
    CONFIG.model_path = str(model_path)

with importlib.resources.path(names_generator.resources.__name__, 'output') as out_path:
    CONFIG.output_path = str(out_path)

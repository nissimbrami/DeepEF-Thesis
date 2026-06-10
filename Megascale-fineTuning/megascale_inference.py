# Inference by epoch
import os
import sys
from validation.validation import run_validation
from analysis.analysis_runner import run_analysis

MODELS_PATH = ['./res/trianed_models-cycle_per2', './res/trianed_models-cycle_per','./res/trianed_models-no_exdu_nosigmoid']
DATA_PATH = './data/Processed_K50_dG_datasets'

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

for base_models in MODELS_PATH:
    print('running validation for all models in: ', base_models)
    for model in os.listdir(base_models):
        print('running validation for model: ', model)
        run_validation(DATA_PATH, model_path=os.path.join(base_models, model))
        

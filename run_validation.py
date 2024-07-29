# This script runs the validation and analysis for all models in the models_list
import os 
import subprocess
import gc
from validation.validation import run_validation
from analysis.analysis_runner import run_analysis



if __name__ == "__main__":
    
    model_list = []
    for i in range(1, 20):
        cycle_norm = 'res/trianed_models-cycle_per_norm/' + str(i) + '_final_model.pt'
        cycle_norm_2 = 'res/trianed_models-cycle_per_2_norm/' + str(i) + '_final_model.pt'
        cycle_norm_SM = 'res/trianed_models-cycle_per_norm_SM/' + str(i) + '_final_model.pt'
        model_list.append(cycle_norm)
        model_list.append(cycle_norm_2)
        model_list.append(cycle_norm_SM)
    
    # run validation for all models
    for model in model_list:
        eval_path ='/'.join(model.split('/')[1:])
        if os.path.exists(eval_path):
            print('model already validated: ', model)
            continue
        try: 
            print('running validation for model: ', model)
            run_validation(r'./data/Processed_K50_dG_datasets',model_path=r'./'+model)
            # clear memory
            gc.collect()
            print('validation done for model: ', model)
            print('running analysis for model: ', model)
            # run_analysis(r'./'+model)
            print('analysis done for model: ', model)
        except Exception as e:
            print('error occured for model: ', model)
            print(e)
            continue
        gc.collect()


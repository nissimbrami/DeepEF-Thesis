from validation.validation import run_validation
from analysis.analysis_runner import run_analysis

if __name__ == "__main__":
    
    # models_list = ['res/trianed_models-noLLMemb', 'res/trianed_models-newDecoys', 'res/trianed_models-nocoords']
    models_list = ['res/trianed_models-noLLMemb']
    # run validation for all models
    for model in models_list:
        run_validation(r'./data/Processed_K50_dG_datasets',model_path=r'./'+model+'/')
        print('validation done for model: ', model)
        run_analysis(r'./'+model+'/')
        print('analysis done for model: ', model)


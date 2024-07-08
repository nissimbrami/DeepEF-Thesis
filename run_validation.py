from validation.validation import run_validation
from analysis.analysis_runner import run_analysis

if __name__ == "__main__":
    
    # trained_models_path = 'res/trianed_models-no_exdu_nosigmoid'
    trained_models_path = './res/trianed_models-cycle_per/4_final_model.pt'
    # models_list = ['res/trianed_models-noLLMemb', 'res/trianed_models-newDecoys', 'res/trianed_models-nocoords','res/trianed_models-numerical_LG]
    # models_list = ['res/trianed_models-unfolded_reg','res/trianed_models-no_exdu','res/trianed_models-MSEgrad']
    models_list = [trained_models_path]
    # models_list = ['Megascale-fineTuning/models/PEM_fine_tuned/49.pt','Megascale-fineTuning/models/PEM_full_trained/38.pt']
    # run validation for all models
    for model in models_list:
        print('running validation for model: ', model)
        run_validation(r'./data/Processed_K50_dG_datasets',model_path=r'./'+model)
        print('validation done for model: ', model)
        print('running analysis for model: ', model)
        run_analysis(r'./'+model)
        print('analysis done for model: ', model)


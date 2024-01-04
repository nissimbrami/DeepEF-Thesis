from validation.validation import run_validation
from analysis.analysis_runner import run_analysis

if __name__ == "__main__":
    run_validation(r'./data/Processed_K50_dG_datasets',model_path=r'./res/trianed_models-newDecoys/')
    print('validation done')
    run_analysis()
    print('analysis done')

#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main                        ### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 2-10:30:00                       ### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name DeePEF_train_no_const_val                       ### name of the job
#SBATCH --output DeePEF_train_no_const_val.out                     ### output log for running job - %J for job number
#SBATCH --gpus=1                                ### number of GPUs, allocating more than 1 requires IT team's permission

# Note: the following 4 lines are commented out
#BATCH --mail-user=shaharax@post.bgu.ac.il        ### user's email for sending job status messages
#SBATCH --mail-type=END,FAIL                   ### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
##SBATCH --mem=24G                              ### ammount of RAM memory, allocating more than 60G requires IT team's permission

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"


module load anaconda ### load anaconda module
source activate esm2_env ### activating Conda environment, environment must be configured before running the job
python train-hydro.py ./trianed_models_no_const/ 0 ### execute python script – replace with your own command
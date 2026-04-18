#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition rtx6000                        ### specify partition name where to run a job.
#SBATCH --time 7-10:30:00                          ### limit the time of job running. Format: D-H:MM:SS
#SBATCH --job-name DeePEF                          ### name of the job
#SBATCH --output DeePEF.out                        ### stdout + stderr log (merged below)
#SBATCH --error DeePEF.out                         ### merge stderr so tqdm/scan progress is visible
#SBATCH --gpus-per-node=4                          ### 4 GPUs per node (requires IT multi-GPU permission)
#SBATCH --cpus-per-task=32                         ### CPU cores: num_workers × n_gpus (8×4=32)
#SBATCH --ntasks-per-node=1                        ### 1 torchrun launcher per node (torchrun spawns ranks)
#SBATCH --qos=keasar

# Note: the following 2 lines are commented out
#SBATCH --mail-user=shaharax@post.bgu.ac.il        ### user's email for sending job status messages
#SBATCH --mail-type=END,FAIL                       ### conditions for sending the email.
##SBATCH --mem=24G                                 ### RAM (>60G requires IT permission)

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"
nvidia-smi

module load anaconda                               ### load anaconda module
source activate esm2_env_py38                      ### PyTorch 2.4.1 — enables torch.compile, torch.amp, torchrun

NGPUS=${SLURM_GPUS_ON_NODE:-1}
echo "Launching with $NGPUS GPU(s)"

# NCCL tuning for clusters without NVLink (RTX 6000 Ada uses PCIe)
export NCCL_DEBUG=INFO                  # log NCCL init details to diagnose failures
export NCCL_P2P_DISABLE=1              # disable GPU-to-GPU P2P (use host memory copies)
export NCCL_IB_DISABLE=1              # disable InfiniBand (use Ethernet/socket)
export NCCL_SOCKET_IFNAME=eth0,ib0    # try Ethernet first, then IB

# torchrun handles DDP rank setup; works transparently with 1 GPU (no dist init)
torchrun --nproc_per_node=$NGPUS --master_addr=localhost --master_port=29500 train.py

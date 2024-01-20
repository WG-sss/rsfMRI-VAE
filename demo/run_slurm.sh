#!/bin/bash
#--------------------------------------------------------------------------------------
#	Created on: 8 Oct. 2022 11:33:26 UTC/GMT +8.00 (Beijing)
#   	Author: Changbo Zhu
#   	E-mail: 1207146285@qq.com
#   Last update: 29th Dec. 2023
#
# USAGE: bash run_slurm.sh slurm.job
#---------------------------------------------------------------------------------------

# Load Personal Environmental Variables
# Not Recommended
# source $HOME/.bashrc
#############################################################################################################
#
#  Define directories
#
WORKDIR=$HOME/rsfMRI-VAE/demo
RES_DIR=$WORKDIR             # Specify the directory of
DATA_LOC=/datacenter/scratch/${USER}/dataset # specify the directory of dataset
#############################################################################################################
#
# Define slurm parameters
#
SLURM_JOB=$1                    # Input slurm.job
JOB_NAME=preprocess             # Specify the name of job.
PARTITION=cpu_short              # Specify partition or queue.  Default: cpu_long
NUM_NODES=1                     # Number of Nodes. Default: 1
NTASKS_PER_NODE=2               # Number of Logic CPU Cores Per node, not Physic CPU Cores.  Default: 2
NTASKS=2                        # Total Number of Logic CPU Cores over all nodes
NODELIST=node04                 # List of nodes, Format: node[01-02],node04. Default: node04
QOS=normal                      # QOS of Service for user.  Default: normal
NUM_GPU_PER_NODE=1              # Number of GPUs per node, only for GPU task.  Default: 0
VARIANT_GPU=gres                # gres or bind. Recommeded: gres.
REQMEM=$((NTASKS_PER_NODE * 8)) #GB            # Required memory, Default: $[NTASKS_PER_NODE*4] GB
MAX_TIME=24:00:00               # Format： Hours:Minutes:Seconds, Default: 24 hours
SLEEP_EPS=0                     # Time Interval between two adjacent submmitions, Default: 0(s)
PARA=(1)                        # List of parameters

##########################################################################################################
#
#		Submit job using sbatch
#
LEN=${#PARA[*]} # Length of parameter list
for idx in $(seq 0 1 $((LEN - 1))); do
	SLEEP_TIME=$((SLEEP_EPS * idx))
	MSG=$RES_DIR/msg
	mkdir -p $MSG
	Error=$MSG/${JOB_NAME}'_para.'${PARA[$idx]}'_slurm.pid'%j'.err'
	Out=$MSG/${JOB_NAME}'_para.'${PARA[$idx]}'_slurm.pid'%j'.out'

	echo 'Submitting job '$JOB_NAME' with para.'${PARA[$idx]}' in '${PARTITION}
	###########################################################################################################################
	# CPU Example
	if [ $PARTITION = "cpu_test" ] || [ $PARTITION = "cpu_short" ] || [ $PARTITION = "cpu_long" ] || [ $PARTITION = "cpu_big" ] || [ $PARTITION = "cpu_super" ]; then
		sbatch --partition=$PARTITION --qos=$QOS -e $Error -o $Out --job-name=${JOB_NAME} --time=${MAX_TIME} \
			--ntasks=${NTASKS} --ntasks-per-node=${NTASKS_PER_NODE} --nodes=$NUM_NODES --nodelist=${NODELIST} --mem=${REQMEM}GB \
			--export=SLEEP_TIME=$SLEEP_TIME,RES_DIR=${RES_DIR},DATA_LOC=${DATA_LOC},WORKDIR=${WORKDIR},para=${PARA[$idx]} $SLURM_JOB
	########################################################################################################################
	# Variant of GPU Example A:  number of GPUs per node (gres=gpu:N)
	elif [ $PARTITION = "gpu_test" ] || [ $PARTITION = "gpu_short" ] || [ $PARTITION = "gpu_long" ] || [ $PARTITION = "gpu_big" ] || [ $PARTITION = "gpu_super" ] && [ $VARIANT_GPU="gres" ]; then
		sbatch --partition=$PARTITION --qos=$QOS -e $Error -o $Out --job-name=${JOB_NAME} --time=${MAX_TIME} \
			--ntasks=${NTASKS} --ntasks-per-node=${NTASKS_PER_NODE} --nodes=$NUM_NODES --nodelist=${NODELIST} --mem=${REQMEM}GB \
			--gres=gpu:${NUM_GPU_PER_NODE} \
			--export=SLEEP_TIME=$SLEEP_TIME,RES_DIR=${RES_DIR},DATA_LOC=${DATA_LOC},WORKDIR=${WORKDIR},para=${PARA[$idx]} $SLURM_JOB
	######################################################################################################################
	# Variant of GPU Example B:  number of GPUs per process AND bind each process to its own GPU (single:<tasks_per_gpu>)
	#elif [ $PARTITION = "gpu_test" ] || [ $PARTITION = "gpu_short" ] || [ $PARTITION = "gpu_long" ] || [ $PARTITION = "gpu_big" ] || [ $PARTITION = "gpu_super" ] && [ $VARIANT_GPU="bind" ]
	#then
	#	sbatch --partition=$PARTITION --qos=$QOS -e $Error -o $Out --job-name=${JOB_NAME} --time=${MAX_TIME} \
	#       --ntasks=${NTASKS} --ntasks-per-node=${NTASKS_PER_NODE} --nodes=$NUM_NODES --nodelist=${NODELIST} --mem=${REQMEM}GB \
	#	    --gpus-per-task=1 --gpu-bind=single:${NUM_GPU_PER_NODE} \
	#	    --export=SLEEP_TIME=$SLEEP_TIME,RES_DIR=${RES_DIR},DATA_LOC=${DATA_LOC},WORKDIR=${WORKDIR},para=${PARA[$idx]} $SLURM_JOB
	fi
done

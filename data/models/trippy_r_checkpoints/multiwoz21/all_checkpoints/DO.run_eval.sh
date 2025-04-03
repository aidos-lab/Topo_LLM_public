#PBS -l select=1:ncpus=2:mem=64gb:ngpus=1:accelerator_model=rtx6000
#PBS -l walltime=47:59:00
#PBS -A "DialSys"
#PBS -q "CUDA"
#PBS -r y
#PBS -e /gpfs/project/ruppik/job_logs
#PBS -o /gpfs/project/ruppik/job_logs
#PBS -N DO.run_eval.sh_trippy_r_checkpoints_multiwoz21

# ======================================================================================================= #
# Notes:
# - PBS cannot submit non-rerunable Array Jobs, thus we need to set the rerunable flag to "-r y" in the PBS header.
#
# - For training, use the following configuration:
# 	- #PBS -l select=1:ncpus=1:mem=32gb:ngpus=1:accelerator_model=a100

contains() {
    local seeking="$1"; shift
    for element; do
        if [[ "$element" == "$seeking" ]]; then
            return 0
        fi
    done
    return 1
}

# Argument Parsing -----------------------------------------------

echo ">>> [INFO] Parsing arguments ..."

DO_TRAINING="False"

for arg in "$@"; do
    case $arg in
        --do-training)
        DO_TRAINING="True"
        shift
        ;;
    esac
done

echo ">>> [INFO] DO_TRAINING: ${DO_TRAINING}"

echo ">>> [INFO] Parsing arguments DONE"

# # # # # # # # # # # # # # # #
# Load environment

echo ">>> [INFO] Loading environment ..."

if [[ $(hostname) == *"hilbert"* || $(hostname) == *"hpc"* ]]; then
	echo ">>> Sourcing .bashrc ..."

	source /gpfs/project/ruppik/.usr_tls/.bashrc
	# source ~/.bashrc

	echo ">>> Sourcing .bashrc DONE"

	echo ">>> Loading environment modules ..."
	
	# Modules ---------------------------------------------------------
	# load_python
	module load Python/3.12.3

	# load_cuda
	module load CUDA/11.7.1

	echo ">>> Loading environment modules DONE"

	export TOPO_LLM_REPOSITORY_BASE_PATH="/gpfs/project/ruppik/git-source/Topo_LLM"
	export CONVLAB3_REPOSITORY_BASE_PATH="/gpfs/project/ruppik/git-source/ConvLab3"

	# >>> Setup in Michael's environment:

	# module load Python/3.8.3
	# module load APEX/0.1

	# export PYTHONPATH=/gpfs/project/$USER/tools/ConvLab3/:$PYTHONPATH
fi

# >>> Setup on my machine:
export PYTHONPATH=/gpfs/project/$USER/git-source/ConvLab3/:$PYTHONPATH

echo ">>> [INFO] Loading environment DONE"

VARIABLES_TO_LOG=(
    "PBS_ARRAY_INDEX"
    "TOPO_LLM_REPOSITORY_BASE_PATH"
    "CONVLAB3_REPOSITORY_BASE_PATH"
)

echo ""
echo ">>> -------------------------------------"
echo ">>> Environment variables:"
echo ">>> -------------------------------------"
for var in "${VARIABLES_TO_LOG[@]}"; do
    echo ">>> $var=${!var}"
done
echo ">>> -------------------------------------"
echo ""

# Check that environment variables are set
if [ -z "${TOPO_LLM_REPOSITORY_BASE_PATH}" ]; then
    echo "@@@ Error: TOPO_LLM_REPOSITORY_BASE_PATH is not set." >&2
    exit 1
fi
if [ -z "${CONVLAB3_REPOSITORY_BASE_PATH}" ]; then
    echo "@@@ Error: CONVLAB3_REPOSITORY_BASE_PATH is not set." >&2
    exit 1
fi

# If $PBS_ARRAY_INDEX is not set, set it to a default value.
if [ -z "${PBS_ARRAY_INDEX}" ]; then
    echo "@@@ Warning: PBS_ARRAY_INDEX is not set. Setting it to 0." >&2
    PBS_ARRAY_INDEX=0
fi

echo ">>> [INFO] PBS_ARRAY_INDEX: ${PBS_ARRAY_INDEX}"

# Move to project folder
cd "${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints"

# Print the current directory
echo ">>> [INFO] Current directory: $(pwd)"
echo ">>> [INFO] Running on host: $(hostname)"
echo ">>> [INFO] Running on node: $(hostname -s)"
echo ">>> [INFO] Running on job ID: ${PBS_JOBID}"

# mode=scratch # scratch|local
mode=local # scratch|local
copy_cached=1 # 0|1


#echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}."
#if [ "${CUDA_VISIBLE_DEVICES}" -lt "2" ]; then
#    sleep 60
#    echo "Aborting."
#    exit
#fi



# Project paths ---------------------------------------------------

if [ -z ${PROJECT_FOLDER} ]; then
    PROJECT_FOLDER=`realpath $0`
    PROJECT_FOLDER=`dirname ${PROJECT_FOLDER}`
fi

SCRATCH_FOLDER=/gpfs/scratch/${USER}/${PBS_JOBID}
if [ $# -gt 0 ]; then
    SCRATCH_FOLDER=$1
fi

if ! [[ $SCRATCH_FOLDER =~ ^/gpfs/scratch/${USER} ]]; then
    echo "Invalid sync path ($SCRATCH_FOLDER). Should be a valid and path in /gpfs/scratch/${USER}. Aborting."
    exit 1
fi

# Michael's environment:
#
# PBS_O_WORKDIR=${PROJECT_FOLDER}
# cd "${PBS_O_WORKDIR}"

mkdir -p logs
RES_DIR=results
if [ "$mode" = "local" ]; then
    OUT_DIR=${RES_DIR}
else
    OUT_DIR=${SCRATCH_FOLDER}/${RES_DIR}
    ln -s ${SCRATCH_FOLDER} scratch.${PBS_JOBID}
    if [ "$copy_cached" = "1" ]; then
	mkdir -p ${SCRATCH_FOLDER}
	cp cached_* ${SCRATCH_FOLDER}
    fi
fi

# Main ------------------------------------------------------------

TOOLS_DIR="${TOPO_LLM_REPOSITORY_BASE_PATH}/data/models/trippy_r_checkpoints/multiwoz21/all_checkpoints_code"
DATASET_CONFIG="${TOOLS_DIR}/dataset_config/unified_multiwoz21.json"

# # # # # # # # # # # # # # # #
# START: Parameters

# Select seed via PBS_ARRAY_INDEX
SEEDS_SELECTION=(
    42
	43
	44
)
SEED=${SEEDS_SELECTION[$((PBS_ARRAY_INDEX))]}

# Make a one-element array, this is for compatibility with the rest of the script,
# and for added flexibility in the future.
SEEDS=(${SEED})

echo ">>> [INFO] SEEDS: ${SEEDS[@]}"

TRAIN_PHASES="-1" # -1: regular training, 0: proto training, 1: tagging, 2: spanless training
VALUE_MATCHING_WEIGHT=0.0 # When 0.0, value matching is not used

# Notes:
# - Uncomment the steps you want to run for each individual part of the pipeline.
# - The loop will run over all the steps in the outer loop array, and for each step,
#   it will call the inner scripts for the steps you want to run.
STEPS_TO_RUN_IN_OUTER_LOOP=(
	"train"
	"dev"
	"test"
)

STEPS_TO_RUN_FOR_EVALUATION=(
	"train"
	# "dev"
	# "test"
)

STEPS_TO_RUN_FOR_METRIC_DST=(
	"train"
	# "dev"
	# "test"
)

# END: Parameters
# # # # # # # # # # # # # # # #

args_add=""
phases="-1"

for x in ${SEEDS}; do
	echo ">>> [INFO] Running seed loop with seed ${x} ..."
    mkdir -p ${OUT_DIR}.${x}

	# # # #
	# Call the run_dst.py script for the training.
	if [ "$DO_TRAINING" = "True" ]; then
		echo ">>> [INFO] Running training ..."

		args_add="--do_train --predict_type=dev --hd=0.1"
		phases=${TRAIN_PHASES}

		echo ">>> NOTE: TRAINING CALL NOT YET IMPLEMENTED"

		# TODO: Call the training here

		echo ">>> [INFO] Running training DONE"
	fi

	# # # #
	# Run over all steps in the outer loop and call the selected parts of the pipeline for each step.
    for step in ${STEPS_TO_RUN_IN_OUTER_LOOP[@]}; do
		echo ">>> [INFO] Running outer loop for step ${step} ..."

		# Set the arguments for the run_dst.py script for evaluation.
		args_add="--do_eval --predict_type=${step}"


		if contains "$step" "${STEPS_TO_RUN_FOR_EVALUATION[@]}"; then
			echo ">>> [INFO] Running evaluation via run_dst.py with step=${step} ..."

			# START: THIS SHOULD BE A FUNCTION
			for phase in ${phases}; do
				echo ">>> [INFO] Running run_dst.py with step=${step}; phase=${phase}; seed=${x} ..."
				args_add_0=""
				if [ "$phase" = 0 ]; then
				args_add_0=""
				fi
				args_add_1=""
				if [ "$phase" = 1 ]; then
				args_add_1=""
				fi
				args_add_2=""
				if [ "$phase" = 2 ]; then
				args_add_2=""
				fi

				echo "args_add: ${args_add}"
				echo "args_add_0: ${args_add_0}"
				echo "args_add_1: ${args_add_1}"
				echo "args_add_2: ${args_add_2}"

				uv run python3 ${TOOLS_DIR}/run_dst.py \
					--task_name="unified" \
					--data_dir="" \
					--dataset_config=${DATASET_CONFIG} \
					--model_type="roberta" \
					--model_name_or_path="roberta-base" \
					--seed=${x} \
					--do_lower_case \
					--learning_rate=5e-5 \
					--num_train_epochs=20 \
					--max_seq_length=180 \
					--per_gpu_train_batch_size=32 \
					--per_gpu_eval_batch_size=32 \
					--output_dir=${OUT_DIR}.${x} \
					--save_epochs=1 \
					--patience=-1 \
					--eval_all_checkpoints \
					--warmup_proportion=0.05 \
					--adam_epsilon=1e-6 \
					--weight_decay=0.01 \
					--fp16 \
					--value_matching_weight=${VALUE_MATCHING_WEIGHT} \
					--none_weight=0.1 \
					--training_phase=${phase} \
					--local_files_only \
					${args_add} \
					${args_add_0} \
					${args_add_1} \
					${args_add_2} \
					2>&1 | tee ${OUT_DIR}.${x}/${step}.${phase}.log

				echo ">>> [INFO] Running run_dst.py with step=${step}; phase=${phase}; seed=${x} DONE"
			done
			# END: THIS SHOULD BE A FUNCTION
			
			echo ">>> [INFO] Running evaluation via run_dst.py with step=${step} DONE"
		else
			echo ">>> [INFO] Skipping evaluation via run_dst.py with step=${step}."
		fi


		if contains "$step" "${STEPS_TO_RUN_FOR_METRIC_DST[@]}"; then
			echo ">>> [INFO] Running metric_dst.py block for step ${step} ..."
			
			confidence=1.0
			if [[ ${VALUE_MATCHING_WEIGHT} > 0.0 ]]; then
			confidence="1.0 0.9 0.8 0.7 0.6 0.5"
			fi

			for dist_conf_threshold in ${confidence}; do
				uv run python3 ${TOOLS_DIR}/metric_dst.py \
					--dataset_config=${DATASET_CONFIG} \
					--confidence_threshold=${dist_conf_threshold} \
					--file_list="${OUT_DIR}.${x}/pred_res.${step}*json" \
					2>&1 | tee ${OUT_DIR}.${x}/eval_pred_${step}.${dist_conf_threshold}.log
			done
			
			echo ">>> [INFO] Running metric_dst.py block for step ${step} DONE"
		else
			echo ">>> [INFO] Skipping metric_dst.py block for step ${step}."
		fi

		echo ">>> [INFO] Running outer loop for step ${step} DONE"
    done
	echo ">>> [INFO] Running seed loop with seed ${x} DONE"
done

if [ "$mode" = "scratch" ]; then
    mv ${SCRATCH_FOLDER}/results* .
    mv ${SCRATCH_FOLDER}/cached_* .
    ./DO.cleanUp ${PBS_JOBID}
fi

echo ">>> [INFO] Job ${PBS_JOBID} finished. Exiting."
exit 0
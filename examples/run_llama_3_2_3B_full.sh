
#!/bin/bash


CONFIG="${CONFIG:-/workspace/torchtune/recipes/configs/llama3_2/3B_full.yaml}"
MODEL_DIR="${MODEL_DIR:-./models/Llama-3.2-3B-Instruct}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"

PACKED="${PACKED:-False}"
MAX_STEPS="${MAX_STEPS:null}"
MBS="${MBS:-64}"
GAS="${GAS:-1}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-True}"
CPU_OFFLOAD="${CPU_OFFLOAD:-True}"
COMPILE="${COMPILE:-True}"
EPOCHS="${EPOCHS:-3}"
SAVE_WEIGHTS="${SAVE_WEIGHTS:-True}"
SEQ_LEN="${SEQ_LEN:-null}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
MAX_AUTOTUNE="${MAX_AUTOTUNE:-False}"
VALIDATE="${VALIDATE:-False}"
SEED="${SEED:-42}"

LOG_FILES="history.txt"

echo "Running with environment variables..." | tee -a $LOG_FILES
env | tee -a $LOG_FILES


if [ ! -f "$CONFIG" ]; then
    echo "$CONFIG not found, download it first" >&2
    exit 1
fi

if [ ! -d $MODEL_DIR ]; then
    echo "3B model not found in $MODEL_DIR" >&2
    echo "Download with: "
    echo "  huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir $MODEL_DIR --exclude 'original/*.pth'"
    exit 1
fi

tune run --nproc_per_node 8 \
    full_finetune_distributed --config $CONFIG \
    log_peak_memory_stats=True \
    output_dir=./logs \
    checkpointer.output_dir=./checkpoints \
    dataset.data_files=$TRAIN_FILE \
    tokenizer.path=${MODEL_DIR}/original/tokenizer.model \
    tokenizer.max_seq_len=$SEQ_LEN \
    checkpointer.checkpoint_dir=$MODEL_DIR \
    gradient_accumulation_steps=$GAS \
    max_steps_per_epoch=$MAX_STEPS \
    epochs=$EPOCHS \
    dataset.packed=$PACKED \
    fsdp_cpu_offload=$CPU_OFFLOAD \
    batch_size=$MBS \
    enable_activation_checkpointing=$ACTIVATION_CHECKPOINTING \
    compile=$COMPILE \
    seed=$SEED \
    $VALIDATE_ARGS \
    $EXTRA_ARGS \
        2>&1 | tee stdout.log
LOG_PATH=`cat stdout.log | grep 'Writing logs to ' | head -1 | awk '{print $4}'`

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")

echo ========================================================================== | tee -a history.txt
#COMMIT=$(git rev-parse --short HEAD)
echo TORCH=$TORCH_VERSION COMMIT=$COMMIT | tee -a $LOG_FILES
echo COMPILE=$COMPILE CPU_OFFLOAD=$CPU_OFFLOAD PACKED=$PACKED SEQ_LEN=$SEQ_LEN ACTIVATION_CHECKPOINTING=$ACTIVATION_CHECKPOINTING MAX_AUTOTUNE=$MAX_AUTOTUNE MBS=$MBS GAS=$GAS SEED=$SEED | tee -a $LOG_FILES
if [ ! -z "$EXTRA_ARGS" ] ; then
    echo "EXTRA_ARGS=$EXTRA_ARGS" | tee -a $LOG_FILES
fi

if [ -n "$LOG_PATH" ]; then
    grep -Eo "peak_memory_alloc:[0-9]+\.[0-9]+" $LOG_PATH | grep -Eo "([^:]*)$" | awk 'NR>1 {if(max<$1) max=$1;} END{print "Max memory alloc:", max}' | tee -a $LOG_FILES
    grep -Eo "tokens_per_second_per_gpu:[0-9]+\.[0-9]+" $LOG_PATH | grep -Eo "([^:]*)$" | awk 'NR>1 {sum+=$1;} END{print "Average tokens/s/gpu:", sum/NR}' | tee -a $LOG_FILES
    #grep time_per_validation_epoch $LOG_PATH | awk -F'[: ]'  '{ print "Step", $2, "validation loss:", $5}' | tee -a $LOG_FILES

    if [ "${SAVE_WEIGHTS,,}" = "true" ]; then
        cp $LOG_PATH $CHECKPOINT_DIR/steps.txt
        cp stdout.log $CHECKPOINT_DIR
    fi
else
    echo "No log path found in command output" >&2
    exit 1
fi

#!/bin/bash
# Setting environment
# Change the below command to point to your own conda execution script
source /c/Users/TuAhnDinh/Anaconda3/etc/profile.d/conda.sh
conda activate BachelorThesisST
# Manual variable setting
CONT_FROM_CHECKPOINT=no  # yes or no
SRC_LANG=en
TGT_LANG=de
SRC_FORMAT=mix # Can be text or audio or mix
# End of manual variable setting
TGT_FORMAT=text
TGT_EXTENSION=txt
if [ "${SRC_LANG}" = "en" ]; then
  DATA_DIR=data/CoVoST2/preprocessed/super_dummy/en-X
else
  DATA_DIR=data/CoVoST2/preprocessed/super_dummy/${SRC_LANG}-${TGT_LANG}
fi
if [ "$SRC_FORMAT" = "audio" ]; then
  SRC_EXTENSION=scp
  CONCAT=4
  FORMAT=scp
elif [ "$SRC_FORMAT" = "text" ]; then
  SRC_EXTENSION=txt
  CONCAT=1
  FORMAT=bin
elif [ "$SRC_FORMAT" = "mix" ]; then
  CONCAT=4
  FORMAT=bin
fi
SUB_DIR=${SRC_FORMAT}_${SRC_LANG}_${TGT_FORMAT}_${TGT_LANG}
# Preprocess data
if [ -d ${DATA_DIR}/${SUB_DIR} ]; then
  echo "${SUB_DIR} already preprocessed"
else
  echo "Preprocessing ${SUB_DIR} data"
  mkdir ${DATA_DIR}/${SUB_DIR}
  if [ "$SRC_FORMAT" = "audio" ]; then
    python preprocess.py -train_src $DATA_DIR/${SRC_LANG}_${SRC_FORMAT}_train.${SRC_EXTENSION}  \
        -train_tgt $DATA_DIR/${TGT_LANG}_${TGT_FORMAT}_train.${TGT_EXTENSION}  \
        -valid_src $DATA_DIR/${SRC_LANG}_${SRC_FORMAT}_val.${SRC_EXTENSION}  \
        -valid_tgt $DATA_DIR/${TGT_LANG}_${TGT_FORMAT}_val.${TGT_EXTENSION}  \
        -src_seq_length 1024  \
        -tgt_seq_length 512  \
        -concat $CONCAT \
        -asr \
        -src_type $SRC_FORMAT \
        -asr_format scp \
        -save_data $DATA_DIR/${SUB_DIR}/data \
        -format $FORMAT
  elif [ "$SRC_FORMAT" = "text" ]; then
    python preprocess.py -train_src $DATA_DIR/${SRC_LANG}_${SRC_FORMAT}_train.${SRC_EXTENSION}  \
        -train_tgt $DATA_DIR/${TGT_LANG}_${TGT_FORMAT}_train.${TGT_EXTENSION}  \
        -valid_src $DATA_DIR/${SRC_LANG}_${SRC_FORMAT}_val.${SRC_EXTENSION}  \
        -valid_tgt $DATA_DIR/${TGT_LANG}_${TGT_FORMAT}_val.${TGT_EXTENSION}  \
        -src_seq_length 512  \
        -tgt_seq_length 512  \
        -concat $CONCAT \
        -src_type $SRC_FORMAT \
        -save_data $DATA_DIR/${SUB_DIR}/data \
        -format $FORMAT
  elif [ "$SRC_FORMAT" = "mix" ]; then
    python preprocess_multi_format.py -train_src_text $DATA_DIR/${SRC_LANG}_text_train.txt \
        -train_src_audio $DATA_DIR/${SRC_LANG}_audio_train.scp  \
        -train_tgt_of_text $DATA_DIR/${TGT_LANG}_${TGT_FORMAT}_train.${TGT_EXTENSION}  \
        -train_tgt_of_audio $DATA_DIR/${SRC_LANG}_${TGT_FORMAT}_train.${TGT_EXTENSION}  \
        -valid_src_text $DATA_DIR/${SRC_LANG}_text_val.txt \
        -valid_src_audio $DATA_DIR/${SRC_LANG}_audio_val.scp  \
        -valid_tgt_of_text $DATA_DIR/${TGT_LANG}_${TGT_FORMAT}_val.${TGT_EXTENSION}  \
        -valid_tgt_of_audio $DATA_DIR/${SRC_LANG}_${TGT_FORMAT}_val.${TGT_EXTENSION}  \
        -src_audio_seq_length 1024  \
        -tgt_of_audio_seq_length 512  \
        -src_text_seq_length 512  \
        -tgt_of_text_seq_length 512  \
        -concat $CONCAT \
        -asr \
        -src_type $SRC_FORMAT \
        -asr_format scp \
        -save_data $DATA_DIR/${SUB_DIR}/data \
        -format $FORMAT
  fi
fi
# Whether continue from a checkpoint
MODEL_DIR=models/${SUB_DIR}
EXPERIMENT_DIR=experiments/${SUB_DIR}
TOTAL_EPOCHS=64
if [ "$CONT_FROM_CHECKPOINT" = "yes" ]; then
  # Find best model to continue from
  BEST_CHECKPONTED=${MODEL_DIR}/$(python finding_best_model.py -model_dir $MODEL_DIR)
  # Set the number of remanining epochs to be run
  CURRENT_EPOCH=`echo $BEST_CHECKPONTED | sed -nr 's/.*e(.*).00.pt.*/\1/p'`
  N_EPOCHS=$(($TOTAL_EPOCHS-$CURRENT_EPOCH+1))
else
  # Delete old models and log files if any and create new ones
  if [ -d ${MODEL_DIR} ]; then
    rm -r ${MODEL_DIR}
  fi
  mkdir ${MODEL_DIR}
  if [ -d ${EXPERIMENT_DIR} ]; then
    rm -r ${EXPERIMENT_DIR}
  fi
  mkdir ${EXPERIMENT_DIR}
  # No checkpointed model to train from
  BEST_CHECKPONTED=''
  # Set the number of epochs to be run
  N_EPOCHS=$TOTAL_EPOCHS
fi
# Train model
echo "Training model..."
# Define some argument values
if [ "$SRC_FORMAT" = "audio" ]; then
  input_size=$((80*$CONCAT))
  LAYER=12
  TRANSFORMER=transformer
  OPTIM=Adam
  LR=0.001
  size=512
  innersize=$((size*4))
  ENC_LAYER=32
  optim_str="-optim adam"
  BATCH_SIZE_WORDS=2048
  DEATH_RATE=0.5
elif [ "$SRC_FORMAT" = "text" ]; then
  input_size=2048
  LAYER=8
  TRANSFORMER=transformer
  OPTIM=Adam
  LR=0.001
  size=512
  innersize=$((size*4))
  ENC_LAYER=-1
  optim_str="-optim adam"
  BATCH_SIZE_WORDS=3584
  DEATH_RATE=0.0
elif [ "$SRC_FORMAT" = "mix" ]; then
  input_size=$((80*$CONCAT))
  LAYER=12
  TRANSFORMER=transformer
  OPTIM=Adam
  LR=0.001
  size=512
  innersize=$((size*4))
  ENC_LAYER=32
  optim_str="-optim adam"
  BATCH_SIZE_WORDS=2048
  DEATH_RATE=0.5
fi
python -u train.py -data ${DATA_DIR}/${SUB_DIR}/data \
        -data_format $FORMAT \
        -save_model ${MODEL_DIR}/model \
        -model $TRANSFORMER \
        -batch_size_words $BATCH_SIZE_WORDS \
        -batch_size_update 24568 \
        -batch_size_sents 9999 \
        -batch_size_multiplier 8 \
        -encoder_type $SRC_FORMAT \
        -checkpointing 0 \
        -input_size $input_size \
        -concat $CONCAT \
        -layers $LAYER \
        -encoder_layer $ENC_LAYER \
        -death_rate $DEATH_RATE \
        -model_size $size \
        -inner_size $innersize \
        -n_heads 8 \
        -dropout 0.2 \
        -attn_dropout 0.2 \
        -word_dropout 0.1 \
        -emb_dropout 0.2 \
        -label_smoothing 0.1 \
        -epochs $N_EPOCHS \
        $optim_str \
        -learning_rate $LR \
        -normalize_gradient \
        -warmup_steps 8000 \
        -tie_weights \
        -seed 8877 \
        -log_interval 1000 \
        -update_frequency -1 \
        -gpus 0 | tee -a ${EXPERIMENT_DIR}/train.log
head -16 ${EXPERIMENT_DIR}/train.log > ${EXPERIMENT_DIR}/shortened_train.log
grep "Validation perplexity" ${EXPERIMENT_DIR}/train.log >> ${EXPERIMENT_DIR}/shortened_train.log

#!/bin/bash
# Setting environment
# Change the below command to point to your own conda execution script
source /c/Users/TuAhnDinh/Anaconda3/etc/profile.d/conda.sh
conda activate BachelorThesisST
# Manual variable setting
CONT_FROM_CHECKPOINT="no"  # yes or no
SRC_LANG=en
TGT_LANG=de
SRC_MODALITY=text # Can be text or audio
SUB_DATA_NAME=full
EXPERIMENT_NAME=${SUB_DATA_NAME}
FINAL_MODEL="best" # if best, evaluate the best model. if latest, evaluate the latest model
# End of manual variable setting
TGT_MODALITY=text
TGT_EXTENSION=txt
if [ "${SRC_LANG}" = "en" ]; then
  DATA_DIR=data/CoVoST2/preprocessed/${SUB_DATA_NAME}/en-X
else
  DATA_DIR=data/CoVoST2/preprocessed/${SUB_DATA_NAME}/${SRC_LANG}-${TGT_LANG}
fi
if [ "$SRC_MODALITY" = "audio" ]; then
  SRC_EXTENSION=scp
  CONCAT=4
  FORMAT=scp
elif [ "$SRC_MODALITY" = "text" ]; then
  SRC_EXTENSION=txt
  CONCAT=1
  FORMAT=mmem
fi
SUB_DIR=${SRC_MODALITY}_${SRC_LANG}_${TGT_MODALITY}_${TGT_LANG}
# Preprocess data
if [ -d ${DATA_DIR}/${SUB_DIR} ]; then
  echo "${SUB_DIR} already preprocessed"
else
  echo "Preprocessing ${SUB_DIR} data"
  mkdir ${DATA_DIR}/${SUB_DIR}
  if [ "$SRC_MODALITY" = "audio" ]; then
    python preprocess.py -train_src $DATA_DIR/${SRC_LANG}_${SRC_MODALITY}_train.${SRC_EXTENSION}  \
        -train_tgt $DATA_DIR/${TGT_LANG}_${TGT_MODALITY}_train.${TGT_EXTENSION}  \
        -valid_src $DATA_DIR/${SRC_LANG}_${SRC_MODALITY}_val.${SRC_EXTENSION}  \
        -valid_tgt $DATA_DIR/${TGT_LANG}_${TGT_MODALITY}_val.${TGT_EXTENSION}  \
        -src_seq_length 1024  \
        -tgt_seq_length 512  \
        -concat $CONCAT \
        -asr \
        -src_type $SRC_MODALITY \
        -asr_format scp \
        -save_data $DATA_DIR/${SUB_DIR}/data \
        -format $FORMAT
  elif [ "$SRC_MODALITY" = "text" ]; then
    python preprocess.py -train_src $DATA_DIR/${SRC_LANG}_${SRC_MODALITY}_train.${SRC_EXTENSION}  \
        -train_tgt $DATA_DIR/${TGT_LANG}_${TGT_MODALITY}_train.${TGT_EXTENSION}  \
        -valid_src $DATA_DIR/${SRC_LANG}_${SRC_MODALITY}_val.${SRC_EXTENSION}  \
        -valid_tgt $DATA_DIR/${TGT_LANG}_${TGT_MODALITY}_val.${TGT_EXTENSION}  \
        -src_seq_length 512  \
        -tgt_seq_length 512  \
        -concat $CONCAT \
        -src_type $SRC_MODALITY \
        -save_data $DATA_DIR/${SUB_DIR}/data \
        -format $FORMAT
  fi
fi
# Whether continue from a checkpoint
MODEL_DIR=models/${SUB_DIR}_${EXPERIMENT_NAME}
EXPERIMENT_DIR=experiments/${SUB_DIR}_${EXPERIMENT_NAME}
TOTAL_EPOCHS=64
if [ "$CONT_FROM_CHECKPOINT" = "yes" ]; then
  # Find latest model to continue from
  LATEST_CHECKPONTED=${MODEL_DIR}/$(python finding_latest_model.py -model_dir $MODEL_DIR)
  # Set the number of remanining epochs to be run
  CURRENT_EPOCH=`echo $LATEST_CHECKPONTED | sed -nr 's/.*e(.*).00.pt.*/\1/p'`
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
  LATEST_CHECKPONTED=""
  # Set the number of epochs to be run
  N_EPOCHS=$TOTAL_EPOCHS
fi
# Train model
echo "Training model..."
# Define some argument values
if [ "$SRC_MODALITY" = "audio" ]; then
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
elif [ "$SRC_MODALITY" = "text" ]; then
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
fi
if [ $CONT_FROM_CHECKPOINT == 'yes' ]; then
  python -u train.py -data ${DATA_DIR}/${SUB_DIR}/data \
          -data_format $FORMAT \
          -save_model ${MODEL_DIR}/model \
          -load_from $LATEST_CHECKPONTED \
          -model $TRANSFORMER \
          -batch_size_words $BATCH_SIZE_WORDS \
          -batch_size_update 24568 \
          -batch_size_sents 9999 \
          -batch_size_multiplier 8 \
          -encoder_type $SRC_MODALITY \
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
else
  python -u train.py -data ${DATA_DIR}/${SUB_DIR}/data \
        -data_format $FORMAT \
        -save_model ${MODEL_DIR}/model \
        -model $TRANSFORMER \
        -batch_size_words $BATCH_SIZE_WORDS \
        -batch_size_update 24568 \
        -batch_size_sents 9999 \
        -batch_size_multiplier 8 \
        -encoder_type $SRC_MODALITY \
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
fi
head -16 ${EXPERIMENT_DIR}/train.log > ${EXPERIMENT_DIR}/shortened_train.log
grep "Validation perplexity" ${EXPERIMENT_DIR}/train.log >> ${EXPERIMENT_DIR}/shortened_train.log
if [ "${FINAL_MODEL}" = "best" ]; then
  # Run best model on test set
  CHOSEN_MODEL_NAME=$(python finding_best_model.py -model_dir ${MODEL_DIR})
else
  # Run latest model on test set
  CHOSEN_MODEL_NAME=$(python finding_latest_model.py -model_dir ${MODEL_DIR})
fi
echo "Running ${FINAL_MODEL} model: ${CHOSEN_MODEL_NAME} on test set..." | tee ${EXPERIMENT_DIR}/note.txt
python translate.py -model ${MODEL_DIR}/$CHOSEN_MODEL_NAME \
    -src $DATA_DIR/${SRC_LANG}_${SRC_MODALITY}_test.${SRC_EXTENSION} \
    -concat $CONCAT \
    -asr_format scp \
    -encoder_type $SRC_MODALITY \
    -tgt $DATA_DIR/${TGT_LANG}_${TGT_MODALITY}_test.${TGT_EXTENSION}  \
    -output ${EXPERIMENT_DIR}/encoded_translations.txt \
    -batch_size 5 \
    -max_sent_length 1024 \
    -gpu 0
# Evaluate the model's translations
if [ "${SRC_LANG}" = "${TGT_LANG}" ]; then
  TASK=asr
else
  TASK=translation
fi
python translation_evaluation.py -save_data ${EXPERIMENT_DIR} \
    -encoded_output_text ${EXPERIMENT_DIR}/encoded_translations.txt \
    -text_encoder_decoder $DATA_DIR/${TGT_LANG}_${TGT_MODALITY}.model \
    -reference_text $DATA_DIR/${TGT_LANG}_raw_${TGT_MODALITY}_test.txt \
    -task $TASK

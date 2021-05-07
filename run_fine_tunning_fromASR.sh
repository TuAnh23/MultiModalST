#!/bin/bash
# Setting environment
# Change the below command to point to your own conda execution script
source /c/Users/TuAhnDinh/Anaconda3/etc/profile.d/conda.sh
conda activate BachelorThesisST
# ------------------------- Manual variable setting -------------------------
CONT_FROM_CHECKPOINT="no"  # yes or no
SRC_LANG=en
TGT_LANG=de
PREV_SUB_DATA_NAME=full
PREV_EXPERIMENT_NAME=audio_en_text_en
SUB_DATA_NAME=dummy
EXPERIMENT_NAME=${PREV_EXPERIMENT_NAME}_${PREV_SUB_DATA_NAME}_FT_st_${SUB_DATA_NAME}
# ------------------------- End of manual variable setting -------------------------
FINAL_MODEL="best" # if best, evaluate the best model. if latest, evaluate the latest model
SRC_MODALITY=audio
TGT_MODALITY=text
TGT_EXTENSION=txt
if [ "${SRC_LANG}" = "en" ]; then
  DATA_DIR=data/CoVoST2/preprocessed/${SUB_DATA_NAME}/en-X
  PREV_DATA_DIR=data/CoVoST2/preprocessed/${PREV_SUB_DATA_NAME}/en-X
else
  DATA_DIR=data/CoVoST2/preprocessed/${SUB_DATA_NAME}/${SRC_LANG}-${TGT_LANG}
  PREV_DATA_DIR=data/CoVoST2/preprocessed/${PREV_SUB_DATA_NAME}/${SRC_LANG}-${TGT_LANG}
fi
CONCAT=4
SUB_DIR=${SRC_MODALITY}_${SRC_LANG}_${TGT_MODALITY}_${TGT_LANG}_ft
# Preprocess data
if [ -d ${DATA_DIR}/${SUB_DIR} ]; then
  echo "${SUB_DIR} already preprocessed. Make sure the new data use the SAME VOCAB as the prev data"
else
  echo "Preprocessing ${SUB_DIR} data"
  mkdir ${DATA_DIR}/${SUB_DIR}
  # Create a vocabulary for all text targets
  python vocab_generator.py -filenames "${PREV_DATA_DIR}/${TGT_LANG}_text_train.txt" \
      -out_file $DATA_DIR/${SUB_DIR}/tgt_vocab
  # Use the above vocabs while preprocessing
  # Preprocess ST data
  python preprocess.py -train_src $DATA_DIR/${SRC_LANG}_audio_train.scp  \
      -train_tgt $DATA_DIR/${TGT_LANG}_text_train.txt  \
      -valid_src $DATA_DIR/${SRC_LANG}_audio_val.scp  \
      -valid_tgt $DATA_DIR/${TGT_LANG}_text_val.txt  \
      -src_seq_length 1024  \
      -tgt_seq_length 512  \
      -concat 4 \
      -asr \
      -src_type audio \
      -asr_format scp \
      -save_data $DATA_DIR/${SUB_DIR}/st_data \
      -format scp \
      -tgt_vocab $DATA_DIR/${SUB_DIR}/tgt_vocab
fi
# Whether continue from a checkpoint
MODEL_DIR=models/${EXPERIMENT_NAME}
PREV_MODEL_DIR=models/${PREV_EXPERIMENT_NAME}
EXPERIMENT_DIR=experiments/${EXPERIMENT_NAME}
TOTAL_EPOCHS=64 # Should manually stop training when converge
if [ "$CONT_FROM_CHECKPOINT" = "yes" ]; then
  # Find latest model to continue from
  LATEST_CHECKPONTED=${MODEL_DIR}/$(python finding_latest_model.py -model_dir $MODEL_DIR)
  # Set the number of remanining epochs to be run
  CURRENT_EPOCH=`echo $LATEST_CHECKPONTED | sed -nr 's/.*e(.*).00.pt.*/\1/p'`
  N_EPOCHS=$(($TOTAL_EPOCHS-$CURRENT_EPOCH+1))
  cont_checkpoint_str="-load_from ${LATEST_CHECKPONTED}"
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
  # Continue running from the best model of the prev experiment
  BEST_PREV=${PREV_MODEL_DIR}/$(python finding_best_model.py -model_dir $PREV_MODEL_DIR)
  # Set the number of epochs to be run
  N_EPOCHS=$TOTAL_EPOCHS
  load_enc_str="-load_encoder_from ${BEST_PREV}"
  load_dec_str="-load_decoder_from ${BEST_PREV}"
fi
# Train model
echo "Training model..."
# Define some argument values
# NOTE, the main data should have src audio, not text, since with the same number of sentences, src audio would need
# more batches, and we want all data to be covered
DATA=${DATA_DIR}/${SUB_DIR}/st_data
DATA_FORMAT=scp
input_size=$((80*$CONCAT))
LAYER=12
TRANSFORMER=transformer
OPTIM=Adam
LR=0.001
size=512
innersize=$((size*4))
ENC_LAYERS=32
optim_str="-optim adam"
BATCH_SIZE_WORDS=2048
BATCH_SIZE_SENT=9999
DEATH_RATE=0.0
# Run training process
python -u train.py -data $DATA \
$cont_checkpoint_str \
$load_enc_str \
$load_dec_str \
-data_format $DATA_FORMAT \
-save_model ${MODEL_DIR}/model \
-model $TRANSFORMER \
-batch_size_words $BATCH_SIZE_WORDS \
-batch_size_update 24568 \
-batch_size_sents $BATCH_SIZE_SENT \
-batch_size_multiplier 8 \
-encoder_type $SRC_MODALITY \
-checkpointing 0 \
-input_size $input_size \
-concat $CONCAT \
-layers $LAYER \
-encoder_layers $ENC_LAYERS \
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
sed '/.*Validation perplexity.*/{s///;q;}' ${EXPERIMENT_DIR}/train.log > ${EXPERIMENT_DIR}/shortened_train.log
grep -e "Train perplexity" -e "Validation perplexity" ${EXPERIMENT_DIR}/train.log >> ${EXPERIMENT_DIR}/shortened_train.log
if [ "${FINAL_MODEL}" = "best" ]; then
  # Run best model on test set
  CHOSEN_MODEL_NAME=$(python finding_best_model.py -model_dir ${MODEL_DIR})
else
  # Run latest model on test set
  CHOSEN_MODEL_NAME=$(python finding_latest_model.py -model_dir ${MODEL_DIR})
fi
echo "Running ${FINAL_MODEL} model: ${CHOSEN_MODEL_NAME} on test set..." | tee ${EXPERIMENT_DIR}/note.txt
# Here we set -encoder_type=audio since we're only insterested in Speech Translation task
echo "Evaluating ST"
python translate.py -model ${MODEL_DIR}/$CHOSEN_MODEL_NAME \
    -src $DATA_DIR/${SRC_LANG}_audio_test.scp \
    -concat $CONCAT \
    -asr_format scp \
    -encoder_type audio \
    -tgt $DATA_DIR/${TGT_LANG}_${TGT_MODALITY}_test.${TGT_EXTENSION}  \
    -output ${EXPERIMENT_DIR}/encoded_translations_st.txt \
    -batch_size 5 \
    -max_sent_length 1024 \
    -gpu 0
# Evaluate the model's translations
TASK=translation
python translation_evaluation.py -save_data ${EXPERIMENT_DIR} \
    -encoded_output_text ${EXPERIMENT_DIR}/encoded_translations_st.txt \
    -text_encoder_decoder $DATA_DIR/${TGT_LANG}_${TGT_MODALITY}.model \
    -reference_text $DATA_DIR/${TGT_LANG}_raw_${TGT_MODALITY}_test.txt \
    -task $TASK \
    -specific_task st
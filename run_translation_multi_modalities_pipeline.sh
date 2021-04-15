#!/bin/bash
# Setting environment
# Change the below command to point to your own conda execution script
source /c/Users/TuAhnDinh/Anaconda3/etc/profile.d/conda.sh
conda activate BachelorThesisST
# Manual variable setting
CONT_FROM_CHECKPOINT=no  # yes or no
SRC_LANG=en
TGT_LANG=de
SUB_DATA_NAME=full
# Add task name (i.e. asr, mt, st) to the end of EXPERIMENT_NAME
# DEPI: disentangling positional info
# SE: share encoders
EXPERIMENT_NAME=${SUB_DATA_NAME}_asr_mt_DEPI_SE
FINAL_MODEL="latest" # if best, evaluate the best model. if latest, evaluate the latest model
EVALUATE_ADDITIONAL_TASKS="yes" # whether to evaluate on test set for additional tasks 
# End of manual variable setting
SRC_MODALITY=mix
TGT_MODALITY=text
TGT_EXTENSION=txt
if [ "${SRC_LANG}" = "en" ]; then
  DATA_DIR=data/CoVoST2/preprocessed/${SUB_DATA_NAME}/en-X
else
  DATA_DIR=data/CoVoST2/preprocessed/${SUB_DATA_NAME}/${SRC_LANG}-${TGT_LANG}
fi
CONCAT=4
SUB_DIR=${SRC_MODALITY}_${SRC_LANG}_${TGT_MODALITY}_${TGT_LANG}
# Preprocess data
if [ -d ${DATA_DIR}/${SUB_DIR} ]; then
  echo "${SUB_DIR} already preprocessed"
else
  echo "Preprocessing ${SUB_DIR} data"
  mkdir ${DATA_DIR}/${SUB_DIR}
  # Create a vocabulary for all text sources
  python vocab_generator.py -filenames $DATA_DIR/${SRC_LANG}_text_train.txt \
      -out_file $DATA_DIR/${SUB_DIR}/src_vocab
  # Create a vocabulary for all text targets
  python vocab_generator.py -filenames "$DATA_DIR/${SRC_LANG}_text_train.txt|${DATA_DIR}/${TGT_LANG}_text_train.txt" \
      -out_file $DATA_DIR/${SUB_DIR}/tgt_vocab
  # Use the above vocabs while preprocessing
  # Preprocess ASR data
  python preprocess.py -train_src $DATA_DIR/${SRC_LANG}_audio_train.scp  \
      -train_tgt $DATA_DIR/${SRC_LANG}_text_train.txt  \
      -valid_src $DATA_DIR/${SRC_LANG}_audio_val.scp  \
      -valid_tgt $DATA_DIR/${SRC_LANG}_text_val.txt  \
      -train_src_lang ${SRC_LANG} \
      -train_tgt_lang ${SRC_LANG} \
      -valid_src_lang ${SRC_LANG} \
      -valid_tgt_lang ${SRC_LANG} \
      -all_langs "${SRC_LANG}|${TGT_LANG}" \
      -src_seq_length 1024  \
      -tgt_seq_length 512  \
      -concat 4 \
      -asr \
      -src_type audio \
      -asr_format scp \
      -save_data $DATA_DIR/${SUB_DIR}/asr_data \
      -format scp \
      -tgt_vocab $DATA_DIR/${SUB_DIR}/tgt_vocab
  # Preprocess ST data
  python preprocess.py -train_src $DATA_DIR/${SRC_LANG}_audio_train.scp  \
      -train_tgt $DATA_DIR/${TGT_LANG}_text_train.txt  \
      -valid_src $DATA_DIR/${SRC_LANG}_audio_val.scp  \
      -valid_tgt $DATA_DIR/${TGT_LANG}_text_val.txt  \
      -train_src_lang ${SRC_LANG} \
      -train_tgt_lang ${TGT_LANG} \
      -valid_src_lang ${SRC_LANG} \
      -valid_tgt_lang ${TGT_LANG} \
      -all_langs "${SRC_LANG}|${TGT_LANG}" \
      -src_seq_length 1024  \
      -tgt_seq_length 512  \
      -concat 4 \
      -asr \
      -src_type audio \
      -asr_format scp \
      -save_data $DATA_DIR/${SUB_DIR}/st_data \
      -format scp \
      -tgt_vocab $DATA_DIR/${SUB_DIR}/tgt_vocab
  # Preprocess MT data
  python preprocess.py -train_src $DATA_DIR/${SRC_LANG}_text_train.txt  \
      -train_tgt $DATA_DIR/${TGT_LANG}_text_train.txt  \
      -valid_src $DATA_DIR/${SRC_LANG}_text_val.txt  \
      -valid_tgt $DATA_DIR/${TGT_LANG}_text_val.txt  \
      -train_src_lang ${SRC_LANG} \
      -train_tgt_lang ${TGT_LANG} \
      -valid_src_lang ${SRC_LANG} \
      -valid_tgt_lang ${TGT_LANG} \
      -all_langs "${SRC_LANG}|${TGT_LANG}" \
      -src_seq_length 512  \
      -tgt_seq_length 512  \
      -concat 1 \
      -src_type text \
      -save_data $DATA_DIR/${SUB_DIR}/mt_data \
      -format mmem \
      -src_vocab $DATA_DIR/${SUB_DIR}/src_vocab \
      -tgt_vocab $DATA_DIR/${SUB_DIR}/tgt_vocab
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
# NOTE, the main data should have src audio, not text, since with the same number of sentences, src audio would need
# more batches, and we want all data to be covered
DATA=${DATA_DIR}/${SUB_DIR}/asr_data
DATA_FORMAT=scp
ADDITIONAL_DATA="${DATA_DIR}/${SUB_DIR}/mt_data"
ADDITIONAL_DATA_FORMAT="mmem"
DATA_RATIO="-1"
input_size=$((80*$CONCAT))
LAYER=12
TRANSFORMER=transformer
OPTIM=Adam
LR=0.001
size=512
innersize=$((size*4))
AUDIO_ENC_LAYERS=32
TEXT_ENC_LAYERS=$LAYER
optim_str="-optim adam"
BATCH_SIZE_WORDS=2048
BATCH_SIZE_SENT=9999
DEATH_RATE=0.5
SHARE_ENCODERS="all_text_enc"
# Run training process
if [ "$CONT_FROM_CHECKPOINT" = "yes" ]; then
    python -u train.py -data $DATA \
    -load_from $LATEST_CHECKPONTED \
    -data_format $DATA_FORMAT \
    -additional_data $ADDITIONAL_DATA \
    -additional_data_format $ADDITIONAL_DATA_FORMAT \
    -data_ratio $DATA_RATIO \
    -use_language_embedding \
    -language_embedding_type concat \
    -text_enc_change_residual_at $((TEXT_ENC_LAYERS/2)) \
    -text_enc_change_residual 2 \
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
    -audio_encoder_layers $AUDIO_ENC_LAYERS \
    -text_encoder_layers $TEXT_ENC_LAYERS \
    -share_encoders_parameter $SHARE_ENCODERS \
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
    python -u train.py -data $DATA \
    -data_format $DATA_FORMAT \
    -additional_data $ADDITIONAL_DATA \
    -additional_data_format $ADDITIONAL_DATA_FORMAT \
    -data_ratio $DATA_RATIO \
    -use_language_embedding \
    -language_embedding_type concat \
    -text_enc_change_residual_at $((TEXT_ENC_LAYERS/2)) \
    -text_enc_change_residual 2 \
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
    -audio_encoder_layers $AUDIO_ENC_LAYERS \
    -text_encoder_layers $TEXT_ENC_LAYERS \
    -share_encoders_parameter $SHARE_ENCODERS \
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
    -src_lang $SRC_LANG \
    -tgt_lang $TGT_LANG \
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
# Evaluate ASR
if [[ "$EXPERIMENT_NAME" = *"asr"* ]] && [[ "$EVALUATE_ADDITIONAL_TASKS" = "yes" ]]; then
  echo "Evaluating ASR"
  python translate.py -model ${MODEL_DIR}/$CHOSEN_MODEL_NAME \
      -src $DATA_DIR/${SRC_LANG}_audio_test.scp \
      -src_lang $SRC_LANG \
      -tgt_lang $SRC_LANG \
      -concat $CONCAT \
      -asr_format scp \
      -encoder_type audio \
      -tgt $DATA_DIR/${SRC_LANG}_text_test.txt  \
      -output ${EXPERIMENT_DIR}/encoded_translations_asr.txt \
      -batch_size 5 \
      -max_sent_length 1024 \
      -gpu 0
  # Evaluate the model's translations
  TASK=asr
  python translation_evaluation.py -save_data ${EXPERIMENT_DIR} \
      -encoded_output_text ${EXPERIMENT_DIR}/encoded_translations_asr.txt \
      -text_encoder_decoder $DATA_DIR/${SRC_LANG}_text.model \
      -reference_text $DATA_DIR/${SRC_LANG}_raw_text_test.txt \
      -task $TASK \
      -specific_task asr
fi
# Evaluate MT
if [[ "$EXPERIMENT_NAME" = *"mt"* ]] && [[ "$EVALUATE_ADDITIONAL_TASKS" = "yes" ]]; then
  echo "Evaluating MT"
  python translate.py -model ${MODEL_DIR}/$CHOSEN_MODEL_NAME \
      -src $DATA_DIR/${SRC_LANG}_text_test.txt \
      -src_lang $SRC_LANG \
      -tgt_lang $TGT_LANG \
      -concat $CONCAT \
      -encoder_type text \
      -tgt $DATA_DIR/${TGT_LANG}_text_test.txt  \
      -output ${EXPERIMENT_DIR}/encoded_translations_mt.txt \
      -batch_size 5 \
      -max_sent_length 1024 \
      -gpu 0
  # Evaluate the model's translations
  TASK=translation
  python translation_evaluation.py -save_data ${EXPERIMENT_DIR} \
      -encoded_output_text ${EXPERIMENT_DIR}/encoded_translations_mt.txt \
      -text_encoder_decoder $DATA_DIR/${TGT_LANG}_text.model \
      -reference_text $DATA_DIR/${TGT_LANG}_raw_text_test.txt \
      -task $TASK \
      -specific_task mt
fi
#!/bin/bash
# Setting environment
# Change the below command to point to your own conda execution script
source /c/Users/TuAhnDinh/Anaconda3/etc/profile.d/conda.sh
conda activate BachelorThesisST
# ------------------------- Manual variable setting -------------------------
CONT_FROM_CHECKPOINT="no"  # yes or no
SRC_LANG=en
TGT_LANG=de
SUB_DATA_NAME=dummy
EVAL_MODEL_DIR=
EXPERIMENT_NAME=modality_classifier_${SUB_DATA_NAME}
# ------------------------- End of manual variable setting -------------------------
EVAL_MODEL=$EVAL_MODEL_DIR/$(python finding_best_model.py -model_dir ${EVAL_MODEL_DIR})
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
  # Create a vocabulary for all text sources and targets
  python vocab_generator.py -filenames "$DATA_DIR/${SRC_LANG}_text_train.txt|${DATA_DIR}/${TGT_LANG}_text_train.txt" \
      -out_file $DATA_DIR/${SUB_DIR}/src_tgt_vocab
  # Use the above vocabs while preprocessing
  # Preprocess ASR data
  python preprocess.py -train_src $DATA_DIR/${SRC_LANG}_audio_train.scp  \
      -train_tgt $DATA_DIR/${SRC_LANG}_text_train.txt  \
      -valid_src $DATA_DIR/${SRC_LANG}_audio_val.scp  \
      -valid_tgt $DATA_DIR/${SRC_LANG}_text_val.txt  \
      -train_src_lang audio \
      -train_tgt_lang text \
      -valid_src_lang audio \
      -valid_tgt_lang text \
      -all_langs "audio|text" \
      -src_seq_length 1024  \
      -tgt_seq_length 512  \
      -concat 4 \
      -asr \
      -src_type audio \
      -asr_format scp \
      -save_data $DATA_DIR/${SUB_DIR}/asr_data \
      -format scp \
      -tgt_vocab $DATA_DIR/${SUB_DIR}/src_tgt_vocab
  # Preprocess MT data
  python preprocess.py -train_src $DATA_DIR/${SRC_LANG}_text_train.txt  \
      -train_tgt $DATA_DIR/${TGT_LANG}_text_train.txt  \
      -valid_src $DATA_DIR/${SRC_LANG}_text_val.txt  \
      -valid_tgt $DATA_DIR/${TGT_LANG}_text_val.txt  \
      -train_src_lang text \
      -train_tgt_lang text \
      -valid_src_lang text \
      -valid_tgt_lang text \
      -all_langs "audio|text" \
      -src_seq_length 512  \
      -tgt_seq_length 512  \
      -concat 1 \
      -src_type text \
      -save_data $DATA_DIR/${SUB_DIR}/mt_data \
      -format mmem \
      -src_vocab $DATA_DIR/${SUB_DIR}/src_tgt_vocab \
      -tgt_vocab $DATA_DIR/${SUB_DIR}/src_tgt_vocab
fi
# Whether continue from a checkpoint
MODEL_DIR=models/${EXPERIMENT_NAME}
EXPERIMENT_DIR=experiments/${EXPERIMENT_NAME}
TOTAL_EPOCHS=64
if [ "$CONT_FROM_CHECKPOINT" = "yes" ]; then
  # Find latest model to continue from
  LATEST_CHECKPONTED=${MODEL_DIR}/$(python finding_latest_model.py -model_dir $MODEL_DIR)
  # Set the number of remanining epochs to be run
  CURRENT_EPOCH=`echo $LATEST_CHECKPONTED | sed -nr 's/.*e(.*).00.pt.*/\1/p'`
  N_EPOCHS=$(($TOTAL_EPOCHS-$CURRENT_EPOCH))
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
  # Set the number of epochs to be run
  N_EPOCHS=$TOTAL_EPOCHS
  load_enc_str="-load_encoder_from ${EVAL_MODEL}"
  load_dec_str="-load_decoder_from ${EVAL_MODEL}"
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
DEATH_RATE=0.0
SHARE_ENCODERS="all_text_enc"
# Setting for share encoders (SE)
share_encoder_str="-share_encoders_parameter ${SHARE_ENCODERS}"
# Setting for JoinEmbedding
join_embedding_str="-join_embedding"
# Run training process
python -u train.py -data $DATA \
-language_classifier \
-language_classifier_tok \
-token_classifier 0 \
-num_classifier_languages 2 \
$cont_checkpoint_str \
$load_enc_str \
$load_dec_str \
-data_format $DATA_FORMAT \
-additional_data $ADDITIONAL_DATA \
-additional_data_format $ADDITIONAL_DATA_FORMAT \
-data_ratio $DATA_RATIO \
-use_language_embedding \
-language_embedding_type concat \
$text_enc_depi_layer_str \
$text_enc_depi_type_str \
$aux_loss_start_from_str \
$sim_loss_type_str \
$aux_loss_weight_str \
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
$share_encoder_str \
-death_rate $DEATH_RATE \
-model_size $size \
-inner_size $innersize \
-n_heads 8 \
-dropout 0.0 \
-attn_dropout 0.0 \
-word_dropout 0.0 \
-emb_dropout 0.0 \
-label_smoothing 0.1 \
-epochs $N_EPOCHS \
$optim_str \
-reset_optim \
-learning_rate $LR \
-normalize_gradient \
-warmup_steps 400 \
-tie_weights \
$join_embedding_str \
-seed 8877 \
-log_interval 1000 \
-update_frequency -1 \
-gpus 0 | tee -a ${EXPERIMENT_DIR}/train.log
sed '/.*Classifier accuracy.*/{s///;q;}' ${EXPERIMENT_DIR}/train.log > ${EXPERIMENT_DIR}/shortened_train.log
grep -e "Train perplexity" -e "Classifier accuracy" ${EXPERIMENT_DIR}/train.log >> ${EXPERIMENT_DIR}/shortened_train.log
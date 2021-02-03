#!/bin/bash
# Setting environment
# Change the below command to point to your own conda execution script
source /c/Users/TuAhnDinh/Anaconda3/etc/profile.d/conda.sh
conda activate BachelorThesisST
# Setting variables
SRC_LANG=en
TGT_LANG=en
if [ "${SRC_LANG}" = "en" ]; then
  DATA_DIR=data/CoVoST2/preprocessed/full/en-X
else
  DATA_DIR=data/CoVoST2/preprocessed/full/${SRC_LANG}-${TGT_LANG}
fi
SRC_FORMAT=audio # Can be text or audio
TGT_FORMAT=text
TGT_EXTENSION=txt
if [ "$SRC_FORMAT" = "audio" ]; then
  SRC_EXTENSION=scp
  CONCAT=4
  FORMAT=raw
elif [ "$SRC_FORMAT" = "text" ]; then
  SRC_EXTENSION=txt
  CONCAT=1
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
  fi
fi
# Train model
echo "Training model..."
if [ ! -d models/${SUB_DIR} ]; then
  mkdir models/${SUB_DIR}
fi
if [ ! -d experiments/${SUB_DIR} ]; then
  mkdir experiments/${SUB_DIR}
fi
if [ "$SRC_FORMAT" = "audio" ]; then
  input_size=$((80*$CONCAT))
  LAYER=12
  TRANSFORMER=stochastic_transformer
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
  LAYER=4
  TRANSFORMER=relative_transformer
  OPTIM=Adam
  LR=0.001
  size=512
  innersize=$((size*4))
  ENC_LAYER=-1
  optim_str="-optim adam"
  BATCH_SIZE_WORDS=3584
  DEATH_RATE=0.0
fi
python -u train.py -data ${DATA_DIR}/${SUB_DIR}/data \
        -data_format $FORMAT \
        -save_model models/$SUB_DIR/model \
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
        -epochs 64 \
        $optim_str \
        -learning_rate $LR \
        -normalize_gradient \
        -warmup_steps 8000 \
        -tie_weights \
        -seed 8877 \
        -log_interval 1000 \
        -gpus 0 | tee experiments/${SUB_DIR}/train.log
head -16 experiments/${SUB_DIR}/train.log > experiments/${SUB_DIR}/shortened_train.log
grep "Validation perplexity" experiments/${SUB_DIR}/train.log >> experiments/${SUB_DIR}/shortened_train.log
# Run best model on test set
BEST_MODEL_NAME=$(python finding_best_model.py -model_dir models/${SUB_DIR})
echo "Running ${BEST_MODEL_NAME} on test set..."
python translate.py -model models/$SUB_DIR/$BEST_MODEL_NAME \
    -src $DATA_DIR/${SRC_LANG}_${SRC_FORMAT}_test.${SRC_EXTENSION} \
    -concat $CONCAT \
    -asr_format scp \
    -encoder_type $SRC_FORMAT \
    -tgt $DATA_DIR/${TGT_LANG}_${TGT_FORMAT}_test.${TGT_EXTENSION}  \
    -output experiments/${SUB_DIR}/encoded_translations.txt \
    -batch_size 5 \
    -max_sent_length 1024 \
    -verbose \
    -gpu 0
# Evaluate the model's translations
python translation_evaluation.py -save_data experiments/${SUB_DIR} \
    -encoded_output_translation experiments/${SUB_DIR}/encoded_translations.txt \
    -text_encoder_decoder $DATA_DIR/${TGT_LANG}_${TGT_FORMAT}.model \
    -reference_translation $DATA_DIR/${TGT_LANG}_raw_${TGT_FORMAT}_test.txt

#!/bin/bash
# Setting environment
# Change the below command to point to your own conda execution script
source /c/Users/TuAhnDinh/Anaconda3/etc/profile.d/conda.sh
conda activate BachelorThesisST
# ------------------------- Manual variable setting -------------------------
MODEL_DIR=
ACTIVATIONS_PATH=activations/name
DATA_DIR=data/CoVoST2/preprocessed/quick_eval/en-X
# ------------------------- End of manual variable setting -------------------------
CHOSEN_MODEL_NAME=$(python finding_best_model.py -model_dir ${MODEL_DIR})
SRC_LANG=en
TGT_LANG=de
if [ -d ${ACTIVATIONS_PATH} ]; then
  rm -r ${ACTIVATIONS_PATH}
fi
mkdir ${ACTIVATIONS_PATH}
# Save activations
echo "Save audio activations"
  python translate.py -model ${MODEL_DIR}/$CHOSEN_MODEL_NAME \
      -src $DATA_DIR/${SRC_LANG}_audio_test.scp \
      -src_lang $SRC_LANG \
      -tgt_lang $SRC_LANG \
      -concat 4 \
      -asr_format scp \
      -encoder_type audio \
      -tgt $DATA_DIR/${SRC_LANG}_text_test.txt \
      -output ${ACTIVATIONS_PATH}/encoded_translations_asr.txt \
      -batch_size 5 \
      -max_sent_length 1024 \
      -gpu 0 \
      -save_activation activations/audio_activation.pt
echo "Save text activations"
python translate.py -model ${MODEL_DIR}/$CHOSEN_MODEL_NAME \
    -src $DATA_DIR/${SRC_LANG}_text_test.txt \
    -src_lang $SRC_LANG \
    -tgt_lang $TGT_LANG \
    -encoder_type text \
    -tgt $DATA_DIR/${TGT_LANG}_text_test.txt  \
    -output ${ACTIVATIONS_PATH}/encoded_translations_mt.txt \
    -batch_size 5 \
    -max_sent_length 1024 \
    -gpu 0 \
    -save_activation activations/text_activation.pt
# Inspect SVCCA
python svcca/inspect_st.py activations
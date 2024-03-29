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
mkdir ${ACTIVATIONS_PATH}/en_audio_activation
echo "Save audio activations"
python translate.py -model ${MODEL_DIR}/$CHOSEN_MODEL_NAME \
    -src $DATA_DIR/${SRC_LANG}_audio_test.scp \
    -src_lang $SRC_LANG \
    -tgt_lang $TGT_LANG \
    -concat 4 \
    -asr_format scp \
    -encoder_type audio \
    -tgt $DATA_DIR/${TGT_LANG}_text_test.txt \
    -output ${ACTIVATIONS_PATH}/encoded_translations_st.txt \
    -batch_size 5 \
    -max_sent_length 1024 \
    -gpu 0 \
    -save_activation $ACTIVATIONS_PATH/en_audio_activation
python post_process_activations.py -activations_dir $ACTIVATIONS_PATH/en_audio_activation \
    -save_activation $ACTIVATIONS_PATH/en_audio_activation.pt
rm -r $ACTIVATIONS_PATH/en_audio_activation
echo "Save en text activations"
mkdir ${ACTIVATIONS_PATH}/en_text_activation
python translate.py -model ${MODEL_DIR}/$CHOSEN_MODEL_NAME \
    -src $DATA_DIR/${SRC_LANG}_text_test.txt \
    -src_lang $SRC_LANG \
    -tgt_lang $TGT_LANG \
    -encoder_type text \
    -tgt $DATA_DIR/${TGT_LANG}_text_test.txt  \
    -output ${ACTIVATIONS_PATH}/encoded_translations_ende_mt.txt \
    -batch_size 5 \
    -max_sent_length 1024 \
    -gpu 0 \
    -save_activation $ACTIVATIONS_PATH/en_text_activation
python post_process_activations.py -activations_dir $ACTIVATIONS_PATH/en_text_activation \
    -save_activation $ACTIVATIONS_PATH/en_text_activation.pt
rm -r $ACTIVATIONS_PATH/en_text_activation
echo "Save de text activations"
mkdir ${ACTIVATIONS_PATH}/de_text_activation
python translate.py -model ${MODEL_DIR}/$CHOSEN_MODEL_NAME \
    -src $DATA_DIR/${TGT_LANG}_text_test.txt \
    -src_lang $TGT_LANG \
    -tgt_lang $TGT_LANG \
    -encoder_type text \
    -tgt $DATA_DIR/${TGT_LANG}_text_test.txt  \
    -output ${ACTIVATIONS_PATH}/encoded_translations_dede_mt.txt \
    -batch_size 5 \
    -max_sent_length 1024 \
    -gpu 0 \
    -save_activation $ACTIVATIONS_PATH/de_text_activation
python post_process_activations.py -activations_dir $ACTIVATIONS_PATH/de_text_activation \
    -save_activation $ACTIVATIONS_PATH/de_text_activation.pt
rm -r $ACTIVATIONS_PATH/de_text_activation
# Inspect SVCCA
python svcca/inspect_st.py $ACTIVATIONS_PATH 50 | tee ${ACTIVATIONS_PATH}/svcca_output.txt
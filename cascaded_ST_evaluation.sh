#!/bin/bash
# Setting environment
# Change the below command to point to your own conda execution script
source /c/Users/TuAhnDinh/Anaconda3/etc/profile.d/conda.sh
conda activate BachelorThesisST
# Manual variable setting
SRC_LANG=en
TGT_LANG=de
# End of manual variable setting
if [ "${SRC_LANG}" = "en" ]; then
  DATA_DIR=data/CoVoST2/preprocessed/full/en-X
else
  DATA_DIR=data/CoVoST2/preprocessed/full/${SRC_LANG}-${TGT_LANG}
fi
# Load MT and ASR model
MT_MODEL_DIR=models/text_${SRC_LANG}_text_${TGT_LANG}
MT_MODEL=${MT_MODEL_DIR}/$(python finding_best_model.py -model_dir ${MT_MODEL_DIR})
ASR_MODEL_DIR=models/audio_${SRC_LANG}_text_${SRC_LANG}
ASR_MODEL=${ASR_MODEL_DIR}/$(python finding_best_model.py -model_dir ${ASR_MODEL_DIR})
# Delete log files if any and create new ones
EXPERIMENT_DIR=experiments/cascaded_ST_${SRC_LANG}_${TGT_LANG}
if [ -d ${EXPERIMENT_DIR} ]; then
  rm -r ${EXPERIMENT_DIR}
fi
mkdir ${EXPERIMENT_DIR}
# Do translations
# Note: the -concat parameters are derived from the training shell script at run_translation_pipeline.sh
TRANSCRIPTION_PATH=${EXPERIMENT_DIR}/encoded_transcription.txt  # This will be output by ASR and input for MT
echo "Running ${ASR_MODEL} on ${SRC_LANG} audio on test set..."
python translate.py -model ${ASR_MODEL} \
    -src $DATA_DIR/${SRC_LANG}_audio_test.scp \
    -concat 4 \
    -asr_format scp \
    -encoder_type audio \
    -tgt $DATA_DIR/${SRC_LANG}_text_test.txt  \
    -output $TRANSCRIPTION_PATH \
    -batch_size 5 \
    -max_sent_length 1024 \
    -gpu 0
echo "Running ${MT_MODEL} on ${SRC_LANG} text on test set..."
python translate.py -model ${MT_MODEL} \
    -src $TRANSCRIPTION_PATH \
    -concat 1 \
    -asr_format scp \
    -encoder_type text \
    -tgt $DATA_DIR/${TGT_LANG}_text_test.txt  \
    -output ${EXPERIMENT_DIR}/encoded_translations.txt \
    -batch_size 5 \
    -max_sent_length 1024 \
    -gpu 0
# Evaluate the cascaded translations
python translation_evaluation.py -save_data ${EXPERIMENT_DIR} \
    -encoded_output_text ${EXPERIMENT_DIR}/encoded_translations.txt \
    -text_encoder_decoder $DATA_DIR/${TGT_LANG}_text.model \
    -reference_text $DATA_DIR/${TGT_LANG}_raw_text_test.txt \
    -task translation
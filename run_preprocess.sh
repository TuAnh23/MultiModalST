#!/bin/bash
source /c/Users/TuAhnDinh/Anaconda3/etc/profile.d/conda.sh
conda activate BachelorThesisST
python preprocess.py -train_src data/CoVoST2/preprocessed/dummy/nl_audios_train.scp  \
    -train_tgt data/CoVoST2/preprocessed/dummy/nl_text_train.txt  \
    -valid_src data/CoVoST2/preprocessed/dummy/nl_audios_val.scp  \
    -valid_tgt data/CoVoST2/preprocessed/dummy/nl_text_val.txt  \
    -src_seq_length 1024  \
    -tgt_seq_length 512  \
    -concat 4 \
    -asr \
    -src_type audio \
    -asr_format scp \
    -save_data data/CoVoST2/preprocessed/dummy/nl
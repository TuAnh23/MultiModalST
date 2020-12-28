#!/bin/bash
source /c/Users/TuAhnDinh/Anaconda3/etc/profile.d/conda.sh
conda activate BachelorThesisST
python translate.py -model models/dummy/model_ppl_26881171418161356094253400435962903554686976.000000_e5.00.pt \
    -src data/CoVoST2/preprocessed/dummy/nl_audios_val.scp \
    -concat 4 \
    -asr_format scp \
    -encoder_type audio \
    -tgt data/CoVoST2/preprocessed/dummy/nl_text_val.txt \
    -output experiments/dummy.txt \
    -batch_size 100 \
    -max_sent_length 1024 \
    -verbose \
    -gpu 0

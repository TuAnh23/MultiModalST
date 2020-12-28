#!/bin/bash
source /c/Users/TuAhnDinh/Anaconda3/etc/profile.d/conda.sh
conda activate BachelorThesisST
python train.py -data data/CoVoST2/preprocessed/dummy/nl \
		-data_format raw \
		-save_model models/dummy/model \
		-model transformer \
		-batch_size_words 100 \
		-batch_size_update 100 \
		-batch_size_sents 100 \
		-batch_size_multiplier 8 \
		-encoder_type audio \
		-checkpointing 0 \
		-layers 2 \
		-death_rate 0.5 \
		-n_heads 8 \
		-dropout 0.2 \
		-attn_dropout 0.2 \
		-word_dropout 0.1 \
		-emb_dropout 0.2 \
		-label_smoothing 0.1 \
    -epochs 5 \
		-learning_rate 2 \
		-normalize_gradient \
		-warmup_steps 8000 \
		-tie_weights \
		-seed 8877 \
		-log_interval 1000 \
		-gpus 0 \
		-input_size 320 `# Size of audio features * concat argument used to call preprocess.py` \
		-inner_size 1024 \
		-model_size 512
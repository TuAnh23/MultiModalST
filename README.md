# Bachelor thesis: Zero-shot Speech Translation

<em> This repository is derived from the NMTGMinor project at 
https://github.com/quanpn90/NMTGMinor <br/>
The SVCCA calculation is derived from https://github.com/nlp-dke/svcca <br/><br/>
Powered by Mediaan.com
</em> <br/><br/>

Speech Translation (ST) is the task of translating speech audio in a source language into text in a target language. This repository implements and experiments on 3 different approaches for ST:
- Cascaded ST, including 2 steps: Automatic Speech Recognition and Machine Translation
- End-to-end ST with end-to-end data
- End-to-end ST with no end-to-end data using Zero-shot learning

The Transformer architecture is used as the baseline for the implementation.


High-level instruction to use the repo:
- Run `covost_data_preparation.py` to download and preprocess the data.
- Run the shell script of interst, change the variables in the script if needed.
	- `run_translation_pipeline.sh` for single-task models
	- `cascaded_ST_evaluation.sh` evaluates cascaded ST using pretrained ASR and MT models
	- `run_translation_multi_modalities_pipeline.sh` for multi-task, multi-modality models (including zero-shot)
	- `run_zeroshot_with_artificial_data.sh` for zero-shot models using data augmentation
	- `run_bidirectional_zeroshot.sh` for zero-shot models using additional opposite training data
	- `run_fine_tunning.sh`, `run_fine_tunning_fromASR.sh` for fine-tuning pre-trained models
	- `modality_similarity_svcca.sh`, `modality_similarity_classifier.sh` measure text-audio similarity in representation
	
See `notebooks/Repo_Instruction.ipynb` for more details.
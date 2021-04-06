import argparse
import sentencepiece as spm
import sacrebleu
from vizseq.scorers.wer import WERScorer


parser = argparse.ArgumentParser()
parser.add_argument('-save_data', required=True,
                    help='Path to the directory to save the results.')
parser.add_argument('-encoded_output_text', required=True,
                    help='Path to the encoded output text.')
parser.add_argument('-text_encoder_decoder', required=True,
                    help='Path to the model used to decode the text output.')
parser.add_argument('-reference_text', required=True,
                    help='Path to the reference text.')
parser.add_argument('-task', required=True,
                    help='Options are [asr|translation]')
parser.add_argument('-specific_task', required=False, default="",
                    help='Options are [asr|st|mt]')


def decode_text(model_path, encoded_text_file, output_text_file):
    """
    :param model_path: the path to the model used to decode the text
    :param encoded_text_file: the encoded text file
    :param output_text_file: the output (decoded) text file
    :return:
    """
    sp = spm.SentencePieceProcessor(model_file=model_path)
    with open(encoded_text_file, 'r', encoding="utf-8") as f:
        encoded_lines = f.readlines()
    encoded_lines = [[i for i in string.split()] for string in encoded_lines]
    decoded_lines = [sp.decode(line) for line in encoded_lines]
    with open(output_text_file, 'w', encoding='utf-8') as f:
        for line in decoded_lines:
            f.write(line)
            f.write('\n')


def main():
    opt = parser.parse_args()
    model_output_file = f'{opt.save_data}/raw_text_output_{opt.specific_task}.txt'
    decode_text(model_path=opt.text_encoder_decoder, encoded_text_file=opt.encoded_output_text,
                output_text_file=model_output_file)
    with open(opt.reference_text, 'r', encoding="utf-8") as f:
        reference_texts = f.readlines()
    with open(model_output_file, 'r', encoding="utf-8") as f:
        model_outputs = f.readlines()

    if opt.specific_task != "":
        with open(f"{opt.save_data}/score.txt", 'a', encoding="utf-8") as f:
            f.write(f"{opt.specific_task}: ")
    if opt.task == "translation":
        bleu = sacrebleu.corpus_bleu(model_outputs, [reference_texts])
        with open(f"{opt.save_data}/score.txt", 'a', encoding="utf-8") as f:
            f.write(f"BLEU score: {bleu.score} \n")
    elif opt.task == "asr":
        scorer = WERScorer()
        with open(f"{opt.save_data}/score.txt", 'a', encoding="utf-8") as f:
            f.write(f"WER score: {scorer.score(model_outputs, [reference_texts]).corpus_score} \n")


if __name__ == "__main__":
    main()

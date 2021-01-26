import argparse
import sentencepiece as spm
import sacrebleu


parser = argparse.ArgumentParser()
parser.add_argument('-save_data', required=True,
                    help='Path to the directory to save the results.')
parser.add_argument('-encoded_output_translation', required=True,
                    help='Path to the encoded output translation.')
parser.add_argument('-text_encoder_decoder', required=True,
                    help='Path to the model used to decode the translation output.')
parser.add_argument('-reference_translation', required=True,
                    help='Path to the reference translation.')


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
    encoded_lines = [[int(i) for i in string.split()] for string in encoded_lines]
    decoded_lines = [sp.decode(line) for line in encoded_lines]
    with open(output_text_file, 'w', encoding='utf-8') as f:
        for line in decoded_lines:
            f.write(line)
            f.write('\n')


def main():
    opt = parser.parse_args()
    model_translation_file = f'{opt.save_data}/raw_text_translation.txt'
    decode_text(model_path=opt.text_encoder_decoder, encoded_text_file=opt.encoded_output_translation,
                output_text_file=model_translation_file)
    with open(opt.reference_translation, 'r', encoding="utf-8") as f:
        reference_translations = f.readlines()
    with open(model_translation_file, 'r', encoding="utf-8") as f:
        model_translations = f.readlines()
    bleu = sacrebleu.corpus_bleu(model_translations, [reference_translations])
    with open(f"{opt.save_data}/BLEU_score.txt", 'w', encoding="utf-8") as f:
        f.write(f"BLEU score: {bleu.score}")


if __name__ == "__main__":
    main()

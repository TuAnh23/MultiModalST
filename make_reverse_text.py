import string
import sentencepiece as spm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-save_data', required=True,
                    help='Path to the directory to save the results.')
parser.add_argument('-original_lang', required=True,
                    help='Name of the original language to be reversed.')
parser.add_argument('-raw_text_train_path', required=True,
                    help='Path to the raw text for training.')
parser.add_argument('-raw_text_val_path', required=True,
                    help='Path to the raw text for validation.')
parser.add_argument('-raw_text_test_path', required=True,
                    help='Path to the raw text for testing.')
parser.add_argument('-sentencepiece_model', required=False, default=None,
                    help='Path to the sentencepiece model.')


def reverse_sentence(line):
    """
    Reverse the sentence character-wise
    Lowercased and strip off all punctuations, except for the last punctuation
    Capitalize the first character of the sentence
    E.g. Hello World! --> Dlrow olleh!

    :param line: line of the sentence to be reversed
    :return: reversed line
    """
    reverse_line = line[::-1]
    reverse_line = reverse_line.lower()
    reverse_line = reverse_line.translate(str.maketrans('', '', string.punctuation))
    if line[-1] in string.punctuation:
        # Put back the last punctuation of the sentence if any
        reverse_line = reverse_line + line[-1]
    # Capitalize the beginning of the sentence
    reverse_line = reverse_line[0].upper() + reverse_line[1:]
    return reverse_line


def reverse_transcription(original_file, output_file):
    with open(original_file, 'r', encoding="utf-8") as f:
        original_sentences = f.readlines()
    original_sentences = [sentence.rstrip('\n') for sentence in original_sentences]
    with open(output_file, "w", encoding="utf-8") as out_file:
        for sentence in original_sentences:
            out_file.write(reverse_sentence(sentence))
            out_file.write("\n")
    return output_file


def preprocess_reversed_transcription(raw_text_train_path, raw_text_val_path, raw_text_test_path, save_location,
                                      lang, given_model=None):
    if given_model is not None:
        model_path = given_model
    else:
        # Train the model to do subword unit on the text
        input_file = raw_text_train_path  # one-sentence-per-line raw corpus file
        model_prefix = f'{save_location}/{lang}_text'
        # TODO
        vocab_size = 8000  # 8000, 16000, or 32000
        if lang == 'zh-CN' or lang == 'ja':
            character_coverage = 0.9995  # 0.9995 for languages with rich character set like Japanese or Chinese
        else:
            character_coverage = 1  # and 1.0 for other languages with small character set
        model_type = 'unigram'
        spm.SentencePieceTrainer.train(input=input_file, model_prefix=model_prefix, vocab_size=vocab_size,
                                       character_coverage=character_coverage, model_type=model_type)
        model_path = f"{model_prefix}.model"

    subword_unit(model_path, raw_text_train_path,
                 f'{save_location}/{lang}_text_train.txt')
    subword_unit(model_path, raw_text_val_path,
                 f'{save_location}/{lang}_text_val.txt')
    subword_unit(model_path, raw_text_test_path,
                 f'{save_location}/{lang}_text_test.txt')


def subword_unit(model_path, raw_text_file, output_file, output_type=str):
    """
    Use a Sentence Piece model to do subword unit on a text file
    """
    sp = spm.SentencePieceProcessor(model_file=model_path)
    with open(raw_text_file, 'r', encoding="utf-8") as f:
        raw_lines = f.readlines()
    processed_lines = [' '.join([str(elem) for elem in sp.encode(line, out_type=output_type)]) for line in raw_lines]
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(line)
            f.write('\n')


def already_exists(save_data, reversed_lang, given_model=None):
    if given_model is None:
        return os.path.exists(f'{save_data}/{reversed_lang}_text_train.txt') \
               and os.path.exists(f'{save_data}/{reversed_lang}_text_val.txt') \
               and os.path.exists(f'{save_data}/{reversed_lang}_text_test.txt') \
               and os.path.exists(f'{save_data}/{reversed_lang}_raw_text_train.txt') \
               and os.path.exists(f'{save_data}/{reversed_lang}_raw_text_val.txt') \
               and os.path.exists(f'{save_data}/{reversed_lang}_raw_text_test.txt') \
               and os.path.exists(f'{save_data}/{reversed_lang}_text.vocab') \
               and os.path.exists(f'{save_data}/{reversed_lang}_text.model')
    else:
        return os.path.exists(f'{save_data}/{reversed_lang}_text_train.txt') \
               and os.path.exists(f'{save_data}/{reversed_lang}_text_val.txt') \
               and os.path.exists(f'{save_data}/{reversed_lang}_text_test.txt') \
               and os.path.exists(f'{save_data}/{reversed_lang}_raw_text_train.txt') \
               and os.path.exists(f'{save_data}/{reversed_lang}_raw_text_val.txt') \
               and os.path.exists(f'{save_data}/{reversed_lang}_raw_text_test.txt')


def main():
    opt = parser.parse_args()
    reversed_lang = opt.original_lang + 'r'
    revesed_raw_text_train_path = reverse_transcription(opt.raw_text_train_path, f'{opt.save_data}'
                                                                                 f'/{reversed_lang}_raw_text_train.txt')
    revesed_raw_text_val_path = reverse_transcription(opt.raw_text_val_path, f'{opt.save_data}'
                                                                                 f'/{reversed_lang}_raw_text_val.txt')
    revesed_raw_text_test_path = reverse_transcription(opt.raw_text_test_path, f'{opt.save_data}'
                                                                                 f'/{reversed_lang}_raw_text_test.txt')
    if not already_exists(opt.save_data, reversed_lang, opt.sentencepiece_model):
        preprocess_reversed_transcription(revesed_raw_text_train_path, revesed_raw_text_val_path,
                                          revesed_raw_text_test_path, opt.save_data, reversed_lang,
                                          opt.sentencepiece_model)
    else:
        print("Reversed text data already existed.")


if __name__ == "__main__":
    main()

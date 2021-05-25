import requests
import os
import tarfile
import pandas as pd
import librosa
import numpy as np
from python_speech_features import logfbank, calculate_delta, normalize
from kaldiio import WriteHelper
import sentencepiece as spm
import csv

COVOST_DIR = 'data/CoVoST2'
PREPROCESSED_DIR = 'preprocessed/full'


def download(urls, xx_en_languages, en_xx_languages):
    """
    Download Covost dataset
    """
    # Downloads voice clips and transcripts
    for lang, url in urls.items():
        lang_dir = COVOST_DIR + '/' + lang
        if not os.path.exists(lang_dir):
            print(f'Downloading {lang} audios')
            filename = url.rsplit('/', 1)[1]
            r = requests.get(url)
            with open(COVOST_DIR + '/' + filename, 'wb') as f:
                f.write(r.content)
            print(f'Extracting {lang} audios')
            tf = tarfile.open(COVOST_DIR + '/' + filename)
            tf.extractall(lang_dir)
            tf.close()
            os.remove(COVOST_DIR + '/' + filename)

    # Download CoVoST 2 translations (covost_v2.<src_lang_code>_<tgt_lang_code>.tsv,
    # which matches the rows in validated.tsv from Common Voice)
    if not os.path.exists(COVOST_DIR + '/covost2'):
        os.mkdir(COVOST_DIR + '/covost2')
    for lang in xx_en_languages:
        if not os.path.exists(COVOST_DIR + '/covost2' + f'/{lang}_en'):
            os.mkdir(COVOST_DIR + '/covost2' + f'/{lang}_en')
            # Download and extract .tsv file
            url = f'https://dl.fbaipublicfiles.com/covost/covost_v2.{lang}_en.tsv.tar.gz'
            filename = url.rsplit('/', 1)[1]
            print(f'Download and extracting {filename}')
            r = requests.get(url)
            with open(COVOST_DIR + '/covost2' + f'/{lang}_en' + f'/{filename}', 'wb') as f:
                f.write(r.content)
            tf = tarfile.open(COVOST_DIR + '/covost2' + f'/{lang}_en' + f'/{filename}')
            tf.extractall(COVOST_DIR + '/covost2' + f'/{lang}_en')
            tf.close()
            os.remove(COVOST_DIR + '/covost2' + f'/{lang}_en' + f'/{filename}')

            # Split .tsv file into train, dev and test set
            os.system(f"python get_covost_splits.py "
                      f"--version 2 --src-lang {lang} --tgt-lang en "
                      f"--root {COVOST_DIR + '/covost2' + f'/{lang}_en'} "
                      f"--cv-tsv {COVOST_DIR + '/' + lang + '/validated.tsv'}")

    for lang in en_xx_languages:
        if not os.path.exists(COVOST_DIR + '/covost2' + f'/en_{lang}'):
            os.mkdir(COVOST_DIR + '/covost2' + f'/en_{lang}')
            # Download and extract .tsv file
            url = f'https://dl.fbaipublicfiles.com/covost/covost_v2.en_{lang}.tsv.tar.gz'
            filename = url.rsplit('/', 1)[1]
            print(f'Download and extracting {filename}')
            r = requests.get(url)
            with open(COVOST_DIR + '/covost2' + f'/en_{lang}' + f'/{filename}', 'wb') as f:
                f.write(r.content)
            tf = tarfile.open(COVOST_DIR + '/covost2' + f'/en_{lang}' + f'/{filename}')
            tf.extractall(COVOST_DIR + '/covost2' + f'/en_{lang}')
            tf.close()
            os.remove(COVOST_DIR + '/covost2' + f'/en_{lang}' + f'/{filename}')

            # Split .tsv file into train, dev and test set
            os.system(f"python get_covost_splits.py "
                      f"--version 2 --src-lang en --tgt-lang {lang} "
                      f"--root {COVOST_DIR + '/covost2' + f'/en_{lang}'} "
                      f"--cv-tsv {COVOST_DIR + '/' + 'en' + '/validated.tsv'}")


def remove_empty_transcription(split_df):
    new_df = split_df.loc[(split_df['sentence'] != "") & (split_df['sentence'] != '""') &
                          (split_df['translation'] != "") & (split_df['translation'] != '""')]
    return new_df


def remove_empty_audio(split_df, audiodir):
    empty = []
    paths = split_df['path'].values
    for path in paths:
        if os.path.getsize(audiodir + '/' + path) == 0:
            print(f"found {path} to be empty")
            empty.append(path)
    new_df = split_df.set_index('path')
    new_df.drop(labels=empty, axis='index', inplace=True)
    new_df.reset_index(inplace=True)
    return new_df


def read_tsv_split(translation_dir, src_lang, tgt_lang, split, audiodir):
    split_df = pd.read_csv(translation_dir + f'/covost_v2.{src_lang}_{tgt_lang}.{split}.tsv', sep='\t', header=0,
                           encoding="utf-8", escapechar="\\", quoting=csv.QUOTE_NONE, na_filter=False)
    return remove_empty_transcription(remove_empty_audio(split_df, audiodir))


def prepare_X_to_en_data(src_lang_list):
    """
    :param src_lang_list: list of source languages
    :return:
    """
    tgt_lang = 'en'
    for src_lang in src_lang_list:
        src_tgt_dir = f'{src_lang}-{tgt_lang}'
        src_tgt_dir_path = f'{COVOST_DIR}/{PREPROCESSED_DIR}/{src_tgt_dir}'

        # If this pair of src-tgt is not yet prepared then we prepare it
        if not os.path.exists(src_tgt_dir_path):
            prepare_src_tgt(src_lang, tgt_lang, src_tgt_dir_path)


def prepare_en_to_X_data(tgt_lang_list):
    """
    :param tgt_lang_list: list of target languages
    :return:
    """
    src_lang = 'en'
    en_X_dir = f'{COVOST_DIR}/{PREPROCESSED_DIR}/en-X'

    if not os.path.exists(en_X_dir):
        os.mkdir(en_X_dir)

    SRC_AUDIO_DIR = COVOST_DIR + '/' + src_lang
    audiodir = SRC_AUDIO_DIR + '/clips'

    info_df_dict = {}
    for tgt_lang in tgt_lang_list:
        TRANSLATIONS_DIR = COVOST_DIR + '/covost2' + f'/{src_lang}_{tgt_lang}'
        train_df = read_tsv_split(TRANSLATIONS_DIR, src_lang=src_lang, tgt_lang=tgt_lang, split='train',
                                  audiodir=audiodir)
        val_df = read_tsv_split(TRANSLATIONS_DIR, src_lang=src_lang, tgt_lang=tgt_lang, split='dev',
                                audiodir=audiodir)
        test_df = read_tsv_split(TRANSLATIONS_DIR, src_lang=src_lang, tgt_lang=tgt_lang, split='test',
                                 audiodir=audiodir)
        info_df_dict[tgt_lang] = {'train_df': train_df, 'val_df': val_df, 'test_df': test_df}
    # Check if the dataframes of different target languages are align (same rows contains info of same EN audios)
    if not is_align(info_df_dict):
        raise RuntimeError('The rows in the en-X dataframes are not align.')
    print('The translation dataframes are aligned.')

    # Since the English audio splits are identical accross languages of translation, the EN audios and EN
    # transcriptions only  need to be preprocessed once

    any_df_splits = info_df_dict[list(info_df_dict.keys())[0]]
    train_df_no_translation = any_df_splits['train_df'].drop(['translation', 'client_id'], axis='columns',
                                                             inplace=False)
    val_df_no_translation = any_df_splits['val_df'].drop(['translation', 'client_id'], axis='columns', inplace=False)
    test_df_no_translation = any_df_splits['test_df'].drop(['translation', 'client_id'], axis='columns', inplace=False)

    train_audios_list = [audiodir + '/' + path for path in train_df_no_translation['path']]
    val_audios_list = [audiodir + '/' + path for path in val_df_no_translation['path']]
    test_audios_list = [audiodir + '/' + path for path in test_df_no_translation['path']]

    # Prepare the audios
    if has_processed_audios(f'{en_X_dir}/{src_lang}'):
        print(f'Processed {src_lang} audio files found. Skip preprocessing {src_lang} audios. Make '
                             f'sure that the transcriptions and translations match the audios.')
    else:
        preprocess_audios(train_audios_list, f'{en_X_dir}/{src_lang}_audio_train')
        preprocess_audios(val_audios_list, f'{en_X_dir}/{src_lang}_audio_val')
        preprocess_audios(test_audios_list, f'{en_X_dir}/{src_lang}_audio_test')

    # Prepare transcriptions
    if has_processed_text(f'{en_X_dir}/{src_lang}'):
        print(f'Processed {src_lang} transcriptions files found. Skip preprocessing {src_lang} '
                             f'transcriptions. Make sure that the transcriptions and translations match the audios.')
    else:
        preprocess_transcription('original', train_df_no_translation, val_df_no_translation, test_df_no_translation,
                                 train_audios_list, val_audios_list, test_audios_list, f'{en_X_dir}', src_lang)

    # Prepare translations
    for tgt_lang in tgt_lang_list:
        if has_processed_text(f'{en_X_dir}/{tgt_lang}'):
            print(f'Processed {tgt_lang} translation files found. Skip preprocessing {tgt_lang} '
                                 f'translations. Make sure that the transcriptions and translations match the audios.')
        else:
            train_df = info_df_dict[tgt_lang]['train_df']
            val_df = info_df_dict[tgt_lang]['val_df']
            test_df = info_df_dict[tgt_lang]['test_df']
            preprocess_transcription('translated', train_df, val_df, test_df, train_audios_list, val_audios_list,
                                     test_audios_list, f'{en_X_dir}', tgt_lang)


def is_align(info_df_dict):
    """
    Check if the dataframes of Covost splits for different translations of EN audios are align (same row
    corresponding to same EN audio and EN transcription)
    :param info_df_dict: dictionary containing the dataframes of Covost splits for different translations of EN audios
    :return:
    """
    # Get any dataframe splits
    any_df_splits = info_df_dict[list(info_df_dict.keys())[0]]
    train_df_no_translation = any_df_splits['train_df'].drop(['translation', 'client_id'], axis='columns', inplace=False)
    val_df_no_translation = any_df_splits['val_df'].drop(['translation', 'client_id'], axis='columns', inplace=False)
    test_df_no_translation = any_df_splits['test_df'].drop(['translation', 'client_id'], axis='columns', inplace=False)

    for key, value in info_df_dict.items():
        if not (train_df_no_translation.equals(value['train_df'].drop(['translation', 'client_id'], axis='columns',
                                                                      inplace=False))
                and val_df_no_translation.equals(value['val_df'].drop(['translation', 'client_id'], axis='columns',
                                                                      inplace=False))
                and test_df_no_translation.equals(value['test_df'].drop(['translation', 'client_id'], axis='columns',
                                                                        inplace=False))):
            return False
    return True


def has_processed_audios(file_prefix):
    return os.path.exists(f'{file_prefix}_audio_train.scp') \
            and os.path.exists(f'{file_prefix}_audio_val.scp') \
            and os.path.exists(f'{file_prefix}_audio_test.scp') \
            and os.path.exists(f'{file_prefix}_audio_train.ark') \
            and os.path.exists(f'{file_prefix}_audio_val.ark') \
            and os.path.exists(f'{file_prefix}_audio_test.ark')


def has_processed_text(file_prefix):
    return os.path.exists(f'{file_prefix}_text_train.txt') \
            and os.path.exists(f'{file_prefix}_text_val.txt') \
            and os.path.exists(f'{file_prefix}_text_test.txt') \
            and os.path.exists(f'{file_prefix}_text.model') \
            and os.path.exists(f'{file_prefix}_text.vocab')


def prepare_src_tgt(src_lang, tgt_lang, preprocessed_src_tgt_dir_path):
    """
    Prepare audios and transcriptions in src_lang and translations in tgt_lang
    :param src_lang: language of the audios and its transcription
    :param tgt_lang: language of the translated transcription
    :param preprocessed_src_tgt_dir_path: location to save the processed audios, transcriptions and translations
    :return:
    """
    if src_lang == tgt_lang:
        raise RuntimeError(f'src_lang and tgt_lang must be different.')
    elif src_lang != 'en' and tgt_lang != 'en':
        raise RuntimeError(f'Translation data with src_lang {src_lang} and tgt_lang {tgt_lang} not available.')

    os.mkdir(preprocessed_src_tgt_dir_path)

    SRC_AUDIO_DIR = COVOST_DIR + '/' + src_lang
    audiodir = SRC_AUDIO_DIR + '/clips'

    TRANSLATIONS_DIR = COVOST_DIR + '/covost2' + f'/{src_lang}_{tgt_lang}'
    train_df = read_tsv_split(TRANSLATIONS_DIR, src_lang=src_lang, tgt_lang=tgt_lang, split='train', audiodir=audiodir)
    val_df = read_tsv_split(TRANSLATIONS_DIR, src_lang=src_lang, tgt_lang=tgt_lang, split='dev', audiodir=audiodir)
    test_df = read_tsv_split(TRANSLATIONS_DIR, src_lang=src_lang, tgt_lang=tgt_lang, split='test', audiodir=audiodir)

    train_audios_list = [audiodir + '/' + path for path in train_df['path']]
    val_audios_list = [audiodir + '/' + path for path in val_df['path']]
    test_audios_list = [audiodir + '/' + path for path in test_df['path']]

    # Prepare the audios
    preprocess_audios(train_audios_list, f'{preprocessed_src_tgt_dir_path}/{src_lang}_audio_train')
    preprocess_audios(val_audios_list, f'{preprocessed_src_tgt_dir_path}/{src_lang}_audio_val')
    preprocess_audios(test_audios_list, f'{preprocessed_src_tgt_dir_path}/{src_lang}_audio_test')

    # Prepare transcriptions
    preprocess_transcription('original', train_df, val_df, test_df, train_audios_list, val_audios_list,
                             test_audios_list, f'{preprocessed_src_tgt_dir_path}', src_lang)

    # Prepare translations
    preprocess_transcription('translated', train_df, val_df, test_df, train_audios_list, val_audios_list,
                             test_audios_list, f'{preprocessed_src_tgt_dir_path}', tgt_lang)


def preprocess_transcription(transcription_type, train_df, val_df, test_df, train_audios_list, val_audios_list,
                             test_audios_list,
                             save_location, lang):
    raw_text_train_path = collect_transcription(train_df, train_audios_list, f'{save_location}/{lang}_raw_text_train',
                                                transcription_type)
    raw_text_val_path = collect_transcription(val_df, val_audios_list, f'{save_location}/{lang}_raw_text_val',
                                              transcription_type)
    raw_text_test_path = collect_transcription(test_df, test_audios_list, f'{save_location}/{lang}_raw_text_test',
                                               transcription_type)

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

    subword_unit(f"{model_prefix}.model", raw_text_train_path,
                 f'{save_location}/{lang}_text_train.txt')
    subword_unit(f"{model_prefix}.model", raw_text_val_path,
                 f'{save_location}/{lang}_text_val.txt')
    subword_unit(f"{model_prefix}.model", raw_text_test_path,
                 f'{save_location}/{lang}_text_test.txt')


def mp3_to_wav(mp3_path):
    """
    Create a .wav verson of a .mp3 file in the same location, and return the path to the .wav file
    :param mp3_path: Path to the .mp3 file
    :return: Path to the .wav file
    """
    wav_path = f"{mp3_path[:-4]}.wav"
    gf = os.system(f"""ffmpeg -i {mp3_path} {wav_path}""")
    return wav_path


def load_mp3(mp3_path, sr=22050, mono=True, offset=0.0, duration=None, dtype=np.float32, res_type='kaiser_best'):
    """
    Wrapper function to load .mp3 audio
    """
    wav_path = mp3_to_wav(mp3_path)
    signal, sample_rate = librosa.load(wav_path, sr, mono, offset, duration, dtype, res_type)
    # Remove the .wav file when we're done
    os.remove(wav_path)
    return signal, sample_rate


def preprocess_audios(audio_paths, output_file_prefix):
    out_ark = output_file_prefix + ".ark"
    out_scp = output_file_prefix + ".scp"
    count = 0

    with WriteHelper('ark,scp:' + out_ark + ',' + out_scp) as writer:
        for audio in audio_paths:
            if audio.endswith('.mp3'):
                signal, sample_rate = load_mp3(audio, sr=16000)
            else:
                signal, sample_rate = librosa.load(audio, sr=16000)
            logmel = logfbank(signal, samplerate=sample_rate)
            delta = calculate_delta(logmel)
            features = np.concatenate([logmel, delta], axis=1)
            features = normalize(features)  # features.shape gives (x, 80)
            writer(str(count), features)
            count = count + 1
    return out_ark, out_scp


def prepare_sentence(sentence):
    """
    Since we use the package SentencePiece to subword the text, which does not require any tokenization of
    normalization on the text, we only remove the double quoutes and the begining and the end of a sentence in this
    function
    :param sentence: the sentence to be prepared
    :return: the prepared sentence
    """
    if sentence.startswith('"') and sentence.endswith('"'):
        return sentence[1:-1]
    return sentence


def collect_transcription(info_df, audio_paths, output_file_prefix, transcription_type='translated'):
    """ transcription_type is either 'original' or 'translated'
    """
    info_df_re_indexed = info_df.set_index('path')
    with open(f"{output_file_prefix}.txt", "w", encoding="utf-8") as out_file:
        for audio_path in audio_paths:
            # write line to output file
            audio_name = os.path.basename(audio_path)
            if transcription_type == 'original':
                out_file.write(prepare_sentence(info_df_re_indexed.loc[audio_name]['sentence']))
            elif transcription_type == 'translated':
                out_file.write(prepare_sentence(info_df_re_indexed.loc[audio_name]['translation']))
            else:
                raise RuntimeError("transcription_type is either 'original' or 'translated'")
            out_file.write("\n")
    return f"{output_file_prefix}.txt"


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


def main():
    print('Downloading Covost data')
    urls = {
        'en': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/en.tar.gz',
        'de': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/de.tar.gz',
        'ca': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/ca.tar.gz'}
    XX_EN_LANGUAGES = ['ca', 'de']
    EN_XX_LANGUAGES = ['de', 'ca']
    download(urls, xx_en_languages=XX_EN_LANGUAGES, en_xx_languages=EN_XX_LANGUAGES)

    print('Preparing Covost data')
    prepare_en_to_X_data(['ca, de'])
    prepare_X_to_en_data(['ca, de'])


if __name__ == "__main__":
    main()

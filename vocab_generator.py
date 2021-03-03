import argparse
import onmt


parser = argparse.ArgumentParser()
parser.add_argument('-filenames', required=True,
                    help='Paths to the files to generate the vocabulary from. Separate by |')
parser.add_argument('-out_file', required=True,
                    help='Location to store the vocab')
parser.add_argument('-num_threads', type=int, default=1,
                    help="Number of threads for multiprocessing")
parser.add_argument('-vocab_size', type=int, default=9999999,
                    help="Size of the vocabulary")
parser.add_argument('-input_type', default="word",
                    help="Input type: word/char")
parser.add_argument('-lower', action='store_true', help='lowercase data')

opt = parser.parse_args()


def make_vocab(filenames, size, tokenizer, num_workers=1):
    vocab = onmt.Dict([onmt.constants.PAD_WORD, onmt.constants.UNK_WORD,
                       onmt.constants.BOS_WORD, onmt.constants.EOS_WORD],
                      lower=opt.lower)

    for filename in filenames:
        print("Generating vocabulary from file %s ... " % filename)
        onmt.Dict.gen_dict_from_file(filename, vocab, tokenizer, num_workers=num_workers)

    original_size = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), original_size))

    vocab.writeFile(opt.out_file)


def main():
    filenames = opt.filenames.split("|")
    tokenizer = onmt.Tokenizer(opt.input_type, opt.lower)
    make_vocab(filenames, opt.vocab_size, tokenizer, num_workers=opt.num_threads)


if __name__ == '__main__':
    main()

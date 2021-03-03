from __future__ import division

import onmt
import onmt.markdown
import onmt.modules
import argparse
import torch
import time, datetime
from onmt.train_utils.trainer import XETrainer
from onmt.data.mmap_indexed_dataset import MMapIndexedDataset
from onmt.data.scp_dataset import SCPIndexDataset
from onmt.modules.loss import NMTLossFunc, NMTAndCTCLossFunc
from onmt.model_factory import build_model, optimize_model
from onmt.bayesian_factory import build_model as build_bayesian_model
from options import make_parser
from collections import defaultdict
import os
import numpy as np

parser = argparse.ArgumentParser(description='train.py')
onmt.markdown.add_md_help_argument(parser)

# Please look at the options file to see the options regarding models and data
parser = make_parser(parser)

opt = parser.parse_args()

print(opt)

# An ugly hack to have weight norm on / off
onmt.constants.weight_norm = opt.weight_norm
onmt.constants.checkpointing = opt.checkpointing
onmt.constants.max_position_length = opt.max_position_length

# Use static dropout if checkpointing > 0
if opt.checkpointing > 0:
    onmt.constants.static = True

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")

torch.manual_seed(opt.seed)


def numpy_to_torch(tensor_list):
    out_list = list()

    for tensor in tensor_list:
        if isinstance(tensor, np.ndarray):
            out_list.append(torch.from_numpy(tensor))
        else:
            out_list.append(tensor)

    return out_list


def main():
    if not opt.multi_dataset:
        # Load the data
        dicts, train_data, valid_data = load_data(data_path=opt.data, data_format=opt.data_format)
        print('----------------------------------------------------')
        print(dicts)
        print('----------------------------------------------------')
        # Load the additional data
        if opt.additional_data != 'none':
            additional_data = []
            additional_data_files = opt.additional_data.split(";")
            additional_data_formats = opt.additional_data_format.split(";")
            assert (len(additional_data_files) == len(additional_data_formats))
            for (additional_data_file, additional_data_format) in zip(additional_data_files, additional_data_formats):
                dicts_additional, train_data_additional, valid_data_additional = load_data(
                    data_path=additional_data_file,
                    data_format=opt.additional_data_format)

                # The additional data must have the same tgt vocab as the data
                assert dicts['tgt'].size() == dicts_additional['tgt'].size()
                # The additional data must have the same src vocab as the data (if any)
                if "src" in dicts_additional:
                    if "src" in dicts:
                        assert dicts['src'].size() == dicts_additional['src'].size()
                    else:
                        # If there is no src vocab in the data dicts yet, then we add the src vocab of the additional
                        # data to it (for convennient)
                        dicts['src'] = dicts_additional['src']

                # Store this data to the list
                additional_data.append({'dicts': dicts_additional, 'train_data': train_data_additional,
                                        'valid_data': valid_data_additional})
    else:
        if opt.additional_data != 'none':
            raise NotImplementedError("Multiple dataset with additional data (both text and audio as input) not "
                                      "implemented")

        print("[INFO] Reading multiple dataset ...")
        # raise NotImplementedError

        dicts = torch.load(opt.data + ".dict.pt")

        root_dir = os.path.dirname(opt.data)

        print("Loading training data ...")

        train_dirs, valid_dirs = dict(), dict()

        # scan the data directory to find the training data
        for dir_ in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, dir_)):
                if str(dir_).startswith("train"):
                    idx = int(dir_.split(".")[1])
                    train_dirs[idx] = dir_
                if dir_.startswith("valid"):
                    idx = int(dir_.split(".")[1])
                    valid_dirs[idx] = dir_

        train_sets, valid_sets = list(), list()

        for (idx_, dir_) in sorted(train_dirs.items()):

            data_dir = os.path.join(root_dir, dir_)
            print("[INFO] Loading training data %i from %s" % (idx_, dir_))

            if opt.data_format in ['bin', 'raw']:
                raise NotImplementedError

            elif opt.data_format in ['scp', 'scpmem', 'mmem']:
                from onmt.data.mmap_indexed_dataset import MMapIndexedDataset
                from onmt.data.scp_dataset import SCPIndexDataset

                if opt.data_format in ['scp', 'scpmem']:
                    audio_data = torch.load(os.path.join(data_dir, "data.scp_path.pt"))
                    src_data = SCPIndexDataset(audio_data, concat=opt.concat)
                else:
                    src_data = MMapIndexedDataset(os.path.join(data_dir, "data.src"))

                tgt_data = MMapIndexedDataset(os.path.join(data_dir, "data.tgt"))

                src_lang_data = MMapIndexedDataset(os.path.join(data_dir, 'data.src_lang'))
                tgt_lang_data = MMapIndexedDataset(os.path.join(data_dir, 'data.tgt_lang'))

                if os.path.exists(os.path.join(data_dir, 'data.src_sizes.npy')):
                    src_sizes = np.load(os.path.join(data_dir, 'data.src_sizes.npy'))
                    tgt_sizes = np.load(os.path.join(data_dir, 'data.tgt_sizes.npy'))
                else:
                    src_sizes, sizes = None, None

                if opt.encoder_type == 'audio':
                    data_type = 'audio'
                else:
                    data_type = 'text'

                if not opt.streaming:
                    train_data = onmt.Dataset(src_data,
                                              tgt_data,
                                              src_sizes, tgt_sizes,
                                              src_lang_data, tgt_lang_data,
                                              batch_size_words=opt.batch_size_words,
                                              data_type=data_type, sorting=True,
                                              batch_size_sents=opt.batch_size_sents,
                                              multiplier=opt.batch_size_multiplier,
                                              src_align_right=opt.src_align_right,
                                              augment=opt.augment_speech,
                                              upsampling=opt.upsampling,
                                              cleaning=True, verbose=True,
                                              num_split=len(opt.gpus))

                    train_sets.append(train_data)

                else:
                    print("Multi-dataset not implemented for Streaming tasks.")
                    raise NotImplementedError

        for (idx_, dir_) in sorted(valid_dirs.items()):

            data_dir = os.path.join(root_dir, dir_)

            print("[INFO] Loading validation data %i from %s" % (idx_, dir_))

            if opt.data_format in ['bin', 'raw']:
                raise NotImplementedError

            elif opt.data_format in ['scp', 'scpmem', 'mmem']:

                if opt.data_format in ['scp', 'scpmem']:
                    audio_data = torch.load(os.path.join(data_dir, "data.scp_path.pt"))
                    src_data = SCPIndexDataset(audio_data, concat=opt.concat)
                else:
                    src_data = MMapIndexedDataset(os.path.join(data_dir, "data.src"))

                tgt_data = MMapIndexedDataset(os.path.join(data_dir, "data.tgt"))

                src_lang_data = MMapIndexedDataset(os.path.join(data_dir, 'data.src_lang'))
                tgt_lang_data = MMapIndexedDataset(os.path.join(data_dir, 'data.tgt_lang'))

                if os.path.exists(os.path.join(data_dir, 'data.src_sizes.npy')):
                    src_sizes = np.load(os.path.join(data_dir, 'data.src_sizes.npy'))
                    tgt_sizes = np.load(os.path.join(data_dir, 'data.tgt_sizes.npy'))
                else:
                    src_sizes, sizes = None, None

                if opt.encoder_type == 'audio':
                    data_type = 'audio'
                else:
                    data_type = 'text'

                if not opt.streaming:
                    valid_data = onmt.Dataset(src_data, tgt_data,
                                              src_sizes, tgt_sizes,
                                              src_lang_data, tgt_lang_data,
                                              batch_size_words=opt.batch_size_words,
                                              data_type=data_type, sorting=True,
                                              batch_size_sents=opt.batch_size_sents,
                                              src_align_right=opt.src_align_right,
                                              cleaning=True, verbose=True, debug=True,
                                              num_split=len(opt.gpus))

                    valid_sets.append(valid_data)

                else:
                    raise NotImplementedError

        train_data = train_sets
        valid_data = valid_sets

    # TODO: see how to handle checkpoint with multiple data format (maybe no need to do anything. try a run)
    if opt.load_from:
        checkpoint = torch.load(opt.load_from, map_location=lambda storage, loc: storage)
        print("* Loading dictionaries from the checkpoint")
        dicts = checkpoint['dicts']
    else:
        dicts['tgt'].patch(opt.patch_vocab_multiplier)
        checkpoint = None

    # Put the vocab mask from dicts to the datasets
    for data in [train_data, valid_data]:
        if isinstance(data, list):
            for i, data_ in enumerate(data):
                data_.set_mask(dicts['tgt'].vocab_mask)
                data[i] = data_
        else:
            data.set_mask(dicts['tgt'].vocab_mask)
    if opt.additional_data != 'none':
        for data_dict in additional_data:
            for data in [data_dict['train_data'], data_dict['valid_data']]:
                if isinstance(data, list):
                    for i, data_ in enumerate(data):
                        data_.set_mask(dicts['tgt'].vocab_mask)
                        data[i] = data_
                else:
                    data.set_mask(dicts['tgt'].vocab_mask)

    if "src" in dicts:
        print(' * vocabulary size. source = %d; target = %d' %
              (dicts['src'].size(), dicts['tgt'].size()))
    else:
        print('[INFO] vocabulary size. target = %d' %
              (dicts['tgt'].size()))

    print('* Building model...')

    if not opt.fusion:
        if opt.bayes_by_backprop:
            model = build_bayesian_model(opt, dicts)
        else:
            model = build_model(opt, dicts)

        """ Building the loss function """
        # if opt.ctc_loss != 0:
        #     pass
        #     loss_function = NMTAndCTCLossFunc(dicts['tgt'].size(),
        #                                       label_smoothing=opt.label_smoothing,
        #                                       ctc_weight=opt.ctc_loss)
        if opt.nce:
            from onmt.modules.nce.nce_loss import NCELoss
            loss_function = NCELoss(opt.model_size, dicts['tgt'].size(), noise_ratio=opt.nce_noise,
                                    logz=9, label_smoothing=opt.label_smoothing)
        else:
            loss_function = NMTLossFunc(opt.model_size, dicts['tgt'].size(),
                                        label_smoothing=opt.label_smoothing,
                                        mirror=opt.mirror_loss,
                                        fast_xentropy=opt.fast_xentropy)

        # This function replaces modules with the more optimized counterparts so that it can run faster
        # Currently exp with LayerNorm
        if not opt.memory_profiling:
            optimize_model(model, fp16=opt.fp16)

    else:
        from onmt.model_factory import build_fusion
        from onmt.modules.loss import FusionLoss

        model = build_fusion(opt, dicts)

        loss_function = FusionLoss(dicts['tgt'].size(), label_smoothing=opt.label_smoothing)

    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)

    if not opt.debugging and len(opt.gpus) == 1:
        if opt.bayes_by_backprop:

            from onmt.train_utils.bayes_by_backprop_trainer import BayesianTrainer
            trainer = BayesianTrainer(model, loss_function, train_data, valid_data, dicts, opt)

        else:
            trainer = XETrainer(model, loss_function, train_data, valid_data, dicts, opt)
    else:
        from onmt.train_utils.new_trainer import Trainer
        trainer = Trainer(model, loss_function, train_data, valid_data, dicts, opt)

    # Add additional data (if any)
    if opt.additional_data != 'none':
        trainer.add_additional_data(additional_data, opt.data_ratio)
    trainer.run(checkpoint=checkpoint)


def load_data(data_path, data_format):
    """
    :param data_path: path to the .train.pt file
    :param data_format: bin, raw, scp, scpmem, or mmem
    :return:
    """
    if data_format in ['bin', 'raw']:
        start = time.time()

        if data_path.endswith(".train.pt"):
            print("Loading data from '%s'" % data_path)
            dataset = torch.load(data_path)
        else:
            print("Loading data from %s" % data_path + ".train.pt")
            dataset = torch.load(data_path + ".train.pt")

        elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
        print("Done after %s" % elapse)

        dicts = dataset['dicts']

        # For backward compatibility
        train_dict = defaultdict(lambda: None, dataset['train'])
        valid_dict = defaultdict(lambda: None, dataset['valid'])

        if train_dict['src_lang'] is not None:
            assert 'langs' in dicts
            train_src_langs = train_dict['src_lang']
            train_tgt_langs = train_dict['tgt_lang']
        else:
            # allocate new languages
            dicts['langs'] = {'src': 0, 'tgt': 1}
            train_src_langs = list()
            train_tgt_langs = list()
            # Allocation one for the bilingual case
            train_src_langs.append(torch.Tensor([dicts['langs']['src']]))
            train_tgt_langs.append(torch.Tensor([dicts['langs']['tgt']]))

        if not opt.streaming:
            train_data = onmt.Dataset(numpy_to_torch(train_dict['src']), numpy_to_torch(train_dict['tgt']),
                                      train_dict['src_sizes'], train_dict['tgt_sizes'],
                                      train_src_langs, train_tgt_langs,
                                      batch_size_words=opt.batch_size_words,
                                      data_type=dataset.get("type", "text"), sorting=True,
                                      batch_size_sents=opt.batch_size_sents,
                                      multiplier=opt.batch_size_multiplier,
                                      augment=opt.augment_speech,
                                      upsampling=opt.upsampling,
                                      num_split=len(opt.gpus))
        else:
            train_data = onmt.StreamDataset(train_dict['src'], train_dict['tgt'],
                                            train_src_langs, train_tgt_langs,
                                            batch_size_words=opt.batch_size_words,
                                            data_type=dataset.get("type", "text"), sorting=True,
                                            batch_size_sents=opt.batch_size_sents,
                                            multiplier=opt.batch_size_multiplier,
                                            augment=opt.augment_speech,
                                            upsampling=opt.upsampling)

        if valid_dict['src_lang'] is not None:
            assert 'langs' in dicts
            valid_src_langs = valid_dict['src_lang']
            valid_tgt_langs = valid_dict['tgt_lang']
        else:
            # allocate new languages
            valid_src_langs = list()
            valid_tgt_langs = list()

            # Allocation one for the bilingual case
            valid_src_langs.append(torch.Tensor([dicts['langs']['src']]))
            valid_tgt_langs.append(torch.Tensor([dicts['langs']['tgt']]))

        if not opt.streaming:
            valid_data = onmt.Dataset(numpy_to_torch(valid_dict['src']), numpy_to_torch(valid_dict['tgt']),
                                      valid_dict['src_sizes'], valid_dict['tgt_sizes'],
                                      valid_src_langs, valid_tgt_langs,
                                      batch_size_words=opt.batch_size_words,
                                      data_type=dataset.get("type", "text"), sorting=True,
                                      batch_size_sents=opt.batch_size_sents,
                                      upsampling=opt.upsampling,
                                      num_split=len(opt.gpus))
        else:
            valid_data = onmt.StreamDataset(numpy_to_torch(valid_dict['src']), numpy_to_torch(valid_dict['tgt']),
                                            valid_src_langs, valid_tgt_langs,
                                            batch_size_words=opt.batch_size_words,
                                            data_type=dataset.get("type", "text"), sorting=True,
                                            batch_size_sents=opt.batch_size_sents,
                                            upsampling=opt.upsampling)

        print(' * number of training sentences. %d' % len(dataset['train']['src']))
        print(' * maximum batch size (words per batch). %d' % opt.batch_size_words)

    elif data_format in ['scp', 'scpmem', 'mmem']:
        print("Loading memory mapped data files ....")
        start = time.time()
        from onmt.data.mmap_indexed_dataset import MMapIndexedDataset
        from onmt.data.scp_dataset import SCPIndexDataset

        dicts = torch.load(data_path + ".dict.pt")
        data_type = torch.load(data_path + ".type.pt")

        if data_format in ['scp', 'scpmem']:
            audio_data = torch.load(data_path + ".scp_path.pt")

        # allocate languages if not
        if 'langs' not in dicts:
            dicts['langs'] = {'src': 0, 'tgt': 1}
        else:
            print(dicts['langs'])

        train_path = data_path + '.train'
        if data_format in ['scp', 'scpmem']:
            train_src = SCPIndexDataset(audio_data['train'], concat=opt.concat)
        else:
            train_src = MMapIndexedDataset(train_path + '.src')

        train_tgt = MMapIndexedDataset(train_path + '.tgt')

        # check the lang files if they exist (in the case of multi-lingual models)
        if os.path.exists(train_path + '.src_lang.bin'):
            assert 'langs' in dicts
            train_src_langs = MMapIndexedDataset(train_path + '.src_lang')
            train_tgt_langs = MMapIndexedDataset(train_path + '.tgt_lang')
        else:
            train_src_langs = list()
            train_tgt_langs = list()
            # Allocate a Tensor(1) for the bilingual case
            train_src_langs.append(torch.Tensor([dicts['langs']['src']]))
            train_tgt_langs.append(torch.Tensor([dicts['langs']['tgt']]))

        # check the length files if they exist
        if os.path.exists(train_path + '.src_sizes.npy'):
            train_src_sizes = np.load(train_path + '.src_sizes.npy')
            train_tgt_sizes = np.load(train_path + '.tgt_sizes.npy')
        else:
            train_src_sizes, train_tgt_sizes = None, None

        # if opt.encoder_type == 'audio':
        #     data_type = 'audio'
        # else:
        #     data_type = 'text'

        if not opt.streaming:
            train_data = onmt.Dataset(train_src,
                                      train_tgt,
                                      train_src_sizes, train_tgt_sizes,
                                      train_src_langs, train_tgt_langs,
                                      batch_size_words=opt.batch_size_words,
                                      data_type=data_type, sorting=True,
                                      batch_size_sents=opt.batch_size_sents,
                                      multiplier=opt.batch_size_multiplier,
                                      src_align_right=opt.src_align_right,
                                      augment=opt.augment_speech,
                                      upsampling=opt.upsampling,
                                      cleaning=True, verbose=True,
                                      num_split=len(opt.gpus))
        else:
            train_data = onmt.StreamDataset(train_src,
                                            train_tgt,
                                            train_src_langs, train_tgt_langs,
                                            batch_size_words=opt.batch_size_words,
                                            data_type=data_type, sorting=False,
                                            batch_size_sents=opt.batch_size_sents,
                                            multiplier=opt.batch_size_multiplier,
                                            upsampling=opt.upsampling)

        valid_path = data_path + '.valid'
        if data_format in ['scp', 'scpmem']:
            valid_src = SCPIndexDataset(audio_data['valid'], concat=opt.concat)
        else:
            valid_src = MMapIndexedDataset(valid_path + '.src')
        valid_tgt = MMapIndexedDataset(valid_path + '.tgt')

        if os.path.exists(valid_path + '.src_lang.bin'):
            assert 'langs' in dicts
            valid_src_langs = MMapIndexedDataset(valid_path + '.src_lang')
            valid_tgt_langs = MMapIndexedDataset(valid_path + '.tgt_lang')
        else:
            valid_src_langs = list()
            valid_tgt_langs = list()

            # Allocation one for the bilingual case
            valid_src_langs.append(torch.Tensor([dicts['langs']['src']]))
            valid_tgt_langs.append(torch.Tensor([dicts['langs']['tgt']]))

        # check the length files if they exist
        if os.path.exists(valid_path + '.src_sizes.npy'):
            valid_src_sizes = np.load(valid_path + '.src_sizes.npy')
            valid_tgt_sizes = np.load(valid_path + '.tgt_sizes.npy')
        else:
            valid_src_sizes, valid_tgt_sizes = None, None

        if not opt.streaming:
            valid_data = onmt.Dataset(valid_src, valid_tgt,
                                      valid_src_sizes, valid_tgt_sizes,
                                      valid_src_langs, valid_tgt_langs,
                                      batch_size_words=opt.batch_size_words,
                                      data_type=data_type, sorting=True,
                                      batch_size_sents=opt.batch_size_sents,
                                      src_align_right=opt.src_align_right,
                                      cleaning=True, verbose=True, debug=True,
                                      num_split=len(opt.gpus))
        else:
            # for validation data, we have to go through sentences (very slow but to ensure correctness)
            valid_data = onmt.StreamDataset(valid_src, valid_tgt,
                                            valid_src_langs, valid_tgt_langs,
                                            batch_size_words=opt.batch_size_words,
                                            data_type=data_type, sorting=True,
                                            batch_size_sents=opt.batch_size_sents)

        elapse = str(datetime.timedelta(seconds=int(time.time() - start)))
        print("Done after %s" % elapse)

    else:
        raise NotImplementedError
    print(' * number of sentences in training data: %d' % train_data.size())
    print(' * number of sentences in validation data: %d' % valid_data.size())
    return dicts, train_data, valid_data


if __name__ == "__main__":
    main()

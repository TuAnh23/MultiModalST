from __future__ import division

import datetime
import gc
import inspect
import math
import os
import re
import time
import torch
import numpy as np
from apex import amp

import onmt
import onmt.markdown
import onmt.modules
from onmt.data.data_iterator import DataIterator
from onmt.data.multidata_iterator import MultiDataIterator
from onmt.data.dataset import rewrap
from onmt.model_factory import build_model, build_language_model, optimize_model
from onmt.model_factory import init_model_parameters
from onmt.train_utils.stats import Logger
from onmt.utils import checkpoint_paths, normalize_gradients

from onmt.multiprocessing.multiprocessing_wrapper import MultiprocessingRunner


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def generate_data_iterator(dataset, seed, num_workers=1, epoch=1., buffer_size=0):
    # check if dataset is a list:
    if isinstance(dataset, list):
        # this is a multidataset
        data_iterator = MultiDataIterator(dataset, seed=seed, num_workers=num_workers,
                                          epoch=epoch, buffer_size=buffer_size)
    else:

        data_iterator = DataIterator(dataset, dataset.collater, dataset.batches, seed=seed,
                                     num_workers=num_workers, epoch=epoch, buffer_size=buffer_size)

    return data_iterator


class BaseTrainer(object):

    def __init__(self, model, loss_function, train_data, valid_data, dicts, opt):

        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data

        self.dicts = dicts
        self.opt = opt
        self.cuda = (len(opt.gpus) >= 1 and opt.gpus[0] >= 0)

        self.loss_function = loss_function
        self.start_time = 0

    def add_additional_data(self, d, ratio):
        self.additional_data_train = [data['train_data'] for data in d]
        self.additional_data_valid = [data['valid_data'] for data in d]
        if ratio != "-1":
            self.additional_data_ratio = [int(s) for s in ratio.split(";")]
            # The first element correspond to the ratio of the main data
            assert (len(self.additional_data_ratio) == len(self.additional_data_train) + 1)

            log_str = "Data ratio: [ main data "
            for i in range(0, len(self.additional_data_train)):
                log_str = log_str + f"; addtional data {i} "
            log_str = log_str + f"] = [{ratio}]"
            print(log_str)
        else:
            # If ratio == -1 (i.e. not specified), we will specified the ratio later on such that every dataset is
            # iterated once in each epoch
            self.additional_data_ratio = None
            print("* Data ratio: adjusted so that each dataset is iterated over once in every epoch")

    def run(self, *args, **kwargs):

        raise NotImplementedError

    def eval(self, data):

        raise NotImplementedError

    def load_encoder_weight(self, checkpoint_file):

        print("Loading pretrained models from %s" % checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)

        pretrained_model = build_model(checkpoint['opt'], checkpoint['dicts'])
        pretrained_model.load_state_dict(checkpoint['model'])

        print("Loading pretrained encoder weights ...")
        pretrained_model.encoder.language_embedding = None
        enc_language_embedding = self.model.encoder.language_embedding
        self.model.encoder.language_embedding = None
        encoder_state_dict = pretrained_model.encoder.state_dict()

        self.model.encoder.load_state_dict(encoder_state_dict)
        self.model.encoder.language_embedding = enc_language_embedding
        return

    def load_decoder_weight(self, checkpoint_file):

        print("Loading pretrained models from %s" % checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        chkpoint_dict = checkpoint['dicts']

        pretrained_model = build_model(checkpoint['opt'], chkpoint_dict)
        pretrained_model.load_state_dict(checkpoint['model'])

        print("Loading pretrained decoder weights ...")
        # first we have to remove the embeddings which probably have difference size ...
        pretrained_word_emb = pretrained_model.decoder.word_lut
        pretrained_model.decoder.word_lut = None
        pretrained_lang_emb = pretrained_model.decoder.language_embeddings
        pretrained_model.decoder.language_embeddings = None

        # actually we assume that two decoders have the same language embeddings...
        untrained_word_emb = self.model.decoder.word_lut
        self.model.decoder.word_lut = None
        untrained_lang_emb = self.model.decoder.language_embeddings
        self.model.decoder.language_embeddings = None

        decoder_state_dict = pretrained_model.decoder.state_dict()
        self.model.decoder.load_state_dict(decoder_state_dict)

        # now we load the embeddings ....
        n_copies = 0
        for token in self.dicts['tgt'].labelToIdx:

            untrained_id = self.dicts['tgt'].labelToIdx[token]

            if token in chkpoint_dict['tgt'].labelToIdx:
                pretrained_id = chkpoint_dict['tgt'].labelToIdx[token]
                untrained_word_emb.weight.data[untrained_id].copy_(pretrained_word_emb.weight.data[pretrained_id])

                self.model.generator[0].linear.bias.data[untrained_id].copy_(pretrained_model
                                                                             .generator[0].linear.bias.data[
                                                                                 pretrained_id])
                n_copies += 1

        print("Copied embedding for %d words" % n_copies)
        self.model.decoder.word_lut = untrained_word_emb

        # now we load the language embeddings ...
        if pretrained_lang_emb and untrained_lang_emb and 'langs' in chkpoint_dict:
            for lang in self.dicts['langs']:

                untrained_id = self.dicts['langs'][lang]
                if lang in chkpoint_dict['langs']:
                    pretrained_id = chkpoint_dict['langs'][lang]
                    untrained_lang_emb.weight.data[untrained_id].copy_(pretrained_lang_emb.weight.data[pretrained_id])

        self.model.decoder.language_embeddings = untrained_lang_emb

    def _get_grads(self):
        grads = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                raise RuntimeError('Model parameter did not receive gradient: ' + name + '. '
                                                                                         'Use the param in the forward pass or set requires_grad=False.' +
                                   ' If you are using Stochastic model + fp16 - '
                                   'try to increase the number of minibatches' +
                                   ' each update to avoid uninitialized gradients.')
            grads.append(p.grad.data)
        return grads

    def _get_flat_grads(self, out=None):
        grads = self._get_grads()
        if out is None:
            grads_size = sum(g.numel() for g in grads)
            out = grads[0].new(
                grads_size).zero_()
        offset = 0
        for g in grads:
            numel = g.numel()
            out[offset:offset + numel].copy_(g.view(-1))
            offset += numel
        return out[:offset]

    def warm_up(self):
        """
        Warmup the memory allocator, by attempting to fit the largest batch
        :return:
        """
        if self.opt.memory_profiling:
            from pytorch_memlab import MemReporter
            reporter = MemReporter()

        batch = self.train_data[0].get_largest_batch() if isinstance(self.train_data, list) \
            else self.train_data.get_largest_batch()
        opt = self.opt

        if self.cuda:
            batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

        self.model.train()
        self.model.zero_grad()
        oom = False

        if self.opt.memory_profiling:
            print("Input size: ")
            print(batch.size, batch.src_size, batch.tgt_size)

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        try:
            targets = batch.get('target_output')
            tgt_mask = None
            outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                 zero_encoder=opt.zero_encoder,
                                 mirror=opt.mirror_loss, streaming_state=streaming_state,
                                 nce=opt.nce)

            outputs['tgt_mask'] = tgt_mask

            loss_dict = self.loss_function(outputs, targets, model=self.model, vocab_mask=batch.vocab_mask)
            loss = loss_dict['loss']  # a little trick to avoid gradient overflow with fp16
            full_loss = loss

            if opt.ctc_loss > 0.0:
                ctc_loss = self.ctc_loss_function(outputs, targets)
                ctc_loss_data = ctc_loss.item()
                full_loss = full_loss + opt.ctc_loss * ctc_loss

            if opt.mirror_loss:
                rev_loss = loss_dict['rev_loss']
                mirror_loss = loss_dict['mirror_loss']
                full_loss = full_loss + rev_loss + mirror_loss

            # reconstruction loss
            if opt.reconstruct:
                rec_loss = loss_dict['rec_loss']
                rec_loss = rec_loss
                full_loss = full_loss + rec_loss

            if opt.lfv_multilingual:
                lid_logits = outputs['lid_logits']
                lid_labels = batch.get('target_lang')
                lid_loss_function = self.loss_function.get_loss_function('lid_loss')
                lid_loss = lid_loss_function(lid_logits, lid_labels)
                full_loss = full_loss + lid_loss

            optimizer = self.optim.optimizer

            if self.opt.memory_profiling:
                reporter.report(verbose=True)

                # for obj in gc.get_objects():
                #     try:
                #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #             # print(varname(obj))
                #             # we can rule out parameter cost later
                #             # if 'parameter' not in type(obj):
                #             # if len(obj.shape) == 3:
                #             # if not isinstance(obj, torch.nn.parameter.Parameter):
                #             #     tensor = obj
                #             #     numel = tensor.
                #             print(type(obj), obj.type(), obj.size())
                #     except:
                #         pass

                # print("Memory profiling complete.")
                # print(torch.cuda.memory_summary())
                # exit()

            if self.cuda:
                with amp.scale_loss(full_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.div_(batch.tgt_size).backward()

            if self.opt.memory_profiling:
                print('========= after backward =========')
                reporter.report(verbose=True)

            self.model.zero_grad()
            self.optim.zero_grad()
            # self.optim.step()
            # self.optim.reset()

        except RuntimeError as e:
            if 'out of memory' in str(e):
                oom = True
            else:
                raise e

        if oom:
            print("* Warning: out-of-memory in warming up. This is due to the largest batch is too big for the GPU.")
        else:
            print("* Warming up successuflly.")

        if self.opt.memory_profiling:
            if hasattr(torch.cuda, 'memory_summary'):
                print(torch.cuda.memory_summary())
            exit()


class XETrainer(BaseTrainer):

    def __init__(self, model, loss_function, train_data, valid_data, dicts, opt, setup_optimizer=True,
                 aux_loss_function=None):
        super().__init__(model, loss_function, train_data, valid_data, dicts, opt)

        if opt.lfv_multilingual:
            from onmt.models.speech_recognizer.lid_loss import CrossEntropyLIDLoss
            lid_loss = CrossEntropyLIDLoss(opt.n_languages, opt.label_smoothing, opt.fast_xentropy)
            self.loss_function.add_loss_function(lid_loss, 'lid_loss')

        self.n_gpus = len(self.opt.gpus)

        if opt.ctc_loss != 0:
            from onmt.speech.ctc_loss import CTC
            self.ctc_loss_function = CTC(dicts['tgt'].size(), opt.model_size, 0.0, reduce=True)

        if self.cuda:
            torch.cuda.set_device(self.opt.gpus[0])
            if self.opt.seed >= 0:
                torch.manual_seed(self.opt.seed)
            self.loss_function = self.loss_function.cuda()
            self.model = self.model.cuda()
            if opt.ctc_loss > 0.0:
                self.ctc_loss_function = self.ctc_loss_function.cuda()

        if setup_optimizer:

            self.optim = onmt.Optim(opt)
            self.optim.set_parameters(self.model.parameters())

            if not self.opt.fp16:
                opt_level = "O0"
                keep_batchnorm_fp32 = False
            elif self.opt.fp16_mixed:
                opt_level = "O1"
                keep_batchnorm_fp32 = None
            else:
                opt_level = "O2"
                keep_batchnorm_fp32 = False

            if self.cuda:
                self.model, self.optim.optimizer = amp.initialize(self.model,
                                                                  self.optim.optimizer,
                                                                  opt_level=opt_level,
                                                                  keep_batchnorm_fp32=keep_batchnorm_fp32,
                                                                  loss_scale="dynamic",
                                                                  verbosity=1 if self.opt.verbose else 0)
        # An ugly hack to switch between align right and align left
        if hasattr(self.model, 'relative'):
            if self.model.relative:
                self.train_data.src_align_right = True
                self.train_data.tgt_align_right = False
                self.valid_data.src_align_right = True
                self.valid_data.tgt_align_right = False

        self.aux_loss_function = aux_loss_function

    def save(self, epoch, valid_ppl, itr=None, additional_itrs=None):
        opt = self.opt
        model = self.model
        dicts = self.dicts

        model_state_dict = self.model.state_dict()
        optim_state_dict = self.optim.state_dict()

        if itr:
            itr_state_dict = itr.state_dict()
        else:
            itr_state_dict = None

        if opt.additional_data != 'none' and additional_itrs:
            additional_itr_state_dicts = [additional_itr.state_dict() for additional_itr in additional_itrs]
        else:
            additional_itr_state_dicts = None
        #  drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'dicts': dicts,
            'opt': opt,
            'epoch': epoch,
            'itr': itr_state_dict,
            'additional_itrs': additional_itr_state_dicts,
            'optim': optim_state_dict,
            'amp': amp.state_dict()
        }

        # check the save directory here
        checkpoint_dir = os.path.dirname(opt.save_model)
        existed_save_files = checkpoint_paths(checkpoint_dir)
        for save_file in existed_save_files[opt.keep_save_files - 1:]:
            print(" * Deleting old save file %s ...." % save_file)
            os.remove(save_file)

        file_name = '%s_ppl_%.6f_e%.2f.pt' % (opt.save_model, valid_ppl, epoch)
        print('Writing to %s' % file_name)
        torch.save(checkpoint, file_name)

    def eval(self, data, return_additional_info=False):
        total_loss = 0
        total_words = 0
        opt = self.opt

        self.model.eval()
        self.loss_function.eval()
        self.model.reset_states()

        # the data iterator creates an epoch iterator
        data_iterator = generate_data_iterator(data, seed=self.opt.seed,
                                               num_workers=opt.num_workers, epoch=1, buffer_size=opt.buffer_size)
        epoch_iterator = data_iterator.next_epoch_itr(False, pin_memory=False)

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        """ PyTorch semantics: save space by not creating gradients """

        data_size = len(epoch_iterator)
        i = 0

        with torch.no_grad():
            # for i in range(len()):
            while not data_iterator.end_of_epoch():
                # batch = data.next()[0]
                batch = next(epoch_iterator)
                if isinstance(batch, list):
                    batch = batch[0]
                batch = rewrap(batch)

                if self.cuda:
                    batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

                """ outputs can be either 
                        hidden states from decoder or
                        prob distribution from decoder generator
                """
                targets = batch.get('target_output')
                tgt_mask = targets.ne(onmt.constants.PAD)
                outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                     mirror=opt.mirror_loss, streaming_state=streaming_state, nce=opt.nce)

                if opt.streaming:
                    streaming_state = outputs['streaming_state']

                outputs['tgt_mask'] = tgt_mask

                loss_dict = self.loss_function(outputs, targets, model=self.model, eval=True,
                                               vocab_mask=batch.vocab_mask)

                loss_data = loss_dict['data']

                total_loss += loss_data
                total_words += batch.tgt_size
                i = i + 1

        self.model.train()
        self.loss_function.train()

        if return_additional_info:
            return total_loss, total_words, total_loss / total_words
        else:
            return total_loss / total_words

    def train_epoch(self, epoch, resume=False, itr_progress=None, additional_itrs_progresses=None):

        global rec_ppl
        opt = self.opt
        train_data = self.train_data
        streaming = opt.streaming

        self.model.train()
        self.loss_function.train()
        # Clear the gradients of the model
        # self.runner.zero_grad()
        self.model.zero_grad()
        self.model.reset_states()

        dataset = train_data

        # data iterator: object that controls the
        # data_iterator = DataIterator(dataset, dataset.collater, dataset.batches, seed=self.opt.seed,
        #                              num_workers=opt.num_workers, epoch=epoch, buffer_size=opt.buffer_size)
        data_iterator = generate_data_iterator(dataset, seed=self.opt.seed, num_workers=opt.num_workers,
                                               epoch=epoch, buffer_size=opt.buffer_size)
        if opt.additional_data != 'none':
            additional_data_iterators = [generate_data_iterator(additional_dataset, seed=self.opt.seed,
                                                                num_workers=opt.num_workers, epoch=epoch,
                                                                buffer_size=opt.buffer_size)
                                         for additional_dataset in self.additional_data_train]

        if opt.additional_data != 'none' and self.additional_data_ratio is None:
            # Set the ratio such that every dataset is iterated once every epoch
            numb_of_batches_main_data = len(data_iterator)
            numbs_of_batches_additional_data = [len(x) for x in additional_data_iterators]
            numbs_of_batches_all_data = [numb_of_batches_main_data] + numbs_of_batches_additional_data
            min_numb_batches = min(numbs_of_batches_all_data)
            ratio = [x//min_numb_batches for x in numbs_of_batches_all_data]
            self.additional_data_ratio = ratio

        if self.aux_loss_function is not None:
            # If auxilary loss is used, make sure ASR and MT data has aligned batches
            assert self.additional_data_ratio[0] == self.additional_data_ratio[1]

        if resume:
            data_iterator.load_state_dict(itr_progress)
            if opt.additional_data != 'none':
                if additional_itrs_progresses is not None:
                    assert len(additional_data_iterators) == len(additional_itrs_progresses)
                    for i in range(0, len(additional_data_iterators)):
                        additional_data_iterators[i].load_state_dict(additional_itrs_progresses[i])
                else:
                    for i in range(0, len(additional_data_iterators)):
                        additional_data_iterators[i].load_state_dict(None)

        epoch_iterator = data_iterator.next_epoch_itr(shuffle=not streaming, pin_memory=opt.pin_memory)
        if opt.additional_data != 'none':
            additional_epoch_iterators = [additional_data_iterator.next_epoch_itr(shuffle=not streaming,
                                                                                  pin_memory=opt.pin_memory)
                                          for additional_data_iterator in additional_data_iterators]

        total_tokens, total_loss, total_words = 0, 0, 0
        total_non_pads = 0
        report_loss, report_tgt_words = 0, 0
        report_src_words = 0
        report_ctc_loss = 0
        report_rec_loss, report_rev_loss, report_mirror_loss, report_aux_sim_loss = 0, 0, 0, 0
        start = time.time()
        n_samples = len(epoch_iterator)

        counter = 0
        num_accumulated_words = 0
        num_accumulated_sents = 0
        grad_scaler = 1

        nan = False
        nan_counter = 0

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        i = data_iterator.iterations_in_epoch if not isinstance(train_data, list) else epoch_iterator.n_yielded
        if opt.additional_data != 'none':
            additional_data_i = [additional_data_iterator.iterations_in_epoch if not isinstance(
                additional_train_data, list) else additional_epoch_iterator.n_yielded
                                 for additional_data_iterator, additional_train_data, additional_epoch_iterator
                                 in
                                 zip(additional_data_iterators, self.additional_data_train, additional_epoch_iterators)]

        # This will stores the batches of additional data waiting to be run in between batches of data
        waiting_batches = []

        if self.aux_loss_function is not None:
            # Stores the encoder output of audio to calculate loss difference between text and audio
            audio_context = None
            audio_src_mask = None
            # Stores the target lengths of the audio batch to make sure it is identical with the source length in the
            # text batch (i.e. make sure audio and text sentences are aligned)
            audio_tgt_lengths = None
            # A variable to distinguish MT batch from other additional batches
            mt_batch = False
            # A variable to keep track of when we should skip calculating aux loss (i.e. when the graph is updated
            # and the previous audio output is lost)
            skip_aux_loss = False

        update_flag = False
        while not data_iterator.end_of_epoch():
            curriculum = (epoch < opt.curriculum)

            # this batch generator is not very clean atm
            if len(waiting_batches) == 0:
                # Run the next batch of data
                batch = next(epoch_iterator)
                run_waiting_batch = False
            else:
                # Run the batches in waiting list if any
                batch = waiting_batches.pop(0)
                run_waiting_batch = True

            # Add batches of additional data to the wait list
            if self.aux_loss_function is None:
                if not (run_waiting_batch or data_iterator.end_of_epoch()):
                    if opt.additional_data != 'none' and i % self.additional_data_ratio[0] == 0:
                        for j in range(len(self.additional_data_train)):
                            for k in range(0, self.additional_data_ratio[j + 1]):
                                if additional_data_iterators[j].end_of_epoch():
                                    additional_data_iterators[j] = generate_data_iterator(self.additional_data_train[j],
                                                                                          seed=self.opt.seed,
                                                                                          num_workers=opt.num_workers,
                                                                                          epoch=epoch,
                                                                                          buffer_size=opt.buffer_size)
                                    additional_epoch_iterators[j] = additional_data_iterators[j].next_epoch_itr(shuffle=True,
                                                                                                                pin_memory=opt.pin_memory)
                                waiting_batches.append(next(additional_epoch_iterators[j]))
                                additional_data_i[j] = additional_data_i[j] + 1
            else:
                # If using auxilary loss:
                # Use a different strategy for ordering the batches so that one batch of ASR follows by one batch of MT
                if not (run_waiting_batch or data_iterator.end_of_epoch()):
                    if opt.additional_data != 'none':
                        # Always add a follow up MT batch
                        waiting_batches.append(next(additional_epoch_iterators[0]))
                        additional_data_i[0] = additional_data_i[0] + 1
                        mt_batch = True
                        # Whether to add the other additional batches depends on the specified ratio
                        if i % self.additional_data_ratio[0] == 0:
                            for j in range(1, len(self.additional_data_train)):
                                for k in range(0, self.additional_data_ratio[j + 1]):
                                    if additional_data_iterators[j].end_of_epoch():
                                        additional_data_iterators[j] = generate_data_iterator(self.additional_data_train[j],
                                                                                              seed=self.opt.seed,
                                                                                              num_workers=opt.num_workers,
                                                                                              epoch=epoch,
                                                                                              buffer_size=opt.buffer_size)
                                        additional_epoch_iterators[j] = additional_data_iterators[j].next_epoch_itr(shuffle=True,
                                                                                                                    pin_memory=opt.pin_memory)
                                    waiting_batches.append(next(additional_epoch_iterators[j]))
                                    additional_data_i[j] = additional_data_i[j] + 1

            if isinstance(batch, list) and self.n_gpus == 1:
                batch = batch[0]
            batch = rewrap(batch)
            if self.cuda:
                batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)
            # if opt.streaming:
            #     if train_data.is_new_stream():
            #         streaming_state = self.model.init_stream()
            # else:
            #     streaming_state = None

            oom = False
            try:
                # outputs is a dictionary containing keys/values necessary for loss function
                # can be flexibly controlled within models for easier extensibility
                targets = batch.get('target_output')
                tgt_mask = targets.ne(onmt.constants.PAD)
                outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                     zero_encoder=opt.zero_encoder,
                                     mirror=opt.mirror_loss, streaming_state=streaming_state,
                                     nce=opt.nce)

                batch_size = batch.size

                outputs['tgt_mask'] = tgt_mask

                loss_dict = self.loss_function(outputs, targets, model=self.model, vocab_mask=batch.vocab_mask)
                loss_data = loss_dict['data']
                loss = loss_dict['loss']  # a little trick to avoid gradient overflow with fp16
                full_loss = loss

                if opt.ctc_loss > 0.0:
                    ctc_loss = self.ctc_loss_function(outputs, targets)
                    ctc_loss_data = ctc_loss.item()
                    full_loss = full_loss + opt.ctc_loss * ctc_loss
                    report_ctc_loss += ctc_loss_data

                if opt.mirror_loss:
                    rev_loss = loss_dict['rev_loss']
                    rev_loss_data = loss_dict['rev_loss_data']
                    mirror_loss = loss_dict['mirror_loss']
                    full_loss = full_loss + rev_loss + mirror_loss
                    mirror_loss_data = loss_dict['mirror_loss'].item()
                else:
                    rev_loss_data = None
                    mirror_loss_data = 0

                # reconstruction loss
                if opt.reconstruct:
                    rec_loss = loss_dict['rec_loss']
                    rec_loss = rec_loss
                    full_loss = full_loss + rec_loss
                    rec_loss_data = loss_dict['rec_loss_data']
                else:
                    rec_loss_data = None

                if opt.lfv_multilingual:
                    lid_logits = outputs['lid_logits']
                    lid_labels = batch.get('target_lang')
                    lid_loss_function = self.loss_function.get_loss_function('lid_loss')
                    lid_loss = lid_loss_function(lid_logits, lid_labels)
                    full_loss = full_loss + lid_loss

                optimizer = self.optim.optimizer

                # When the batch size is large, each gradient step is very easy to explode on fp16
                # Normalizing the loss to grad scaler ensures this will not happen
                full_loss.div_(grad_scaler)

                use_aux_loss = epoch >= self.opt.aux_loss_start_from

                if use_aux_loss and self.aux_loss_function is not None:
                    # retain_graph if: (1) not at the end of epoch, (2) ASR or not-skipped MT batch
                    retain_graph = (not data_iterator.end_of_epoch())\
                                   and ((not run_waiting_batch) or (run_waiting_batch and mt_batch and not skip_aux_loss))
                else:
                    retain_graph = None

                if self.cuda:
                    with amp.scale_loss(full_loss, optimizer) as scaled_loss:
                        scaled_loss.backward(retain_graph=retain_graph)
                else:
                    full_loss.backward(retain_graph=retain_graph)

                if use_aux_loss and (self.aux_loss_function is not None):
                    aux_loss_data = 0
                    if run_waiting_batch and mt_batch:
                        if not skip_aux_loss:
                            # Make sure text and audio sentences are aligned
                            assert len(batch.src_lengths) == len(audio_tgt_lengths)
                            for len_i in range(0, len(batch.src_lengths)):
                                # Tgt sentences has 2 additional tokens: bos and eos
                                assert batch.src_lengths[len_i] == audio_tgt_lengths[len_i] - 2

                            # Calculate the difference between the text and audio
                            aux_loss_dict = self.aux_loss_function(audio_context,
                                                                   outputs['context'],
                                                                   audio_src_mask,
                                                                   outputs['src_mask'])
                            aux_loss_data = aux_loss_dict['data']
                            loss = aux_loss_dict['loss'].div_(
                                grad_scaler)  # a little trick to avoid gradient overflow with fp16

                            if self.cuda:
                                with amp.scale_loss(loss, optimizer) as scaled_loss:
                                    scaled_loss.backward()
                            else:
                                loss.backward()
                        else:
                            skip_aux_loss = False

                        mt_batch = False

                    elif not run_waiting_batch:
                        audio_context = outputs['context']  # .clone()
                        audio_src_mask = outputs['src_mask']  # .clone()
                        audio_tgt_lengths = batch.tgt_lengths

                del outputs

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory on GPU , skipping batch')
                    oom = True
                    torch.cuda.empty_cache()
                    loss = 0
                    if opt.streaming:  # reset stream in this case ...
                        streaming_state = self.model.init_stream()
                else:
                    raise e

            if loss != loss:
                # catching NAN problem
                oom = True
                self.model.zero_grad()
                self.optim.zero_grad()
                num_accumulated_words = 0
                num_accumulated_sents = 0
                nan_counter = nan_counter + 1
                print("Warning!!! Loss is Nan")
                if nan_counter >= 15:
                    raise ValueError("Training stopped because of multiple NaN occurence. "
                                     "For ASR, using the Relative Transformer is more stable and recommended.")
            else:
                nan_counter = 0

            if not oom:
                src_size = batch.src_size
                tgt_size = batch.tgt_size

                counter = counter + 1
                num_accumulated_words += tgt_size
                num_accumulated_sents += batch_size

                #   We only update the parameters after getting gradients from n mini-batches
                update_flag = False
                if counter >= opt.update_frequency > 0:
                    update_flag = True
                elif 0 < opt.batch_size_update <= num_accumulated_words:
                    update_flag = True
                elif i == n_samples:  # update for the last minibatch
                    update_flag = True

                if update_flag:
                    if use_aux_loss and self.aux_loss_function is not None:
                        if not run_waiting_batch:
                            # Skip the next aux loss because when we update the graph, we cannot use the previously
                            # stored audio encoder output anymore
                            skip_aux_loss = True

                    # accumulated gradient case, in this case the update frequency
                    grad_denom = 1 / grad_scaler
                    if self.opt.normalize_gradient:
                        grad_denom = num_accumulated_words * grad_denom
                    # When we accumulate the gradients, each gradient is already normalized by a constant grad_scaler
                    normalize_gradients(amp.master_params(optimizer), grad_denom)
                    # Update the parameters.
                    if self.opt.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.opt.max_grad_norm)
                    self.optim.step()
                    self.optim.zero_grad()
                    self.model.zero_grad()
                    counter = 0
                    num_accumulated_words = 0
                    num_accumulated_sents = 0
                    num_updates = self.optim._step
                    if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every:
                        valid_total_loss, valid_total_words, valid_loss = self.eval(self.valid_data,
                                                                                    return_additional_info=True)
                        valid_ppl = math.exp(min(valid_loss, 100))
                        print('Validation perplexity: %g' % valid_ppl)

                        if opt.additional_data != 'none':
                            combined_total_loss = valid_total_loss
                            combined_total_words = valid_total_words
                            for i_additional in range(0, len(self.additional_data_valid)):
                                additional_valid_total_loss, additional_valid_total_words, additional_valid_loss = \
                                    self.eval(self.additional_data_valid[i_additional], return_additional_info=True)
                                additional_valid_ppl = math.exp(min(additional_valid_loss, 100))
                                print('Validation perplexity on additional data %d: %g' % (i_additional,
                                                                                           additional_valid_ppl))
                                combined_total_loss = combined_total_loss + additional_valid_total_loss
                                combined_total_words = combined_total_words + additional_valid_total_words

                            combined_valid_ppl = math.exp(min(combined_total_loss / combined_total_words, 100))
                            print('Validation perplexity combined: %g' % combined_valid_ppl)

                        ep = float(epoch) - 1. + ((float(i) + 1.) / n_samples)
                        if opt.additional_data != 'none':
                            self.save(ep, combined_valid_ppl, itr=data_iterator,
                                      additional_itrs=additional_data_iterators)
                        else:
                            self.save(ep, valid_ppl, itr=data_iterator)

                num_words = tgt_size
                report_loss += loss_data
                report_tgt_words += num_words
                report_src_words += src_size
                total_loss += loss_data
                total_words += num_words
                total_tokens += batch.get('target_output').nelement()
                total_non_pads += batch.get('target_output').ne(onmt.constants.PAD).sum().item()
                optim = self.optim
                batch_efficiency = total_non_pads / total_tokens

                if opt.reconstruct:
                    report_rec_loss += rec_loss_data

                if opt.mirror_loss:
                    report_rev_loss += rev_loss_data
                    report_mirror_loss += mirror_loss_data

                if use_aux_loss and self.aux_loss_function is not None:
                    report_aux_sim_loss += aux_loss_data

                if (i == 0 or (i % opt.log_interval == -1 % opt.log_interval)) and (not run_waiting_batch):
                    log_string = ("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; " %
                                  (epoch, i + 1, len(data_iterator),
                                   np.exp(report_loss / report_tgt_words)))

                    if opt.reconstruct:
                        rec_ppl = math.exp(report_rec_loss / report_src_words.item())
                        log_string += (" rec_ppl: %6.2f ; " % rec_ppl)

                    if opt.mirror_loss:
                        rev_ppl = math.exp(report_rev_loss / report_tgt_words)
                        log_string += (" rev_ppl: %6.2f ; " % rev_ppl)
                        # mirror loss per word
                        log_string += (" mir_loss: %6.2f ; " % (report_mirror_loss / report_tgt_words))

                    if epoch >= self.opt.aux_loss_start_from:
                        log_string += (" Aux loss: %6.2f ; " % (report_aux_sim_loss / report_src_words))

                    log_string += ("lr: %.7f ; updates: %7d; " %
                                   (optim.getLearningRate(),
                                    optim._step))

                    log_string += ("%5.0f src tok/s; %5.0f tgt tok/s; " %
                                   (report_src_words / (time.time() - start),
                                    report_tgt_words / (time.time() - start)))

                    if opt.ctc_loss > 0.0:
                        # if torch.isinf(report_ctc_loss):
                        #     report_ctc_loss.zero_()
                        # dist.all_reduce(report_ctc_loss, op=dist.ReduceOp.SUM, group=self.group)
                        ctc_loss = report_ctc_loss / report_tgt_words
                        log_string += (" ctcloss: %8.2f ; " % ctc_loss)

                    log_string += ("%s elapsed" %
                                   str(datetime.timedelta(seconds=int(time.time() - self.start_time))))

                    print(log_string)

                    report_loss = 0
                    report_tgt_words, report_src_words = 0, 0
                    report_rec_loss, report_rev_loss, report_mirror_loss, report_aux_sim_loss = 0, 0, 0, 0
                    report_ctc_loss = 0
                    start = time.time()

                if not run_waiting_batch:
                    i = i + 1

        return total_loss / total_words

    # def run(self, save_file=None):
    def run(self, checkpoint=None):

        opt = self.opt
        model = self.model
        optim = self.optim

        if checkpoint is not None:
            self.model.load_state_dict(checkpoint['model'])
            prec_opt = checkpoint['opt'] if 'opt' in checkpoint else None
            if not opt.reset_optim:
                print("* Loading optimizer states ... ")
                self.optim.load_state_dict(checkpoint['optim'])
                if prec_opt is not None and hasattr(prec_opt, "fp16_mixed"):
                    # Only load amp information if the mode is the same
                    # Maybe its better to change between optimization mode?
                    if opt.fp16_mixed == prec_opt.fp16_mixed and opt.fp16 == prec_opt.fp16:
                        if 'amp' in checkpoint:
                            amp.load_state_dict(checkpoint['amp'])

                # Only load the progress when we use the same optimizer
                if 'itr' in checkpoint:
                    itr_progress = checkpoint['itr']
                else:
                    itr_progress = None

                if opt.additional_data != 'none' and 'additional_itrs' in checkpoint:
                    additional_itrs_progresses = checkpoint['additional_itrs']
                else:
                    additional_itrs_progresses = None

                resume = True
                if prec_opt is not None and prec_opt.save_every == -1:
                    # Last epoch was completed entirely, so we move on to the next epoch
                    start_epoch = checkpoint['epoch'] + 1 if 'epoch' in checkpoint else 1
                else:
                    start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 1
                if start_epoch is None:
                    start_epoch = 1
            else:
                itr_progress = None
                additional_itrs_progresses = None
                resume = False
                start_epoch = 1

            del checkpoint['model']
            del checkpoint['optim']
            del checkpoint
        else:
            itr_progress = None
            additional_itrs_progresses = None
            print('Initializing model parameters')
            init_model_parameters(model, opt)
            resume = False
            start_epoch = 1

        if opt.load_encoder_from:
            self.load_encoder_weight(opt.load_encoder_from)

        if opt.load_decoder_from:
            self.load_decoder_weight(opt.load_decoder_from)

        # if we are on a GPU: warm up the memory allocator
        if self.cuda:
            self.warm_up()

            valid_total_loss, valid_total_words, valid_loss = self.eval(self.valid_data,
                                                                        return_additional_info=True)
            valid_ppl = math.exp(min(valid_loss, 100))
            print('Validation perplexity: %g' % valid_ppl)

            if opt.additional_data != 'none':
                combined_total_loss = valid_total_loss
                combined_total_words = valid_total_words
                for i_additional in range(0, len(self.additional_data_valid)):
                    additional_valid_total_loss, additional_valid_total_words, additional_valid_loss = \
                        self.eval(self.additional_data_valid[i_additional], return_additional_info=True)
                    additional_valid_ppl = math.exp(min(additional_valid_loss, 100))
                    print('Validation perplexity on additional data %d: %g' % (i_additional,
                                                                               additional_valid_ppl))
                    combined_total_loss = combined_total_loss + additional_valid_total_loss
                    combined_total_words = combined_total_words + additional_valid_total_words

                combined_valid_ppl = math.exp(min(combined_total_loss / combined_total_words, 100))
                print('Validation perplexity combined: %g' % combined_valid_ppl)

        self.start_time = time.time()
        for epoch in range(start_epoch, start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            if opt.additional_data != 'none':
                train_loss = self.train_epoch(epoch, resume=resume, itr_progress=itr_progress,
                                              additional_itrs_progresses=additional_itrs_progresses)
            else:
                train_loss = self.train_epoch(epoch, resume=resume, itr_progress=itr_progress)
            train_ppl = math.exp(min(train_loss, 100))
            print('Train perplexity: %g' % train_ppl)

            #  (2) evaluate on the validation set
            valid_total_loss, valid_total_words, valid_loss = self.eval(self.valid_data,
                                                                        return_additional_info=True)
            valid_ppl = math.exp(min(valid_loss, 100))
            print('Validation perplexity: %g' % valid_ppl)

            if opt.additional_data != 'none':
                combined_total_loss = valid_total_loss
                combined_total_words = valid_total_words
                for i_additional in range(0, len(self.additional_data_valid)):
                    additional_valid_total_loss, additional_valid_total_words, additional_valid_loss = \
                        self.eval(self.additional_data_valid[i_additional], return_additional_info=True)
                    additional_valid_ppl = math.exp(min(additional_valid_loss, 100))
                    print('Validation perplexity on additional data %d: %g' % (i_additional,
                                                                               additional_valid_ppl))
                    combined_total_loss = combined_total_loss + additional_valid_total_loss
                    combined_total_words = combined_total_words + additional_valid_total_words

                combined_valid_ppl = math.exp(min(combined_total_loss / combined_total_words, 100))
                print('Validation perplexity combined: %g' % combined_valid_ppl)

            if opt.additional_data != 'none':
                self.save(epoch, combined_valid_ppl)
            else:
                self.save(epoch, valid_ppl)
            itr_progress = None
            additional_itrs_progresses = None
            resume = False

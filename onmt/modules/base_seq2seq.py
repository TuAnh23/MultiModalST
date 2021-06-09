import torch
import torch.nn as nn
import torch.nn.functional as F
import onmt, math


class Generator(nn.Module):

    def __init__(self, hidden_size, output_size, fix_norm=False):
        
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear = nn.Linear(hidden_size, output_size)
        self.fix_norm = fix_norm
        
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        
        torch.nn.init.uniform_(self.linear.weight, -stdv, stdv)
        
        self.linear.bias.data.zero_()

    def forward(self, output_dicts):
        """
        :param output_dicts: dictionary contains the outputs from the decoder
        :return: logits (the elements before softmax)
        """

        input = output_dicts['hidden']
        fix_norm = self.fix_norm
        target_mask = output_dicts['target_mask']

        if not fix_norm:
            logits = self.linear(input)
        else:
            normalized_weights = F.normalize(self.linear.weight, dim=-1)
            normalized_bias = self.linear.bias
            logits = F.linear(input, normalized_weights, normalized_bias)

        # softmax will be done at the loss function
        # output = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        output_dicts['logits'] = logits
        return output_dicts
        

class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, generator=None, rec_decoder=None, rec_generator=None,
                 mirror=False, ctc=False):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.rec_decoder = rec_decoder
        self.rec_generator = rec_generator
        self.ctc = ctc

        if self.rec_decoder:
            self.rec_decoder.word_lut.weight = self.encoder.word_lut.weight
            self.reconstruct = True
        else:
            self.reconstruct = False

    def tie_weights(self):
        assert self.generator is not None, "The generator needs to be created before sharing weights"
        self.generator[0].linear.weight = self.decoder.word_lut.weight

    def share_enc_dec_embedding(self):
        self.encoder.word_lut.weight = self.decoder.word_lut.weight
        
    def mark_pretrained(self):
        
        self.encoder.mark_pretrained()
        self.decoder.mark_pretrained()
        
    def load_state_dict(self, state_dict, strict=True):
        """
        override this method to have back-compatibility
        """

        model_dict = self.state_dict()
        
        def condition(param_name):
            # don't load these buffers (more like a bug)
            if 'positional_encoder' in param_name:
                return False
            if 'time_transformer' in param_name:
                if self.encoder is not None:
                    if type(self.encoder).__name__ == 'MixedEncoder':
                        if self.encoder.text_encoder.time == 'positional_encoding' or \
                                self.encoder.audio_encoder.time == 'positional_encoding':
                            return False
                    else:
                        if self.encoder.time == 'positional_encoding':
                            return False
            if param_name == 'decoder.mask':
                return False
            if param_name == 'decoder.r_w_bias' or param_name == 'decoder.r_r_bias':
                if param_name in model_dict:
                    return True
                return False
            
            return True

        # restore old generated if necessary for loading
        if "generator.linear.weight" in state_dict and type(self.generator) is nn.ModuleList:
            self.generator = self.generator[0]

        # only load the filtered parameters
        filtered = {k: v for k, v in state_dict.items() if condition(k)}

        for k, v in model_dict.items():
            if k not in filtered:
                filtered[k] = v

        super().load_state_dict(filtered)   

        # in case using multiple generators
        if type(self.generator) is not nn.ModuleList:
            self.generator = nn.ModuleList([self.generator])


class Reconstructor(nn.Module):
    """
    This class is currently unused, but can be used to learn to reconstruct from the hidden states
    """
    
    def __init__(self, decoder, generator=None):
        super(Reconstructor, self).__init__()
        self.decoder = decoder
        self.generator = generator


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.
    Modules need to implement this to utilize beam search decoding.
    """

    def update_beam(self, beam, b, remaining_sents, idx):

        raise NotImplementedError

    def prune_complete_beam(self, active_idx, remaining_sents):

        raise NotImplementedError

class Classifier(nn.Module):

    def __init__(self, hidden_size, output_size, fix_norm=False, grad_scale=1.0, mid_layer_size=0, input_name='context'):

        super(Classifier, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mid_layer_size = mid_layer_size
        self.fix_norm = fix_norm

        if self.mid_layer_size == 0:
            self.linear = nn.Linear(hidden_size, output_size)
            stdv = 1.0 / math.sqrt(self.linear.weight.size(1))
            torch.nn.init.uniform_(self.linear.weight, -stdv, stdv)

            self.linear.bias.data.zero_()

        else:
            self.relu = torch.nn.ReLU()
            self.linear = nn.Linear(hidden_size, mid_layer_size)
            self.linear2 = nn.Linear(mid_layer_size, output_size)

            for w in [self.linear, self.linear2]:
                stdv = 1.0 / math.sqrt(w.weight.size(1))
                torch.nn.init.uniform_(w.weight, -stdv, stdv)
                w.bias.data.zero_()

        self.grad_scale = grad_scale
        # self.reversed_loss_landscape = reversed_loss_landscape
        self.input_name = input_name  # define input to classifier

    # def forward(self, input, log_softmax=True):
    def forward(self, output_dicts, hidden_name='hidden'):

        classifier_input = output_dicts[self.input_name]
        fix_norm = self.fix_norm

        # if self.training:
        #     classifier_input = grad_reverse(classifier_input, self.grad_scale)

        # added float to the end
        if self.mid_layer_size == 0:
            if not fix_norm:
                logits = self.linear(classifier_input).float()
            else:
                normalized_weights = F.normalize(self.linear.weight, dim=-1)
                normalized_bias = self.linear.bias
                logits = F.linear(classifier_input, normalized_weights, normalized_bias).float()
        else:
            if not fix_norm:
                hid = self.linear(classifier_input).float()
                relu = self.relu(hid)
                logits = self.linear2(relu)
            else:
                raise NotImplementedError

        # if self.reversed_loss_landscape:
        #     output = F.log_softmax(1.0 - (F.softmax(logits, dim=-1)), dim=-1)  # reversed, when training adversarially  # dtype=torch.float32
        # else:
        output = F.log_softmax(logits, dim=-1)
        return output


# class GradReverse(torch.autograd.Function):
#     scale = 1.0
#
#     @staticmethod
#     def forward(self, x):
#         return x.view_as(x)
#
#     @staticmethod
#     def backward(self, grad_output):
#         return GradReverse.scale * grad_output.neg()
#
# def grad_reverse(x, scale=1.0):
#     GradReverse.scale = scale
#     return GradReverse.apply(x)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tokens_process as tkp

from model_msnovelist.components.Hint import myRNN
from model_msnovelist.components.decoder import SequenceDecoder
from model_msnovelist.components.encoder import FingerprintFormulaEncoder
from model_msnovelist.components.hydrogen_estimator import HydrogenEstimator
from model_msnovelist.components.recurrent_additive import RecurrentAdditiveCell
from model_msnovelist.components.sum_pooling import GlobalSumPooling


class BlueprintedModel(nn.Module):
    class FlattenStatesLayer(nn.Module):
        """
        Go from the states in the model_msnovelist format (z, vi, SG, SH) to a single vector.
        """
        def __init__(self):
            super(BlueprintedModel.FlattenStatesLayer, self).__init__()

        def forward(self, inputs):
            """
            inputs : dictionary with {'auxiliary_counter_states',
                                      'rnn_states',
                                      'hydrogen_estimator_states',
                                      'z',
                                      'tokens_X'}
            """
            # rnn_states = inputs['rnn_states']  # [[h, c],[h, c],[h, c]]  # h,c:(batch_size, hid_size=256)
            # rnn_states = [torch.stack(x, dim=1) for x in rnn_states]  # [(batch_size, 2, hid_size=256)*3]
            # rnn_states = torch.stack(rnn_states, dim=1)  # (batch_size, layers=3, 2, hid_size=256)
            # rnn_states = rnn_states.view(rnn_states.size(0), -1)  # (batch_size, 1536=3*2*256)

            # h,c:(num_layers=3, batch_size, hid_size=256)
            rnn_states = inputs['rnn_states']
            # (batch_size, layers=3, 2, hid_size=256)
            rnn_states = torch.stack(rnn_states, dim=2).transpose(0, 1)
            # (batch_size, 1536=3*2*256)
            rnn_states = rnn_states.reshape(rnn_states.size(0), -1)

            # hydrogen_estimator_states = inputs['hydrogen_estimator_states']  # [[h, c],[h, c]]
            # hydrogen_estimator_states = [torch.stack(x, dim=1)  # (layers=2, (h, c))  # h,c:(batch_size, hid_size=32)
            #                              for x in hydrogen_estimator_states]
            # hydrogen_estimator_states = torch.stack(hydrogen_estimator_states, dim=1)
            # hydrogen_estimator_states = hydrogen_estimator_states.view(hydrogen_estimator_states.size()[0],
            #                                                            -1)  # (batch_size, 2*2*32)

            # h,c:(num_layers=2, batch_size, hid_size=32)
            hydrogen_estimator_states = inputs['hydrogen_estimator_states']
            # (batch_size, layers=2, 2, hid_size=32)
            hydrogen_estimator_states = torch.stack(hydrogen_estimator_states, dim=2).transpose(0, 1)
            # (batch_size, 128=2*2*32)
            hydrogen_estimator_states = hydrogen_estimator_states.reshape(hydrogen_estimator_states.size(0), -1)

            states = torch.cat([
                inputs['auxiliary_counter_states'],  # (batch_size, 11)
                hydrogen_estimator_states,  # (batch_size, 128=2*2*32)
                rnn_states,  # (batch_size, 1536=3*2*256)
                inputs['z']  # (batch_size, 256)
            ], dim=1)
            return states  # (batch_size, 1931)

    class UnflattenStatesLayer(nn.Module):
        def __init__(self, model):
            super(BlueprintedModel.UnflattenStatesLayer, self).__init__()
            self.states = {
                'auxiliary_counter_states': (model.auxiliary_counter_units,),  # (11,)
                'hydrogen_estimator_states': (   # (2, 2, 32)
                    model.config['hcounter_layers'],
                    2,
                    model.config['hcount_hidden_size']
                ),
                'rnn_states': (    # (3, 2, 256)
                    model.config['decoder_layers'],
                    2,
                    model.config['decoder_hidden_size']
                ),
                'z': (model.config['fp_enc_layers'][-1],)  # (256,)
            }
            self.states_shapes = [(-1,) + shape for shape in self.states.values()]
            self.states_length = [np.prod(spec) for spec in self.states.values()]
            self.states_pos = np.concatenate([[0], np.cumsum(self.states_length[:-1])])

        def forward(self, inputs):
            states_split = [
                inputs[:, start:start + length]
                for start, length in zip(self.states_pos, self.states_length)
            ]

            states_reshape = [state.view(shape) for state, shape in zip(states_split, self.states_shapes)]

            # (batch_size, num_layers=2, 2, 32)
            hydrogen_estimator_states = states_reshape[1]
            # (num_layers=2, batch_size, 2, 32)
            hydrogen_estimator_states = hydrogen_estimator_states.transpose(0, 1)
            # [h, c]  h,c: (num_layers=2, batch_size, 32)
            hydrogen_estimator_states = torch.unbind(hydrogen_estimator_states, dim=2)
            hydrogen_estimator_states = [state.contiguous() for state in hydrogen_estimator_states]

            # (batch_size, num_layers=3, 2, 256)
            rnn_states = states_reshape[2]
            # (num_layers=3, batch_size, 2, 256)
            rnn_states = rnn_states.transpose(0, 1)
            # [h, c] h,c: (num_layers=3, batch_size, 256)
            rnn_states = torch.unbind(rnn_states, dim=2)
            rnn_states = [state.contiguous() for state in rnn_states]
            return {
                'auxiliary_counter_states': states_reshape[0],
                'hydrogen_estimator_states': hydrogen_estimator_states,
                'rnn_states': rnn_states,
                'z': states_reshape[3]
            }

    # def copy_weights(self, model_msnovelist):
    #     layers_ref = [name for name, _ in model_msnovelist.named_modules()]
    #
    #     for name, layer in self.named_modules():
    #         if name in layers_ref:
    #             layer_ = dict(model_msnovelist.named_modules())[name]
    #             layer.load_state_dict(layer_.state_dict())

    def construct_counter_matrix(self):
        m11 = tkp.ELEMENT_MAP
        m13 = tkp.GRAMMAR_MAP
        m12 = torch.zeros_like(m13)
        m_up = torch.cat([m11, m12, m13], dim=1)
        m21 = torch.zeros_like(m11[:1, :])
        m22 = torch.ones_like(m12[:1, :])
        m23 = torch.zeros_like(m13[:1, :])
        m_down = torch.cat([m21, m22, m23], dim=1)
        return torch.cat([m_up, m_down], dim=0)

    def __init__(self,
                 config=None,
                 return_states=False,
                 round_fingerprints=True,
                 steps=None,
                 **kwargs):
        super(BlueprintedModel, self).__init__()

        # ====================== 超参配置 =======================
        if config is None:
            config = {}
        self.config = { # (origin)
            'decoder_hidden_size': 256,
            'hcount_hidden_size': 32,
            'fp_enc_layers': [512, 256],
            'loss_weights': {'out_smiles': 1, 'out_nhydrogen': 0.03},
            'hcounter_layers': 2,
            'decoder_layers': 3,
            'decoder_input_size': 306,
            'use_hydrogen_estimator': True,
            'use_auxiliary_counter': True,
            'use_fingerprint': True,
            "out_size": 39,
            "seq_len": 127
        }
        # self.config = { # 扩大lstm的hidden_size会提升retrieval，但是会降低correct MF
        #     'decoder_hidden_size': 512,  # 256(origin), 512
        #     'hcount_hidden_size': 32,
        #     'fp_enc_layers': [512, 512],  # [512, 256](origin)
        #     'loss_weights': {'out_smiles': 1, 'out_nhydrogen': 0.03},
        #     'hcounter_layers': 2,
        #     'decoder_layers': 3,
        #     'decoder_input_size': 562,
        #     'use_hydrogen_estimator': True,
        #     'use_auxiliary_counter': True,
        #     'use_fingerprint': True,
        #     "out_size": 39,
        #     "seq_len": 127
        # }
        if "model_config" in config:
            config_ = config["model_config"]
            self.config.update(config_)

        # ====================== 变量 =======================
        self.round_fingerprints = round_fingerprints
        # (1,1,39)
        self.initial_char = torch.unsqueeze(torch.from_numpy(tkp.tokens2vec([tkp.INITIAL_CHAR])), 0)
        self.pad_mask = torch.unsqueeze(torch.from_numpy(tkp.tokens2vec([tkp.PAD_CHAR])), 0)
        self.final_mask = torch.unsqueeze(torch.from_numpy(tkp.tokens2vec([tkp.FINAL_CHAR])), 0)

        self.hcount_mask = torch.ones_like(self.pad_mask) - self.pad_mask - self.final_mask

        self.counter_matrix = self.construct_counter_matrix().cuda()

        self.tokens_output = self.config['out_size']  # 39
        self.auxiliary_counter_units = self.counter_matrix.shape[1]  # 11
        self.steps = steps or self.config['seq_len']  # 127

        # ====================== 模型 =======================
        self.encoder = FingerprintFormulaEncoder(
            layers_hid_size=self.config['fp_enc_layers'],  # [512, 256]
            layers_decoder=self.config['decoder_layers'],  # 3
            units_decoder=self.config['decoder_hidden_size'],  # 256
        )

        self.hydrogen_estimator = HydrogenEstimator(
            layers=self.config['hcounter_layers'],  # 2
            units=self.config['hcount_hidden_size'],  # 32
            pad_term_mask=self.hcount_mask,  # (1,1,39)
            return_states=return_states  # False
        )
        self.hydrogen_sum = GlobalSumPooling()
        self.gradient_stop = lambda x: x.detach()

        self.auxiliary_counter_input_transformer = lambda x: torch.matmul(x, self.counter_matrix)

        self.auxiliary_counter_start_state_transformer = nn.ZeroPad2d((0, 1, 0, 0))

        self.auxiliary_counter = myRNN(RecurrentAdditiveCell(
            units=self.auxiliary_counter_units,  # 11
            factor=-1),
            return_state=return_states,  # False
            return_sequences=True
        )

        self.sequence_decoder = SequenceDecoder(
            input_size=self.config['decoder_input_size'],  # 306
            output_size=self.tokens_output,
            num_layers=self.config['decoder_layers'],  # 3
            hidden_size=self.config['decoder_hidden_size'],  # 256
            return_states=return_states,  # False
        )

        self.z_time_transformer = lambda x: x.unsqueeze(1).repeat(1, self.steps, 1)

        if self.round_fingerprints:
            self.fingerprint_rounding = lambda x: torch.round(x)
        else:
            self.fingerprint_rounding = lambda x: x

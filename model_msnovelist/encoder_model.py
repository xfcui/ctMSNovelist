import torch
import numpy as np

from .blueprinted_model import BlueprintedModel


class EncoderModel(BlueprintedModel):
    def __init__(self, **kwargs):
        super(EncoderModel, self).__init__(**kwargs)
        # (num_layers, batch_size, 32)
        self.hydrogen_estimator_one_state_creator = lambda x: torch.zeros(
            (self.config['hcounter_layers'], x, self.config['hcount_hidden_size']))

        # [[h,c],[h,c],[h,c]]
        # self.hydrogen_estimator_states_creator = lambda x: [[x, x]] * self.config['hcounter_layers']

        # (1,1,39) --> (batch_size, 1, 39)
        self.initial_token_creator = lambda x: self.initial_char.repeat(x.size(0), 1, 1)

        self.flatten_states_layer = self.FlattenStatesLayer()

    def forward(self, inputs):
        """
        :param inputs:
            dictionary with {'fingerprint', 'mol_form'}
        :return:
            dictionary with {'auxiliary_counter_states',
                            'rnn_states',
                            'hydrogen_estimator_states',
                            'z',
                            'tokens_y'}
        """
        batch_size = inputs['MF'].size(0)
        fingerprints = self.fingerprint_rounding(inputs['FP'])   # (batch_size, 3609)
        # (batch_size, 256)  [h, c] h,c: (num_layers=3, batch_size, 256)
        z, rnn_states = self.encoder([fingerprints, inputs['MF']])

        # (num_layers, batch_size, hid_size)
        hydrogen_estimator_h_state = self.hydrogen_estimator_one_state_creator(batch_size).cuda()
        hydrogen_estimator_c_state = self.hydrogen_estimator_one_state_creator(batch_size).cuda()
        # (h,c)
        hydrogen_estimator_states = (hydrogen_estimator_h_state, hydrogen_estimator_c_state)

        # (batch_size, 11)
        auxiliary_counter_states = self.auxiliary_counter_start_state_transformer(inputs['MF'])

        initial_tokens = self.initial_token_creator(inputs['MF']).cuda()  # (batch_size, 1, 39)

        states_out = {
            'auxiliary_counter_states': auxiliary_counter_states,  # (batch_size, 11)
            'rnn_states': rnn_states,  # (h, c)
            'hydrogen_estimator_states': hydrogen_estimator_states,  # (h, c)
            'z': z   # (batch_size, 256)
        }

        states_flat = self.flatten_states_layer(states_out)

        return {
            'tokens_X': initial_tokens,  # (batch_size,1,39)
            'states': states_flat,  # states_flat
            # 'counts': auxiliary_counter_states
        }

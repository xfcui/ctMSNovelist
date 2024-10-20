import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .blueprinted_model import BlueprintedModel


class DecoderModel(BlueprintedModel):
    def __init__(self, **kwargs):
        super(DecoderModel, self).__init__(**kwargs,
                                           return_states=True,
                                           steps=1)
        self.flatten_states_layer = self.FlattenStatesLayer()
        self.unflatten_states_layer = self.UnflattenStatesLayer(self)

    def forward(self, inputs):
        """
        :param inputs: dictionary with {'tokens_X', 'states', 'counts'}
        :return:
            dict
                a dictionary exactly like the inputs.
        """
        states_flat = inputs['states']
        states = self.unflatten_states_layer(states_flat)

        estimated_h_count, hydrogen_estimator_states = self.hydrogen_estimator(
            inputs['tokens_X'],
            initial_state=states['hydrogen_estimator_states'])

        auxiliary_counter_input = torch.cat([inputs['tokens_X'],
                                             estimated_h_count], dim=-1)

        auxiliary_counter_input_transformed = self.auxiliary_counter_input_transformer(
            auxiliary_counter_input)

        # element_grammar_count:(batch_size,1,11)
        # auxiliary_counter_states:(batch_size,11)
        element_grammar_count, auxiliary_counter_states = self.auxiliary_counter(
            auxiliary_counter_input_transformed,
            initial_state=states['auxiliary_counter_states'])

        z_repeated = self.z_time_transformer(states['z'])

        decoder_input = torch.cat([inputs['tokens_X'],
                         element_grammar_count,
                         z_repeated], dim=-1)

        decoder_out, rnn_states = self.sequence_decoder(
            decoder_input,
            initial_state=states['rnn_states'])

        states_out = {
            'auxiliary_counter_states': auxiliary_counter_states,
            'rnn_states': rnn_states,
            'hydrogen_estimator_states': hydrogen_estimator_states,
            'z': states['z']
        }

        states_out_flat = self.flatten_states_layer(states_out)

        return {'tokens_X': decoder_out,
                'states': states_out_flat,  # (batch_size, 1931)
                'counts': auxiliary_counter_states  # (batch_size, 11)
                }

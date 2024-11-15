import numpy as np
import torch
from torch import nn

from model_msnovelist.blueprinted_model import BlueprintedModel


class TranscoderModel(BlueprintedModel):
    def __init__(self, **kwargs):
        super(TranscoderModel, self).__init__(**kwargs)

    def forward(self, inputs):
        # ======================= encoder =============================
        formula = inputs['MF']  # (batch_size=256, 10)
        fingerprints = inputs['FP']  # (batch_size=256, 3609)
        fingerprints_ = self.fingerprint_rounding(fingerprints)

        # z: (batch_size, 256)  decoder_initial_states: [[h,c]*3]
        z, rnn_states = self.encoder([fingerprints_, formula])

        # rnn_states_ = [item.cpu().detach() for item in rnn_states]
        # rnn_states = [item.cuda() for item in rnn_states_]
        rnn_states = [item.detach() for item in rnn_states]
        # ======================= H_count =============================
        # 预测每个时刻 H 的数目和 H 总数
        # (256, 127, 39)  --> estimated_h_count: (256, 127, 1)
        estimated_h_count = self.hydrogen_estimator(inputs['tokens_X'])

        # (256, 127, 1)  --> estimated_h_sum: (256, 1)
        estimated_h_sum = self.hydrogen_sum(estimated_h_count)

        # 梯度停止层用于阻止梯度回传到estimated_h_count
        estimated_h_count_ = self.gradient_stop(estimated_h_count)
        # ======================= Hint =============================
        # input
        # (256, 127, 39)  (256, 127, 1)  -->  (256, 127, 40)
        auxiliary_counter_input = torch.cat([inputs['tokens_X'], estimated_h_count_], dim=-1)

        # (256, 127, 40) --> (256, 127, 11)  [voc, H] --> [elem*9, H, ()]
        auxiliary_counter_input_transformed = self.auxiliary_counter_input_transformer(
            auxiliary_counter_input)

        # state
        # (256, 10) --> (256, 11=10+1)  [elem*9, H, ()]
        auxiliary_counter_start_states = self.auxiliary_counter_start_state_transformer(inputs['MF'])

        # (256, 127, 11) --> (256,127,11) 计算每个时刻剩余的 原子 和 括号 的数目
        element_grammar_count = self.auxiliary_counter(
            auxiliary_counter_input_transformed,
            initial_state=auxiliary_counter_start_states)

        # ======================= beam_model =============================
        # (batch_size, 256) --> (batch_size, 127, 256)
        z_repeated = self.z_time_transformer(z)

        # (batch_size, 127, 306)
        decoder_input = [inputs['tokens_X'],  # (batch_size, seq_len, 39)
                         element_grammar_count,  # (batch_size, seq_len, 11)
                         z_repeated]    # (batch_size, seq_len, 256)

        # (256, 127, 39)
        decoder_out = self.sequence_decoder(decoder_input, rnn_states)

        # (256, 127, 39)  (256, 1)
        return decoder_out, estimated_h_sum  # , element_grammar_count

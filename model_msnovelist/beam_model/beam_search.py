import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import importlib

from model_msnovelist.beam_model.seqManager import SequenceManager


class myBeamDecoder:
    def __init__(self, encoder, decoder, steps, n, k, kk, temperature=1., config=None):
        if config is None:
            config = {}
        self.config = {
            'tokenization': 'smiles_process',
            'clip_invalid_counts': True,
            'sequence_length_correction': False
        }
        if "decoder_config" in config:
            config_ = config["decoder_config"]
            self.config.update(config_)

        self.tokenization = importlib.import_module(self.config['tokenization'])
        self.clip_invalid_counts = self.config['clip_invalid_counts']

        self.final_char = self.tokenization.ctoy(self.tokenization.FINAL_CHAR)
        self.initial_char = self.tokenization.ctoy(self.tokenization.INITIAL_CHAR)
        self.pad_char = self.tokenization.ctoy(self.tokenization.PAD_CHAR)

        self.counter_matrix = self.construct_counter_matrix()
        self.encoder = encoder
        self.steps = steps
        self.n = n
        self.decoder = decoder
        self.k = k
        self.kk = kk
        self.y_tokens = torch.tensor(self.tokenization.y_tokens)
        self.y_init, self.scores_init, self.pad_mask = self._init_templates()
        self.eps = 0.001
        self.temperature = temperature
        self.clip_invalid_counts_factor = 1. * self.clip_invalid_counts
        self.sequence_length_correction = self.config['sequence_length_correction']

    def construct_counter_matrix(self):
        m11 = self.tokenization.ELEMENT_MAP
        m13 = self.tokenization.GRAMMAR_MAP
        m12 = torch.zeros_like(m13)
        m_up = torch.cat([m11, m12, m13], dim=1)
        m21 = torch.zeros_like(m11[:1, :])
        m22 = torch.ones_like(m12[:1, :])
        m23 = torch.zeros_like(m13[:1, :])
        m_down = torch.cat([m21, m22, m23], dim=1)
        return torch.cat([m_up, m_down], dim=0)

    def softmax(self, logits):
        return F.softmax(logits / self.temperature, dim=-1)

    def _init_templates(self):
        y_init = np.ones((self.n, self.k), dtype=np.int32) * self.pad_char  # (8, 128)
        y_init[:, self.k - 1] = self.initial_char
        y_init = torch.tensor(np.reshape(y_init, (-1,)))  # (1024=8*128,)

        scores_init = np.full((self.n, self.k), -np.inf, dtype=np.float32)  # (8, 128)
        scores_init[:, self.k - 1] = 0
        scores_init = torch.tensor(np.reshape(scores_init, (-1,)))  # (1024=8*128,)

        pad_mask = np.zeros((1, 1, self.y_tokens), dtype=np.float32)  # (1,1,39)
        pad_mask[0, 0, self.pad_char] = -np.inf
        pad_mask = torch.tensor(np.reshape(pad_mask, (1, 1, -1)))
        return y_init, scores_init, pad_mask  # # (1024=8*128,)  (1024=8*128,)  (1,1,39)

    def beam_step(self, y, states, scores, counts):
        # y: (1024=8*128,1,39), states,
        # scores: (1024,), counts (1024, 11)
        batch_size = self.n
        beam_width = self.k
        nk = batch_size * beam_width

        counts_min = torch.unsqueeze(  # (1024=8*128,1,1)
            torch.unsqueeze(
                torch.where(
                    torch.any(counts < -self.eps, dim=1),
                    -np.inf * torch.ones(nk),  # counts中存在负数，-inf 无效值
                    torch.zeros(nk)  # 0 有效值
                ),
                1),
            2)
        # if not self.clip_invalid_counts:  # False
        #     counts_min = torch.zeros_like(counts_min)  # 这里没有选择删掉无效值

        # y: (1024,1,39)   scores: (1024,)
        # scores_y: (1024=8*128,1,39)    y: (1024,1,39)
        scores_y = (torch.unsqueeze(torch.reshape(scores, y.shape[:-1]), 2) +  # (1024,1,1) 上一时刻的累计分数
                    torch.log(y) +  # (1024=8*128,1,39) # 当前符号的分数
                    # self.pad_mask)              # (1,1,39)          # 把&符号的位置设为-inf
                    self.pad_mask +
                    + counts_min)  # (1024,1,1)  # 如果counts中有负数，设置为-inf
        scores_y = torch.reshape(scores_y, [self.n, -1])  # (8, 128*1*39)
        # ============>在这里添加额外的过滤筛选条件
        # 1、counts - y  < 0   or  count中"(,)"的值不能超过1

        # 2、y = '='时, seq的最后一个符号是'=0'或'#0'

        # 3、出现结束符号*计算剩余元素数目, (), 环

        # 4、出现了分子式中不存在的元素, counts < 0

        # 为每个序列选择前k个得分
        # 128个序列，每个序列有39种选择，从128*39种选择中选出得分最高的128个
        top_k = torch.topk(scores_y, self.k, dim=-1, largest=True, sorted=False)  # (8, 128)
        top_k_index = torch.reshape(  # (1024=8*128,)
            top_k.indices +  # (8,128) top_k.indices的索引为scores_y的行内索引
            scores_y.shape[1] * torch.reshape(torch.arange(self.n), (-1, 1)),  # (8,1)
            [-1])

        ysequence = top_k_index // y.shape[2]  # (1024=8*128)   父序列索引 ∈ [0, beam_width]
        ymax = top_k_index % y.shape[2]  # (1024=8*128)   单词索引 ∈ [0, voc_size]

        states = states[ysequence, :]  # (1024=8*128, 1931) 选出子序列对应的父序列特征
        scores = torch.reshape(top_k.values, [-1])  # (1024=8*128,)  # 子序列的得分
        return ymax, ysequence, states, scores

    def mod_beam_step(self, y, states, scores, counts):
        # y: (1024=8*128,1,39), states,
        # scores: (1024,), counts (1024, 11)
        batch_size = self.n
        beam_width = self.k
        nk = batch_size * beam_width

        counts_min = torch.unsqueeze(  # (1024=8*128,1,1)
            torch.unsqueeze(
                torch.where(
                    torch.any(counts < -self.eps, dim=1),
                    -np.inf * torch.ones(nk),  # counts中存在负数，-inf 无效值
                    torch.zeros(nk)  # 0 有效值
                ),
                1),
            2)
        # counts_min = self.clip_invalid_counts * counts_min  # 这里没有选择删掉无效值

        # 计算所有候选序列的所有预测的得分
        # 从父序列得分和新的预测得分。
        # 将形状(n*k, 1, tokens)数组展平为(n, k x tokens)数组
        # y: (1024,1,39)   scores: (1024,)
        # scores_y: (1024=8*128,1,39)    y: (1024,1,39)
        scores_y = (torch.unsqueeze(torch.reshape(scores, y.shape[:-1]), 2) +  # (1024,1,1) 上一时刻的累计分数
                    torch.log(y) +  # (1024=8*128,1,39) # 当前符号的分数
                    self.pad_mask)  # (1,1,39)  # 把&符号的位置设为-inf
        # self.pad_mask +     # (1,1,39)
        # + counts_min)       # (1024,1,1)  # 如果counts中有负数，设置为-inf
        scores_y = torch.reshape(scores_y, [self.n, -1])  # (8, 128*1*39)
        # ============>在这里添加额外的过滤筛选条件
        # 1、counts - y  < 0   or  count中"(,)"的值不能超过1
        y_onehot = self.tokenization.embed_ytox(y)
        seqs_ = np.array(self.tokenization.VOC)[y_onehot]
        h_count = np.vectorize(lambda x: int(x[-1]))(seqs_)
        h_count = torch.from_numpy(h_count).unsqueeze(-1).cuda()  # (batch_size, 1, 1)
        y_H_ = torch.cat([y_onehot, h_count], dim=-1)
        cnts = y_H_ * self.counter_matrix
        # if counts-cnts
        # 2、y = '='时, seq的最后一个符号是'=0'或'#0'

        # 3、出现结束符号*计算剩余元素数目, (), 环

        # 4、出现了分子式中不存在的元素, counts < 0

        # 为每个序列选择前k个得分
        # 128个序列，每个序列有39种选择，从128*39种选择中选出得分最高的128个
        top_k = torch.topk(scores_y, self.k, dim=-1, largest=True, sorted=False)  # (8, 128)
        top_k_index = torch.reshape(  # (1024=8*128,)
            top_k.indices +  # (8,128) top_k.indices的索引为scores_y的行内索引
            scores_y.shape[1] * torch.reshape(torch.arange(self.n), (-1, 1)),  # (8,1)
            [-1])
        # 使用内置的unravel_index，我们将其转换为
        # [parent_sequence, y_prediction]在y数组中，
        # 其形状为(n*k, 1, tokens)。
        ysequence = top_k_index // y.shape[2]  # (1024=8*128)   父序列索引 ∈ [0, beam_width]
        ymax = top_k_index % y.shape[2]  # (1024=8*128)   单词索引 ∈ [0, voc_size]

        states = states[ysequence, :]  # (1024=8*128, 1931) 选出子序列对应的父序列特征
        scores = torch.reshape(top_k.values, [-1])  # (1024=8*128,)  # 子序列的得分
        return ymax, ysequence, states, scores

    def decode_beam(self, states_init):
        # self.y_init: (1024=8*128,) --> xstep: (1024=8*128,1,39)
        xstep = self.tokenization.embed_ytox(self.y_init)
        scores = self.scores_init  # (1024=8*128,)
        seqs = np.array([self.tokenization.INITIAL_CHAR] * self.n * self.k).reshape(-1, 1)
        voc_ = np.array(self.tokenization.VOC)
        states = states_init['states']  # (1024=8*128, 1931)

        # 记录提前结束的seqs和scores
        manager = SequenceManager(self.n, self.k, self.steps, self.tokenization.FINAL_CHAR, self.tokenization.PAD_CHAR)
        i = 0
        while i < self.steps:
            decoder_out = self.decoder({'tokens_X': xstep.cuda(), 'states': states.cuda()})
            decoder_out = {k: v.cpu() for k, v in decoder_out.items()}

            y = decoder_out['tokens_X']  # (1024=8*128,1,39)
            states = decoder_out['states']  # (1024=8*128,1931)
            counts = decoder_out['counts']  # (1024=8*128,11)

            y = self.softmax(y)
            # 这里beam_step的参数要添加候选序列
            ymax, ysequence, states, scores = self.beam_step(y, states, scores, counts)
            seqs = seqs[ysequence]  # (1024,)
            yword = voc_[ymax].reshape(-1, 1)  # (1024,)
            seqs = np.hstack((seqs, yword))

            # 把完成的序列填入矩阵中
            complete_inds = np.where(yword == self.tokenization.FINAL_CHAR)[0]
            for ind in complete_inds:
                manager.add_sequence(seqs[ind], scores[ind], ind)

            xstep = self.tokenization.embed_ytox(ymax)
            i += 1
        # tokens --> smiles
        smiles, scores = manager.sequences2smiles()
        return smiles, scores

    def beam_traceback(self, sequences, y, scores, reverse=True):
        # sequences:(127,1024=8*128), y:(127,1024=8*128), scores:(127,1024=8*128)
        # (127, 8*128) --> (8, 128*127)
        r_t_r = lambda x: x.reshape(-1, self.n, self.k).permute(1, 2, 0).reshape(self.n, -1)
        y_ = r_t_r(y)  # (8,128*127)
        sc_ = r_t_r(scores)  # (8,128*127)

        # (8,128*127)  标记final_char的位置,# 不是final_char标记为-inf
        ends = torch.where(y_ == self.final_char, sc_, torch.full_like(sc_, -np.inf))
        top_kk = torch.topk(ends, self.kk, dim=1)  # (8,128)
        scores = top_kk.values  # (8,128) 8个小分子的128候选序列的得分
        # (2, 8*128)
        top_kk_vec_ = np.unravel_index(top_kk.indices.view(-1), (self.k, self.steps))
        top_kk_vec = tuple([torch.from_numpy(i) for i in top_kk_vec_])

        i_source = torch.arange(self.n * self.kk) // self.kk  # (1024,)  这里的代码默认 k==kk
        pos = top_kk_vec[0] + self.k * i_source  # (1024,)
        step = top_kk_vec[1]  #
        length = step

        max_length = step.max()
        traces = torch.zeros((max_length, self.n * self.kk), dtype=torch.int32)

        i = 0  # 这里有bug, 最长的序列会缺失t0时刻预测的值
        while (step > 0).any():
            token_at_pos = y[step, pos]
            continue_at = sequences[step, pos]
            pos = continue_at
            traces[i] = token_at_pos
            i += 1
            step = torch.clamp(step - 1, min=0)

        #
        traces = traces.t()
        # traces = tf.sequence_mask(length + 1, max_length, dtype=tf.int32) * traces
        traces = self.sequence_mask(length + 1, max_length, dtype=torch.bool) * traces

        length_with_termination_capped = torch.min(max_length, length + 1)

        if not reverse:
            return traces, scores, length + 1

        traces = self.reverse_sequence(traces, length_with_termination_capped)

        # traces = torch.flip(traces, dims=[1])
        return traces, scores, length + 1

    def sequence_mask(self, lengths, max_len, dtype=torch.bool):
        row_vector = torch.arange(0, max_len, 1)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix

        return mask.to(dtype)

    def reverse_sequence(self, traces, length_with_termination_capped):
        reversed_traces = torch.zeros_like(traces)
        for i, length in enumerate(length_with_termination_capped):
            reversed_traces[i, :length] = torch.flip(traces[i, :length], [0])

        # 保持未反转的部分与原始数据相同
        for i, length in enumerate(length_with_termination_capped):
            reversed_traces[i, length:] = traces[i, length:]
        return reversed_traces

    def sequence_ytoc(self, seq):
        return self.tokenization.sequence_ytoc(seq)

    def format_results(self, smiles, scores, **kwargs):
        seq_df = pd.DataFrame({
            "smiles": smiles,
            "score": scores.reshape(-1),
            "id": range(len(smiles))
        })
        seq_df["n"] = seq_df["id"] // self.kk
        seq_df["k"] = seq_df["id"] % self.kk
        for k, v in kwargs.items():
            seq_df[k] = v
        return seq_df

    def format_reference(self, smiles, fingerprint):
        seq_df = pd.DataFrame({
            "smiles": smiles,
            "score": np.inf,
            "id": range(len(smiles)),
            "n": range(len(smiles)),
            "k": -1,
            "fingerprint": fingerprint
        })
        return seq_df

    def score_step(self, y_pred, y_sequence, counts):
        counts_min = torch.where(
            counts.min(dim=1)[0] < -self.eps,
            torch.full_like(counts[:, :, 0], -np.inf),
            torch.zeros_like(counts[:, :, 0])
        ).unsqueeze(1).unsqueeze(2)
        counts_min = self.clip_invalid_counts * counts_min

        scores_y = (torch.log(y_pred) + counts_min).squeeze()
        scores_y = (y_pred * y_sequence).max(dim=2)[0]
        ymax = y_sequence.argmax(dim=2).view(-1)
        scores = torch.log(scores_y)
        return ymax, scores

    def score_sequences(self, states_init, y_sequences):
        y_init_ = torch.ones(states_init['states'].shape[0], dtype=torch.int32) * self.initial_char
        xstep = self.tokenization.embed_ytox(y_init_)
        scores = torch.zeros(states_init['states'].shape[0])
        steps = y_sequences.shape[1]
        states = states_init['states']
        scores_chain = []

        for i in range(steps):
            decoder_out = self.decoder({'tokens_X': xstep, 'states': states})
            y = decoder_out['tokens_X']
            states = decoder_out['states']
            counts = decoder_out['counts']

            y = self.softmax(y)
            y_seq_step = y_sequences[:, i, :].unsqueeze(1)
            ymax, scores = self.score_step(y, y_seq_step, counts)
            xstep = self.tokenization.embed_ytox(ymax)
            scores_chain.append(scores)

        scores_final = torch.stack(scores_chain).t()
        return scores_final

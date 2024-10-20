import torch
import numpy as np
import pandas as pd
import importlib

from .decoder_base import DecoderBase


class BeamSearchDecoder(DecoderBase):
    def _init_templates(self):
        """
        生成在解码步骤0中使用的常量输入。

        注意：对于每个n个序列，只有一个输入是有效的，
        并且以y = initial_char, score = 0开始，
        其他输入以score = -Inf开始！“有效”序列被放在k-1而不是0，
        以便于调试关于重塑，整数除法和模运算的操作。

        返回
        -------
        y_init : numpy.array (n*k,)
            一个展平的数组，其中一个序列的预测值是initial_char，
            被输入嵌入矩阵生成第一个x输入。
        scores_init : np.array(n*k,)
            一个展平的数组，得分为零（对于起始序列）或-Inf（对于所有空位置）
        pad_mask : np.array(n*k, 1, y_tokens)
            一个形状为模型y输出的数组，
            为每个pad_char的y结果添加-Inf的得分。

        """
        y_init = np.ones((self.n, self.k), dtype=np.int32) * self.pad_char  # (8, 128)
        y_init[:, self.k - 1] = self.initial_char
        y_init = torch.tensor(np.reshape(y_init, (-1,)))  # (1024=8*128,)
        # 所有无效输入以-无穷大的得分开始，因此它们只有在没有足够有效可能性的情况下才会被继续
        # （即在k > tokens的第一步时）
        scores_init = np.full((self.n, self.k), -np.inf, dtype=np.float32)  # (8, 128)
        scores_init[:, self.k - 1] = 0
        scores_init = torch.tensor(np.reshape(scores_init, (-1,)))  # (1024=8*128,)
        # 定义一个填充掩码，将PAD_CHAR的位置分数降到-无穷大，
        # 这样这些序列不会抑制接近的竞争者。这将导致生成许多额外的长废话，
        # 但我们不在乎，因为我们只回溯一些好的序列。
        pad_mask = np.zeros((1, 1, self.y_tokens), dtype=np.float32)  # (1,1,39)
        pad_mask[0, 0, self.pad_char] = -np.inf
        pad_mask = torch.tensor(np.reshape(pad_mask, (1, 1, -1)))
        return y_init, scores_init, pad_mask  # # (1024=8*128,)  (1024=8*128,)  (1,1,39)

    def beam_step(self, y, states, scores, counts):
        """
        执行一个解码步骤的beam search优先级排序。
        输入k（beam宽度）状态和第n步预测的结果，以及k个"父"序列的累积、未惩罚的得分。
        返回k个下一个查询序列和状态。

        参数
        ----------
        y : Tensor k x 1 x y_tokens
            第n步预测的k个候选beam的预测结果
        states : Tensor k x tensor_size
            第n步预测后k个候选beam的预测状态
        scores : Tensor k x 1
            第n-1步时k个候选beam的累积得分，进入第n步预测
        counts: Tensor k x 1 x (element_tokens + grammar_tokens)
            剩余元素计数。如果任何计数<0，则杀死该序列。

        返回
        -------
        Tuple (ymax, ysequence, states, scores)
        ymax : Tensor k x 1
            第n步优先级排序后选择的k个候选beam的字符
        ysequence : Tensor k x 1
            第n步优先级排序后选择的k个候选beam的父序列；
            将其转换（嵌入）为第n+1步预测的输入字符
        states : Tensor k x 1
            第n步优先级排序后选择的k个候选beam的状态
            （第n+1步预测的输入状态）
        scores : Tensor k x 1
            第n步优先级排序后的累积得分
            （第n+1步优先级排序的输入得分）
        """
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
                    # self.pad_mask
                    # + counts_min)  # (1024,1,1)  # 如果counts中有负数，设置为-inf
        scores_y = torch.reshape(scores_y, [self.n, -1])  # (8, 128*1*39)
        # 为每个序列选择前k个得分
        # 128个序列，每个序列有39种选择，从128*39种选择中选出得分最高的128个
        top_k = torch.topk(scores_y, self.k, dim=-1, largest=True, sorted=False)  # (8, 128)
        # 重新计算top_k的数组索引。
        # torch.topk返回一个形状(n, k)的数组，其范围达到'k*tokens'。
        # * 注意！第一个“k”来自torch.topk中的“k”参数。
        # * 第二个k来自scores_y中每n个候选的k个输入序列。
        # 在原始的y数组中，有n*k个序列，每个序列有'tokens'个预测候选。
        # 在展平的数组中，对于第i个序列，索引从i * (k * tokens)开始，
        # (k*tokens)恰好已经是张量轴1的长度。
        top_k_index = torch.reshape(  # (1024=8*128,)
            top_k.indices +  # (8,128) top_k.indices的索引为scores_y的行内索引
            scores_y.shape[1] * torch.reshape(torch.arange(self.n), (-1, 1)),  # (8,1)
            [-1])
        # 使用内置的unravel_index，我们将其转换为
        # [parent_sequence, y_prediction]在y数组中，
        # 其形状为(n*k, 1, tokens)。
        ysequence = top_k_index // y.shape[2]  # (1024=8*128)   父序列索引 ∈ [0, beam_width]
        ymax = top_k_index % y.shape[2]  # (1024=8*128)   单词索引 ∈ [0, voc_size]

        # 收集下一个预测的前序状态和得分，
        # 它们是优先排序后的父序列的状态和得分
        # （以及方便地由top_k直接返回的得分）
        states = states[ysequence, :]  # (1024=8*128, 1931) 选出子序列对应的父序列特征
        scores = torch.reshape(top_k.values, [-1])  # (1024=8*128,)  # 子序列的得分
        return ymax, ysequence, states, scores

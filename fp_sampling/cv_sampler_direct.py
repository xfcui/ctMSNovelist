# # -*- coding: utf-8 -*-
#
# import numpy as np
# import tensorflow as tf
#
# import warnings
#
# from myMSNovelist.fp_sampling.sampling import Sampler
#
#
# class CVSampler(Sampler):
#     def __init__(self, fp_true, fp_predicted, sampler_config={},
#                  generator=tf.random.experimental.get_global_generator()):
#         """
#         Initializes a random probabilistic sampler, i.e. which samples
#         simulated probabilistic fingerprints (with some added noise) for a true
#         fingerprint, randomly - i.e. picking a random prediction for every
#         bit of corresponding value.
#
#         Parameters
#         ----------
#         fp_true: np.array (n, fp_len)
#             Array of n true fingerprints, expected to be € {0, 1}
#         fp_predicted:
#             Array of n CSI:FingerID crossvalidated predictions,
#             expected to be € [0, 1] (or really, any value, this does not matter)
#         noise:
#             A noise scaling factor for adding uniform noise to the fingerprint.
#         generator: tf.random.Generator
#         Returns
#         -------
#         None.
#         """
#         Sampler.__init__(self)
#
#         # Sampler configuration. Default plus settings
#         sampler_config_ = {
#             'use_similar': False,
#             'n_top': 0,
#             'max_loop': 10,
#             'mask_random': 0,
#             'replace_random': 0,
#             'replace_true': 0,
#             'fill_remaining': True,
#             'final_jitter': 0.1,
#             'final_round': True,
#         }
#         sampler_config_.update(sampler_config)
#         self.use_similar = tf.convert_to_tensor(sampler_config_['use_similar'], "bool")
#         self.n_top = tf.convert_to_tensor(sampler_config_['n_top'], "int32")
#         self.max_loop = tf.convert_to_tensor(sampler_config_['max_loop'], "int32")
#         self.mask_random = tf.convert_to_tensor(sampler_config_['mask_random'], "float32")
#         self.replace_random = tf.convert_to_tensor(sampler_config_['replace_random'], "float32")
#         self.replace_true = tf.convert_to_tensor(sampler_config_['replace_true'], "float32")
#         self.fill_remaining = tf.convert_to_tensor(sampler_config_['fill_remaining'], "bool")
#         self.final_jitter = tf.convert_to_tensor(sampler_config_['final_jitter'], "float32")
#         self.final_round = tf.convert_to_tensor(sampler_config_['final_round'], "bool")
#
#         if self.use_similar:
#             warnings.warn("use_similar is currently deactivated to check if optimization is influenced")
#
#         # Append one completely TP, FP, TN, FN fingerprint
#         # to have at least one sample per bit and type
#         # at least one value per bit
#         fp_true = np.concatenate([
#             fp_true,
#             np.zeros_like(fp_true[:1, :]),
#             np.ones_like(fp_true[:1, :]),
#             np.zeros_like(fp_true[:1, :]),
#             np.ones_like(fp_true[:1, :]),
#         ], axis=0)
#
#         # Note: This throws out quite a few fingerprint bits.
#         # 95 bits have zero positive demos in this dataset!
#         # Perhaps a better idea would be to use the stats as a fallback.
#
#         # At this value, add a complete misprediction so we don't mistakenly
#         # teach the network that this prediction is "good" - if we set 0.5 here,
#         # this would mean that if all fp_predicted < 0.1 for fp_true = 0,
#         # fp_predicted = 0.5 uniquely identifies fp_true = 1.
#         # Anecdotally, for the 95 bits, 100 sampled bits0 were was below 0.1!
#         fp_predicted = np.concatenate([
#             fp_predicted,
#             np.ones_like(fp_true[:1, :]),
#             np.ones_like(fp_true[:1, :]),
#             np.zeros_like(fp_true[:1, :]),
#             np.zeros_like(fp_true[:1, :])
#         ], axis=0)
#
#         self.fp_true = tf.convert_to_tensor(fp_true, "float32")
#         self.fp_predicted = tf.convert_to_tensor(fp_predicted, "float32")
#         self.generator = generator
#
#         # Initialize the blocks with indexers for where 1 and 0 bits are
#         fp_TP_where = tf.transpose((self.fp_true == 1) & (self.fp_predicted > 0.5))
#         fp_FP_where = tf.transpose((self.fp_true == 0) & (self.fp_predicted > 0.5))
#         fp_FN_where = tf.transpose((self.fp_true == 1) & (self.fp_predicted < 0.5))
#         fp_TN_where = tf.transpose((self.fp_true == 0) & (self.fp_predicted < 0.5))
#
#         self.fp_where = tf.cast(
#             tf.stack([fp_TP_where, fp_FP_where, fp_FN_where, fp_TN_where]), "float32")
#         self.fp_positions = tf.where(self.fp_where)
#         self.fp_sums = tf.cast(tf.reduce_sum(self.fp_where, axis=2), "int32")
#         fp_cumsums = tf.cumsum(
#             tf.reshape(self.fp_sums, [-1]), exclusive=True)
#         self.fp_cumsums = tf.reshape(fp_cumsums, [4, -1])
#         self.fp_values = tf.gather_nd(
#             tf.transpose(self.fp_predicted),
#             self.fp_positions[:, 1:]
#         )
#
#         self.fp_where_uncorrelated = tf.cast(
#             tf.stack([fp_TP_where | fp_FN_where,
#                       fp_TN_where | fp_FP_where]), "float32")
#         self.fp_positions_uncorrelated = tf.where(self.fp_where_uncorrelated)
#         self.fp_sums_uncorrelated = tf.cast(tf.reduce_sum(self.fp_where_uncorrelated, axis=2), "int32")
#         fp_cumsums_uncorrelated = tf.cumsum(
#             tf.reshape(self.fp_sums_uncorrelated, [-1]), exclusive=True)
#         self.fp_cumsums_uncorrelated = tf.reshape(fp_cumsums_uncorrelated, [2, -1])
#         self.fp_values_uncorrelated = tf.gather_nd(
#             tf.transpose(self.fp_predicted),
#             self.fp_positions_uncorrelated[:, 1:]
#         )
#
#     # @tf.function
#     def partial_tanimoto_ref(self, fp, positions_):
#         """
#         Calculates the tanimoto similarity between n fingerprints in fp
#         and m reference fingerprints in self.fp_true, for all positions
#         specified in positions.
#
#         Correspondingly, returns a tensor nxm of float 0..1.
#
#         """
#         # Some code from:
#         # https://github.com/keras-team/keras/issues/9395#issuecomment-379228094
#         # Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
#         # -> the score is computed for each class separately and then summed
#         # alpha=beta=0.5 : dice coefficient
#         # alpha=beta=1   : tanimoto coefficient (also known as jaccard)
#         # alpha+beta=1   : produces set of F*-scores
#         # implemented by E. Moebel, 06/04/18
#
#         alpha = 1
#         beta = 1
#
#         fp_subset = tf.cast(fp, 'uint8')
#         ref_subset = tf.cast(self.fp_true, 'uint8')
#         positions = tf.cast(positions_, 'uint8')
#         ones_fp = tf.ones_like(fp_subset)
#         ones_ref = tf.ones_like(ref_subset)
#
#         fp_1 = tf.expand_dims(positions * fp_subset, 1)
#         fp_0 = tf.expand_dims(positions * ones_fp - fp_subset, 1)
#         ref_1 = tf.expand_dims(ref_subset, 0)
#         ref_0 = tf.expand_dims(ones_ref - ref_subset, 0)
#
#         # Tanimoto is C11 / (C10+C01+C11) but only unfilled positions are counted
#         c11 = tf.reduce_sum(fp_1 * ref_1, axis=2)
#         c10 = tf.reduce_sum(fp_1 * ref_0, axis=2)
#         c01 = tf.reduce_sum(fp_0 * ref_1, axis=2)
#         tanimoto = c11 / (c11 + alpha * c10 + beta * c01)
#         return tf.cast(tanimoto, "float32")
#
#     @tf.function
#     def sample(self, fp):
#         fp_simulated, _ = self.sample_(fp)
#         return fp_simulated
#
#     @tf.function
#     def sample_with_missing(self, fp):
#         fp_simulated, missing = self.sample_(fp)
#         return fp_simulated, missing
#
#     @tf.function
#     def pick_source(self, fp):
#         fp_selected = self.generator.uniform_full_int((tf.shape(fp)[0],
#                                                        self.max_loop),
#                                                       dtype='int32') % tf.shape(self.fp_true)[0]
#         fp_sample = tf.gather(self.fp_true, fp_selected)
#         fp_value_to_sample = tf.cast(tf.gather(
#             self.fp_predicted, fp_selected) > 0.5,
#                                      'float32')
#         return fp_sample, fp_value_to_sample
#
#     @tf.function
#     def sample_steps(self, fp, fp_simulated, fp_positions_empty):
#         # Two methods:
#         # Either sample from very similar fingerprints
#         # Or sample from totally random fingerprints
#
#         fp_sample, fp_value_to_sample = self.pick_source(fp)
#         fp_updim = tf.expand_dims(fp, 1)
#         # find the positions to sample:
#         # 1.
#         # Starting from "all unsampled positions", put a random mask to limit
#         # sampling more than "random_mask %" from a single fingerprint
#         # (This is not in the CANOPUS version)
#         fp_positions_sampling = tf.cast(
#             self.generator.uniform(tf.shape(fp_sample), dtype="float32") > self.mask_random,
#             "float32")
#
#         # 2.
#         # For the non-random-masked unsampled positions, find bits that match
#         # between the query and the chosen library fingerprints.
#         # For every 1-1 match,
#         # sample a true positive if the simulated library fingerprint is a true positive;
#         # and a false negative if the simulated library fingerprint is a false negative.
#         c11 = tf.cast(fp_updim * fp_sample, "float32") * fp_positions_sampling
#         sample_tp = c11 * fp_value_to_sample
#         sample_fn = c11 * (1 - fp_value_to_sample)
#         c00 = tf.cast((1 - fp_updim) * (1 - fp_sample), "float32") * fp_positions_sampling
#         sample_tn = c00 * (1 - fp_value_to_sample)
#         sample_fp = c00 * (fp_value_to_sample)
#
#         # 3.
#         # This is perhaps the hardest to understand from the code:
#         # Choose the position within the array of TP,FP,TN,FN values
#         # Let TP=0,FP=1,FN=2,TN=3 be called the four "strata" to choose from
#         # for each bit.
#         # fp_sums is the number of entries for each bit and stratum, and
#         # fp_cumsums is the starting point for each bit and stratum.
#         # So chosen_position will be a n_query * 4 (strata) * n_bits tensor
#         # of positions in an 1-d array (in which the values for each
#         # bit and stratum are stored)
#         # Note:
#         # We only need to sample one value per bit and stratum, because only
#         # one is going to be needed in any case
#         rand = self.generator.uniform_full_int(tf.shape(fp_updim), dtype="int32")
#         rand = tf.expand_dims(rand, 1)
#         chosen_position = (rand % tf.expand_dims(self.fp_sums, 0)
#                            + tf.expand_dims(self.fp_cumsums, 0))
#
#         # 4.
#         # Extract the value from the 1-d array (this is the easy part :) )
#         sampled_bit = tf.gather(self.fp_values, chosen_position)
#         # 5.
#         # We now have a value to sample for every bit AND stratum of the
#         # query fingerprints. By stacking together the "stratum choice" and
#         # multiplying, then summing the stratum values for each bit,
#         # we find which value to actually add to the fingerprint.
#         # Reminder: Let TP=0,FP=1,FN=2,TN=3
#         chosen_stratum = tf.stack([sample_tp, sample_fp, sample_fn, sample_tn], axis=2)
#         chosen_bit = tf.reduce_sum(
#             sampled_bit * tf.cast(chosen_stratum, "float32"), axis=2)
#         # 6.
#         # The c11+c00 determines where there are bits.
#         # Now keep only the first occurrence of value 1 on axis 1,
#         # since every position needs to be sampled only once.
#         # I.e. find the position where the cumsum == 1 && sampled_positions == 1
#         sampled_positions = c11 + c00
#         sampled_positions_cum = tf.cumsum(sampled_positions, axis=1)
#         sampled_positions_chosen = tf.cast(sampled_positions_cum == 1, 'float32') * sampled_positions
#
#         fp_simulated = (fp_simulated +
#                         tf.reduce_sum(chosen_bit * sampled_positions_chosen, axis=1))
#
#         fp_positions_empty = (fp_positions_empty -
#                               tf.reduce_sum(sampled_positions_chosen, axis=1))
#
#         return fp_simulated, fp_positions_empty
#
#     @tf.function
#     def sample_remaining(self, fp, fp_simulated, fp_positions_empty, replace=0):
#         # Fill the remaining positions with random, uncorrelated sampling
#         # from TP + FN or TN + FP
#
#         # Additionally, replace some of the correlatedly sampled positions
#         # with random samples
#         replace_positions = self.generator.uniform(tf.shape(fp), dtype="float32")
#         replace_positions = tf.cast(replace_positions < replace, "float32")
#         replace_positions = replace_positions * (1 - fp_positions_empty)
#
#         fill_positions = (fp_positions_empty + replace_positions)
#
#         c1 = fp * fill_positions
#         c0 = (1 - fp) * fill_positions
#
#         rand = self.generator.uniform_full_int(tf.shape(fp), dtype="int32")
#         rand = tf.expand_dims(rand, 1)
#         # Position chosen for each of c1, c0:
#
#         chosen_position = (rand % tf.expand_dims(self.fp_sums_uncorrelated, 0)
#                            + tf.expand_dims(self.fp_cumsums_uncorrelated, 0))
#         sampled_bit = tf.gather(self.fp_values_uncorrelated, chosen_position)
#         # Here:
#         # TP+FN = c1
#         # TN+FP = c0
#         chosen_stratum = tf.stack([c1, c0], axis=1)
#
#         chosen_bit = tf.reduce_sum(
#             sampled_bit * tf.cast(chosen_stratum, "float32"), axis=1)
#
#         # Compose the final fingerprint from the previous simulated FP
#         # and the newly sampled positions
#         fp_simulated = (fp_simulated * (1 - fill_positions)) + (chosen_bit * fill_positions)
#
#         return fp_simulated
#
#     @tf.function
#     def sample_true(self, fp, fp_simulated, replace=0):
#         """
#         Replace a subset of the fingerprint with only correct predictions
#         """
#         replace_score = tf.expand_dims(
#             self.generator.uniform((tf.shape(fp)[0],)),
#             1
#         ) + 2 * (replace - 0.5)
#
#         replace_positions = self.generator.uniform(tf.shape(fp), dtype="float32")
#         replace_positions = tf.cast(replace_positions < replace_score, "float32")
#         sample_tp = fp * replace_positions
#         sample_tn = (1 - fp) * replace_positions
#         sample_fn = tf.zeros_like(sample_tn)
#         sample_fp = tf.zeros_like(sample_tn)
#
#         rand = self.generator.uniform_full_int(tf.shape(fp), dtype="int32")
#         rand = tf.expand_dims(rand, 1)
#         chosen_position = (rand % tf.expand_dims(self.fp_sums, 0)
#                            + tf.expand_dims(self.fp_cumsums, 0))
#
#         # Extract the value from the 1-d array (this is the easy part :) )
#         sampled_bit = tf.gather(self.fp_values, chosen_position)
#         #
#         # We now have a value to sample for every bit AND stratum of the
#         # query fingerprints. By stacking together the "stratum choice" and
#         # multiplying, then summing the stratum values for each bit,
#         # we find which value to actually add to the fingerprint.
#         # Reminder: Let TP=0,FP=1,FN=2,TN=3
#         chosen_stratum = tf.stack([sample_tp, sample_fp, sample_fn, sample_tn], axis=1)
#         chosen_bit = tf.reduce_sum(
#             sampled_bit * tf.cast(chosen_stratum, "float32"), axis=1)
#         # 6.
#         # Add the sampled bits to the simulated fingerprint
#         # (Note: the (c11+c00) should be unnecessary, since this is already
#         # calculated out in step 2)
#         fp_simulated = ((fp_simulated * (1 - replace_positions)) +
#                         (chosen_bit * replace_positions))
#         # 7.
#         # Remove the sampled positions from the tensor of unsampled positions
#         return fp_simulated
#
#     @tf.function
#     def sample_(self, fp):
#         """
#         Generates simulated predicted fingerprints Y for an array of true
#         fingerprints X [x_j, i] of shape (n, fp_len)
#         (i.e. i is the bit in the fingerprint, j is the fingerprint in the batch)
#
#         using probabilistic correlated sampling from the loaded
#         set of predicted fingerprints.
#
#         Parameters
#         ----------
#         fp : np.array of tf.Tensor(shape=(n, fp_len)) dtype=float32 but
#             Array of n fingerprints expected to be € {0, 1} (but of float type)
#
#         Returns
#         -------
#         Equally-shaped tensor with simulated predicted probabilistic fingerprint, i.e.
#             all values are € (0, 1]
#
#         """
#         replace_random = self.replace_random
#         fill_remaining = self.fill_remaining
#         final_jitter = self.final_jitter
#         final_round = self.final_round
#
#         fp = tf.convert_to_tensor(fp)
#         fp_simulated = tf.zeros_like(fp)
#         fp_positions_empty = tf.ones_like(fp)
#         fp_simulated, fp_positions_empty = self.sample_steps(
#             fp, fp_simulated, fp_positions_empty)
#
#         # Fill all still unfilled positions if fill_remaining is set
#         fp_simulated = tf.cond(
#             fill_remaining,
#             lambda: self.sample_remaining(fp, fp_simulated, fp_positions_empty, replace_random),
#             lambda: fp_simulated
#         )
#
#         # Replace random positions with "true prediction samples" if desired
#         fp_simulated = self.sample_true(fp, fp_simulated, self.replace_true)
#
#         # Add jitter and clip, if desired
#         fp_noise = final_jitter * (self.generator.uniform(tf.shape(fp_simulated)) - 0.5)
#         fp_simulated = tf.clip_by_value(fp_simulated + fp_noise,
#                                         tf.reduce_min(fp_simulated),
#                                         tf.reduce_max(fp_simulated))
#
#         # Round the fingerprint if final_round is set
#         fp_simulated = tf.cond(
#             final_round,
#             lambda: tf.round(fp_simulated),
#             lambda: fp_simulated
#         )
#
#         return fp_simulated, fp_positions_empty
#
#
# # def get_sample(test_df):
# #     # rand_key = random.choice(list(test_dict.keys()))
# #     fp_struct = test_df["fp_true"]  # 结构指纹
# #     fp_spec = test_df["fp"]  # sirius 预测的谱指纹
# #     fp_struct = np.stack(fp_struct.values)
# #     fp_spec = np.stack(fp_spec.values)
# #     sampler = CVSampler(fp_struct, fp_spec, sampler_config={})
# #     return sampler
#
# #
# # def add_error(struct_fp, sampler):
# #     # 使用 spec-fp 向 struct-fp 添加 error, 返回sim-fp
# #     # sim_fp, _ = sampler.gen_sim_fp(struct_fp)
# #     sim_fp, _ = sampler.sample_(struct_fp)
# #     return np.array(sim_fp)
#
#
# def t_CVSampler():
#     # Prepare test data
#     fp_true = np.array([
#         [0, 1, 0, 0, 0],
#         [1, 0, 1, 1, 0]])  # 2 true fingerprints
#
#     fp_predicted = np.array([
#         [0.2, 0.8, 0.3, 0.1, 0.2],
#         [0.6, 0.4, 0.7, 0.8, 0.1]])  # Corresponding predicted fingerprints
#
#     # Create an instance of CVSampler
#     sampler = CVSampler(fp_true, fp_predicted, sampler_config={})
#
#     # Prepare input fingerprints
#     fp_input = np.array([[0, 1, 0, 1, 0],
#                          [1, 0, 1, 1, 1],
#                          [1, 1, 1, 0, 0],
#                          [1, 1, 1, 1, 1]], dtype=np.float32)
#     fp_input = tf.convert_to_tensor(fp_input)
#     # Call the sample method
#     simulated_fp, _ = sampler.sample_(fp_input)
#
#     print("CVSampler sample method test passed.")
#
#
# # if __name__ == "__main__":
# #     # test_df = ""
# #     # sampler = get_sample(test_df)
# #     # sim_fp = add_error(struct_fp=,sampler=)
# #     t_CVSampler()

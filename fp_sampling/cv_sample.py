import torch
import numpy as np
import warnings


# import my_config as mc


# deviceID = mc.config["deviceID"]
# torch.cuda.set_device(f"cuda:{deviceID}")  # 0
# seed = 42
class CVSampler:
    def __init__(self, fp_true, fp_predicted):
        super(CVSampler, self).__init__()

        sampler_config_ = {
            'use_similar': False,
            'n_top': 0,
            'max_loop': 10,
            'mask_random': 0,
            'replace_random': 0,
            'replace_true': 0,
            'fill_remaining': True,
            'final_jitter': 0.1,
            'final_round': False,
        }
        self.use_similar = torch.tensor(sampler_config_['use_similar'], dtype=torch.bool)
        self.n_top = torch.tensor(sampler_config_['n_top'], dtype=torch.int32)
        self.max_loop = torch.tensor(sampler_config_['max_loop'], dtype=torch.int32)
        self.mask_random = torch.tensor(sampler_config_['mask_random'], dtype=torch.float32)
        self.replace_random = torch.tensor(sampler_config_['replace_random'], dtype=torch.float32)
        self.replace_true = torch.tensor(sampler_config_['replace_true'], dtype=torch.float32)
        self.fill_remaining = torch.tensor(sampler_config_['fill_remaining'], dtype=torch.bool).cuda()
        self.final_jitter = torch.tensor(sampler_config_['final_jitter'], dtype=torch.float32)
        self.final_round = torch.tensor(sampler_config_['final_round'], dtype=torch.bool)

        if self.use_similar:
            warnings.warn("use_similar is currently deactivated to check if optimization is influenced")

        fp_true = np.concatenate([
            fp_true,
            np.zeros_like(fp_true[:1, :]),
            np.ones_like(fp_true[:1, :]),
            np.zeros_like(fp_true[:1, :]),
            np.ones_like(fp_true[:1, :]),
        ], axis=0)

        fp_predicted = np.concatenate([
            fp_predicted,
            np.ones_like(fp_true[:1, :]),
            np.ones_like(fp_true[:1, :]),
            np.zeros_like(fp_true[:1, :]),
            np.zeros_like(fp_true[:1, :])
        ], axis=0)

        self.fp_true = torch.from_numpy(fp_true).to(dtype=torch.float32).cuda()
        self.fp_predicted = torch.from_numpy(fp_predicted).to(dtype=torch.float32).cuda()

        fp_TP_where = torch.transpose((self.fp_true == 1) & (self.fp_predicted > 0.5), 0, 1)
        fp_FP_where = torch.transpose((self.fp_true == 0) & (self.fp_predicted > 0.5), 0, 1)
        fp_FN_where = torch.transpose((self.fp_true == 1) & (self.fp_predicted < 0.5), 0, 1)
        fp_TN_where = torch.transpose((self.fp_true == 0) & (self.fp_predicted < 0.5), 0, 1)

        self.fp_where = torch.stack([fp_TP_where, fp_FP_where, fp_FN_where, fp_TN_where], dim=0)
        self.fp_positions = torch.nonzero(self.fp_where)
        self.fp_sums = torch.sum(self.fp_where, dim=2, dtype=torch.int32)

        fp_sums_reshaped = self.fp_sums.reshape(-1)
        fp_cumsums = torch.cumsum(fp_sums_reshaped, dim=0) - fp_sums_reshaped
        self.fp_cumsums = fp_cumsums.reshape(4, -1)

        fp_predicted_transposed = torch.transpose(self.fp_predicted, 0, 1)
        row_indices = self.fp_positions[:, 1:]
        self.fp_values = fp_predicted_transposed[row_indices[:, 0], row_indices[:, 1]]

        # ===
        self.fp_where_uncorrelated = torch.stack([fp_TP_where | fp_FN_where,
                                                  fp_TN_where | fp_FP_where], dim=0)
        self.fp_positions_uncorrelated = torch.nonzero(self.fp_where_uncorrelated)
        self.fp_sums_uncorrelated = torch.sum(self.fp_where_uncorrelated, dim=2, dtype=torch.int32)

        fp_sums_uncorrelated_reshaped = self.fp_sums_uncorrelated.reshape(-1)
        self.fp_cumsums_uncorrelated = torch.cumsum(fp_sums_uncorrelated_reshaped,
                                                    dim=0) - fp_sums_uncorrelated_reshaped
        self.fp_cumsums_uncorrelated = self.fp_cumsums_uncorrelated.reshape(2, -1)

        fp_predicted_transposed = torch.transpose(self.fp_predicted, 0, 1)
        row_indices_uncorrelated = self.fp_positions_uncorrelated[:, 1:]
        self.fp_values_uncorrelated = fp_predicted_transposed[
            row_indices_uncorrelated[:, 0], row_indices_uncorrelated[:, 1]]

    def pick_source(self, fp):
        max_loop = self.max_loop.item()

        fp_selected = torch.randint(-2 ** 31, 2 ** 31 - 1, size=(fp.shape[0], max_loop)) % \
                      self.fp_true.shape[0]
        # fp_selected = fp_selected.long()
        fp_sample = self.fp_true[fp_selected]
        fp_value_to_sample = (self.fp_predicted[fp_selected] > 0.5).float()

        return fp_sample, fp_value_to_sample

    def sample_steps(self, fp, fp_simulated, fp_positions_empty):
        fp_sample, fp_value_to_sample = self.pick_source(fp)
        fp_updim = fp.unsqueeze(1)
        fp_positions_sampling = (torch.rand_like(fp_sample) > self.mask_random).float()
        c11 = (fp_updim * fp_sample).float() * fp_positions_sampling
        sample_tp = c11 * fp_value_to_sample
        sample_fn = c11 * (1 - fp_value_to_sample)
        c00 = ((1 - fp_updim) * (1 - fp_sample)).float() * fp_positions_sampling
        sample_tn = c00 * (1 - fp_value_to_sample)
        sample_fp = c00 * fp_value_to_sample

        rand = torch.randint(-2 ** 31, 2 ** 31 - 1, size=fp_updim.shape).cuda()
        rand = rand.unsqueeze(1)
        chosen_position = (rand % self.fp_sums.unsqueeze(0)
                           + self.fp_cumsums.unsqueeze(0))

        sampled_bit = self.fp_values[chosen_position]
        chosen_stratum = torch.stack([sample_tp, sample_fp, sample_fn, sample_tn], dim=2)
        chosen_bit = (sampled_bit * chosen_stratum.float()).sum(dim=2)

        sampled_positions = c11 + c00
        sampled_positions_cum = sampled_positions.cumsum(dim=1)
        sampled_positions_chosen = sampled_positions * (sampled_positions_cum == 1).float()

        fp_simulated += (chosen_bit * sampled_positions_chosen).sum(dim=1)
        fp_positions_empty -= sampled_positions_chosen.sum(dim=1)

        return fp_simulated, fp_positions_empty

    def sample_remaining(self, fp, fp_simulated, fp_positions_empty, replace=0):
        # Fill the remaining positions with random, uncorrelated sampling
        # from TP + FN or TN + FP

        # Additionally, replace some of the correlatedly sampled positions
        # with random samples
        # torch.manual_seed(seed)
        replace_positions = torch.rand(size=fp.shape, dtype=torch.float32).cuda()
        replace_positions = (replace_positions < replace).float().cuda()
        replace_positions = replace_positions * (1 - fp_positions_empty)

        fill_positions = (fp_positions_empty + replace_positions)

        c1 = fp * fill_positions
        c0 = (1 - fp) * fill_positions

        rand = torch.randint(-2 ** 31, 2 ** 31 - 1, size=fp.shape).cuda()
        rand = rand.unsqueeze(1)
        # Position chosen for each of c1, c0:

        chosen_position = (rand % self.fp_sums_uncorrelated.unsqueeze(0)
                           + self.fp_cumsums_uncorrelated.unsqueeze(0))
        sampled_bit = self.fp_values_uncorrelated[chosen_position]
        # Here:
        # TP+FN = c1
        # TN+FP = c0
        chosen_stratum = torch.stack([c1, c0], dim=1)

        chosen_bit = torch.sum(
            sampled_bit * chosen_stratum.to(torch.float32), dim=1)

        # Compose the final fingerprint from the previous simulated FP
        # and the newly sampled positions
        fp_simulated = (fp_simulated * (1 - fill_positions)) + (chosen_bit * fill_positions)

        return fp_simulated

    def sample_true(self, fp, fp_simulated, replace=0):
        """
        Replace a subset of the fingerprint with only correct predictions
        """
        replace_score = torch.rand(size=(fp.shape[0],)).cuda().unsqueeze(1) + 2 * (replace - 0.5)

        replace_positions = torch.rand(size=fp.shape, dtype=torch.float32).cuda()
        replace_positions = (replace_positions < replace_score).float().cuda()
        sample_tp = fp * replace_positions
        sample_tn = (1 - fp) * replace_positions
        sample_fn = torch.zeros_like(sample_tn).cuda()
        sample_fp = torch.zeros_like(sample_tn).cuda()

        rand = torch.randint(-2 ** 31, 2 ** 31 - 1, size=fp.shape).cuda()
        rand = rand.unsqueeze(1)
        chosen_position = (rand % self.fp_sums.unsqueeze(0)
                           + self.fp_cumsums.unsqueeze(0))

        # Extract the value from the 1-d array (this is the easy part :) )
        sampled_bit = self.fp_values[chosen_position]
        #
        # We now have a value to sample for every bit AND stratum of the
        # query fingerprints. By stacking together the "stratum choice" and
        # multiplying, then summing the stratum values for each bit,
        # we find which value to actually add to the fingerprint.
        # Reminder: Let TP=0,FP=1,FN=2,TN=3
        chosen_stratum = torch.stack([sample_tp, sample_fp, sample_fn, sample_tn], dim=1)
        chosen_bit = torch.sum(
            sampled_bit * chosen_stratum.to(torch.float32), dim=1)
        # 6.
        # Add the sampled bits to the simulated fingerprint
        # (Note: the (c11+c00) should be unnecessary, since this is already
        # calculated out in step 2)
        fp_simulated = ((fp_simulated * (1 - replace_positions)) +
                        (chosen_bit * replace_positions))
        # 7.
        # Remove the sampled positions from the tensor of unsampled positions
        return fp_simulated

    def sample_(self, fp):
        with torch.no_grad():
            replace_random = self.replace_random  # Convert TensorFlow scalar tensor to Python scalar
            fill_remaining = self.fill_remaining  # Convert TensorFlow scalar tensor to Python scalar
            final_jitter = self.final_jitter  # Convert TensorFlow scalar tensor to Python scalar
            final_round = self.final_round  # Convert TensorFlow scalar tensor to Python scalar

            # fp = torch.from_numpy(fp).cuda()
            fp = fp.cuda()
            fp_simulated = torch.zeros_like(fp, dtype=torch.float32).cuda()  # Initialize simulated fingerprints
            fp_positions_empty = torch.ones_like(fp).cuda()  # Initialize positions empty indicator

            # Call sample_steps function
            fp_simulated, fp_positions_empty = self.sample_steps(fp, fp_simulated, fp_positions_empty)

            # Fill all still unfilled positions if fill_remaining is set
            if fill_remaining:
                fp_simulated = self.sample_remaining(fp, fp_simulated, fp_positions_empty, replace_random.item())

            # Replace random positions with "true prediction samples" if desired
            fp_simulated = self.sample_true(fp, fp_simulated, self.replace_true.item())

            # Add jitter and clip, if desired
            # torch.manual_seed(seed)
            fp_noise = final_jitter * (torch.rand_like(fp_simulated) - 0.5)
            fp_simulated = torch.clamp(fp_simulated + fp_noise,
                                       fp_simulated.min(),
                                       fp_simulated.max())

            # Round the fingerprint if final_round is set
            if final_round:
                fp_simulated = torch.round(fp_simulated)

            return fp_simulated.cpu(), fp_positions_empty


# class CVSampler:
#     def __init__(self, fp_true, fp_predicted):
#         super(CVSampler, self).__init__()
#
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
#         self.use_similar = torch.tensor(sampler_config_['use_similar'], dtype=torch.bool)
#         self.n_top = torch.tensor(sampler_config_['n_top'], dtype=torch.int32)
#         self.max_loop = torch.tensor(sampler_config_['max_loop'], dtype=torch.int32)
#         self.mask_random = torch.tensor(sampler_config_['mask_random'], dtype=torch.float32)
#         self.replace_random = torch.tensor(sampler_config_['replace_random'], dtype=torch.float32)
#         self.replace_true = torch.tensor(sampler_config_['replace_true'], dtype=torch.float32)
#         self.fill_remaining = torch.tensor(sampler_config_['fill_remaining'], dtype=torch.bool).cuda()
#         self.final_jitter = torch.tensor(sampler_config_['final_jitter'], dtype=torch.float32)
#         self.final_round = torch.tensor(sampler_config_['final_round'], dtype=torch.bool)
#
#         if self.use_similar:
#             warnings.warn("use_similar is currently deactivated to check if optimization is influenced")
#
#         fp_true = np.concatenate([
#             fp_true,
#             np.zeros_like(fp_true[:1, :]),
#             np.ones_like(fp_true[:1, :]),
#             np.zeros_like(fp_true[:1, :]),
#             np.ones_like(fp_true[:1, :]),
#         ], axis=0)
#
#         fp_predicted = np.concatenate([
#             fp_predicted,
#             np.ones_like(fp_true[:1, :]),
#             np.ones_like(fp_true[:1, :]),
#             np.zeros_like(fp_true[:1, :]),
#             np.zeros_like(fp_true[:1, :])
#         ], axis=0)
#
#         self.fp_true = torch.from_numpy(fp_true).to(dtype=torch.float32).cuda()
#         self.fp_predicted = torch.from_numpy(fp_predicted).to(dtype=torch.float32).cuda()
#
#         fp_TP_where = torch.transpose((self.fp_true == 1) & (self.fp_predicted > 0.5), 0, 1)
#         fp_FP_where = torch.transpose((self.fp_true == 0) & (self.fp_predicted > 0.5), 0, 1)
#         fp_FN_where = torch.transpose((self.fp_true == 1) & (self.fp_predicted < 0.5), 0, 1)
#         fp_TN_where = torch.transpose((self.fp_true == 0) & (self.fp_predicted < 0.5), 0, 1)
#
#         self.fp_where = torch.stack([fp_TP_where, fp_FP_where, fp_FN_where, fp_TN_where], dim=0)
#         self.fp_positions = torch.nonzero(self.fp_where)
#         self.fp_sums = torch.sum(self.fp_where, dim=2, dtype=torch.int32)
#
#         fp_sums_reshaped = self.fp_sums.view(-1)
#         fp_cumsums = torch.cumsum(fp_sums_reshaped, dim=0) - fp_sums_reshaped
#         self.fp_cumsums = fp_cumsums.view(4, -1)
#
#         fp_predicted_transposed = torch.transpose(self.fp_predicted, 0, 1)
#         row_indices = self.fp_positions[:, 1:]
#         self.fp_values = fp_predicted_transposed[row_indices[:, 0], row_indices[:, 1]]
#
#         self.fp_where_uncorrelated = torch.stack([fp_TP_where | fp_FN_where,
#                                                   fp_TN_where | fp_FP_where], dim=0)
#         self.fp_positions_uncorrelated = torch.nonzero(self.fp_where_uncorrelated)
#         self.fp_sums_uncorrelated = torch.sum(self.fp_where_uncorrelated, dim=2, dtype=torch.int32)
#
#         fp_sums_uncorrelated_reshaped = self.fp_sums_uncorrelated.view(-1)
#         self.fp_cumsums_uncorrelated = torch.cumsum(fp_sums_uncorrelated_reshaped,
#                                                     dim=0) - fp_sums_uncorrelated_reshaped
#
#         self.fp_cumsums_uncorrelated = self.fp_cumsums_uncorrelated.view(2, -1)
#
#         fp_predicted_transposed = torch.transpose(self.fp_predicted, 0, 1)
#         row_indices_uncorrelated = self.fp_positions_uncorrelated[:, 1:]
#         self.fp_values_uncorrelated = fp_predicted_transposed[
#             row_indices_uncorrelated[:, 0], row_indices_uncorrelated[:, 1]]
#
#     def pick_source(self, fp):
#         max_loop = self.max_loop.item()  # Convert TensorFlow scalar tensor to Python scalar
#
#         # fp_selected = torch.randint(0, self.fp_true.shape[0], size=(batch_size, max_loop), dtype=torch.int32).cuda()
#         # torch.manual_seed(seed)
#         fp_selected = torch.randint(0, 2 ** 31 - 1, size=(fp.shape[0], max_loop), dtype=torch.int32) % \
#                       self.fp_true.shape[0]
#         fp_selected = fp_selected.long()
#         fp_sample = self.fp_true[fp_selected]
#         fp_value_to_sample = (self.fp_predicted[fp_selected] > 0.5).float()
#
#         return fp_sample, fp_value_to_sample
#
#     def sample_steps(self, fp, fp_simulated, fp_positions_empty):
#         fp_sample, fp_value_to_sample = self.pick_source(fp)
#         fp_updim = fp.unsqueeze(1)
#         # torch.manual_seed(seed)
#         fp_positions_sampling = (torch.rand_like(fp_sample) > self.mask_random).float()
#         c11 = (fp_updim * fp_sample).float() * fp_positions_sampling
#         sample_tp = c11 * fp_value_to_sample
#         sample_fn = c11 * (1 - fp_value_to_sample)
#         c00 = ((1 - fp_updim) * (1 - fp_sample)).float() * fp_positions_sampling
#         sample_tn = c00 * (1 - fp_value_to_sample)
#         sample_fp = c00 * fp_value_to_sample
#
#         # torch.manual_seed(seed)
#         rand = torch.randint(0, 2 ** 31 - 1, size=fp_updim.shape, dtype=torch.int32).cuda()
#         rand = rand.unsqueeze(1)
#         chosen_position = (rand % self.fp_sums.unsqueeze(0) + self.fp_cumsums.unsqueeze(0))
#
#         sampled_bit = self.fp_values[chosen_position]
#         chosen_stratum = torch.stack([sample_tp, sample_fp, sample_fn, sample_tn], dim=2)
#         chosen_bit = (sampled_bit * chosen_stratum.float()).sum(dim=2)
#
#         sampled_positions = c11 + c00
#         sampled_positions_cum = sampled_positions.cumsum(dim=1)
#         sampled_positions_chosen = (sampled_positions_cum == 1) * sampled_positions
#
#         fp_simulated += (chosen_bit * sampled_positions_chosen).sum(dim=1)
#         fp_positions_empty -= sampled_positions_chosen.sum(dim=1)
#
#         return fp_simulated, fp_positions_empty
#
#     def sample_remaining(self, fp, fp_simulated, fp_positions_empty, replace=0):
#         # Fill the remaining positions with random, uncorrelated sampling
#         # from TP + FN or TN + FP
#
#         # Additionally, replace some of the correlatedly sampled positions
#         # with random samples
#         # torch.manual_seed(seed)
#         replace_positions = torch.rand(size=fp.shape, dtype=torch.float32).cuda()
#         replace_positions = torch.as_tensor(replace_positions < replace, dtype=torch.float32).cuda()
#         replace_positions = replace_positions * (1 - fp_positions_empty)
#
#         fill_positions = (fp_positions_empty + replace_positions)
#
#         c1 = fp * fill_positions
#         c0 = (1 - fp) * fill_positions
#
#         # torch.manual_seed(seed)
#         rand = torch.randint(0, 2 ** 31 - 1, size=fp.shape, dtype=torch.int32).cuda()
#         rand = rand.unsqueeze(1)
#         # Position chosen for each of c1, c0:
#
#         chosen_position = (rand % self.fp_sums_uncorrelated.unsqueeze(0)
#                            + self.fp_cumsums_uncorrelated.unsqueeze(0))
#         sampled_bit = self.fp_values_uncorrelated[chosen_position]
#         # Here:
#         # TP+FN = c1
#         # TN+FP = c0
#         chosen_stratum = torch.stack([c1, c0], dim=1)
#
#         chosen_bit = torch.sum(
#             sampled_bit * chosen_stratum.to(torch.float32), dim=1)
#
#         # Compose the final fingerprint from the previous simulated FP
#         # and the newly sampled positions
#         fp_simulated = (fp_simulated * (1 - fill_positions)) + (chosen_bit * fill_positions)
#
#         return fp_simulated
#
#     def sample_true(self, fp, fp_simulated, replace=0):
#         """
#         Replace a subset of the fingerprint with only correct predictions
#         """
#         # torch.manual_seed(seed)
#         replace_score = torch.rand(size=(fp.shape[0],)).cuda().unsqueeze(1) + 2 * (replace - 0.5)
#
#         # torch.manual_seed(seed)
#         replace_positions = torch.rand(size=fp.shape, dtype=torch.float32).cuda()
#         replace_positions = torch.as_tensor(replace_positions < replace_score, dtype=torch.float32).cuda()
#         sample_tp = fp * replace_positions
#         sample_tn = (1 - fp) * replace_positions
#         sample_fn = torch.zeros_like(sample_tn).cuda()
#         sample_fp = torch.zeros_like(sample_tn).cuda()
#
#         # torch.manual_seed(seed)
#         rand = torch.randint(0, 2 ** 31 - 1, size=fp.shape, dtype=torch.int32).cuda()
#         rand = rand.unsqueeze(1)
#         chosen_position = (rand % self.fp_sums.unsqueeze(0)
#                            + self.fp_cumsums.unsqueeze(0))
#
#         # Extract the value from the 1-d array (this is the easy part :) )
#         sampled_bit = self.fp_values[chosen_position]
#         #
#         # We now have a value to sample for every bit AND stratum of the
#         # query fingerprints. By stacking together the "stratum choice" and
#         # multiplying, then summing the stratum values for each bit,
#         # we find which value to actually add to the fingerprint.
#         # Reminder: Let TP=0,FP=1,FN=2,TN=3
#         chosen_stratum = torch.stack([sample_tp, sample_fp, sample_fn, sample_tn], dim=1)
#         chosen_bit = torch.sum(
#             sampled_bit * chosen_stratum.to(torch.float32), dim=1)
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
#     def sample_(self, fp):
#         with torch.no_grad():
#             replace_random = self.replace_random  # Convert TensorFlow scalar tensor to Python scalar
#             fill_remaining = self.fill_remaining  # Convert TensorFlow scalar tensor to Python scalar
#             final_jitter = self.final_jitter  # Convert TensorFlow scalar tensor to Python scalar
#             final_round = self.final_round  # Convert TensorFlow scalar tensor to Python scalar
#
#             # fp = torch.from_numpy(fp).cuda()
#             fp = fp.cuda()
#             fp_simulated = torch.zeros_like(fp, dtype=torch.float32).cuda()  # Initialize simulated fingerprints
#             fp_positions_empty = torch.ones_like(fp).cuda()  # Initialize positions empty indicator
#
#             # Call sample_steps function
#             fp_simulated, fp_positions_empty = self.sample_steps(fp, fp_simulated, fp_positions_empty)
#
#             # Fill all still unfilled positions if fill_remaining is set
#             fp_simulated = torch.where(
#                 fill_remaining,
#                 self.sample_remaining(fp, fp_simulated, fp_positions_empty, replace_random.item()),
#                 fp_simulated
#             )
#
#             # Replace random positions with "true prediction samples" if desired
#             fp_simulated = self.sample_true(fp, fp_simulated, self.replace_true.item())
#
#             # Add jitter and clip, if desired
#             # torch.manual_seed(seed)
#             fp_noise = final_jitter * (torch.rand_like(fp_simulated) - 0.5)
#             fp_simulated = torch.clamp(fp_simulated + fp_noise, fp_simulated.min(), fp_simulated.max())
#
#             # Round the fingerprint if final_round is set
#             if final_round:
#                 fp_simulated = torch.round(fp_simulated)
#
#             return fp_simulated.cpu(), fp_positions_empty


def t_CVSampler():
    # Prepare test data
    fp_true = np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 1, 1, 0]
    ])  # 2 true fingerprints

    fp_predicted = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.8],
        [0.0, 0.0, 0.0, 0.6, 0.7],
        [0.0, 0.0, 0.6, 0.9, 0.9],
        [0.0, 0.7, 0.9, 0.8, 0.6],
        [0.8, 0.6, 0.7, 0.8, 0.9],
        [0.7, 0.0, 0.8, 0.9, 0.0]
    ])  # Corresponding predicted fingerprints

    # Create an instance of CVSampler
    sampler = CVSampler(fp_true, fp_predicted)

    # Prepare input fingerprints
    # fp_input = torch.tensor([[0, 1, 0, 1, 0],
    #                          [1, 0, 1, 1, 1],
    #                          [1, 1, 1, 0, 0],
    #                          [1, 1, 1, 1, 1]], dtype=torch.float32)
    # fp_input = np.array([[0, 1, 0, 1, 0],
    #                      [1, 0, 1, 1, 1],
    #                      [1, 1, 1, 0, 0],
    #                      [1, 1, 1, 1, 1]], dtype=np.float32)
    # fp_input = np.random.rand(4, 5)
    fp_input = torch.rand(4, 5)
    # Call the sample method
    simulated_fp, _ = sampler.sample_(fp_input)
    simulated_fp_, _ = sampler.sample_(fp_input)
    print(simulated_fp.equal(simulated_fp_))

    # fp_input1 = np.array([[0, 1, 0, 1, 0],
    #                       [1, 0, 1, 1, 1]], dtype=np.float32)
    # fp_input2 = np.array([[1, 1, 1, 0, 0],
    #                       [1, 1, 1, 1, 1]], dtype=np.float32)
    fp_input1 = fp_input[:2, :]
    fp_input2 = fp_input[2:, :]
    simulated_fp1, _ = sampler.sample_(fp_input1)
    simulated_fp2, _ = sampler.sample_(fp_input2)
    simulated_fp3 = torch.cat([simulated_fp1, simulated_fp2], dim=0)
    print(simulated_fp.equal(simulated_fp3))
    print("CVSampler sample method test passed.")


if __name__ == "__main__":
    t_CVSampler()

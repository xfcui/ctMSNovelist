import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import model_ms2smiles
import model_msnovelist
import my_config as mc
# from model_ms2smiles.beam_model.beam_search_decoder import BeamSearchDecoder
from fp_management import fp_database as db
from fp_management import fingerprinting as fpr
from fp_management import fingerprint_map as fpm
from rdkit import RDLogger
import infrastructure.utils as utils
import infrastructure.score as msc
from processData.testDataset import PklDataset

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def get_all_file_paths(weights_dir):
    file_paths = []
    for dir_path, _, filenames in os.walk(weights_dir):
        for filename in filenames:
            file_path = os.path.join(dir_path, filename)
            file_paths.append(file_path)
        break
    return sorted(file_paths)


def cal_metrics_5(results_evaluated, eval_key, args):
    batch_size = args.batch_size
    beam_width = args.beam_width
    pkl_folder = args.pkl_folder
    require_pkl = args.require_pkl
    # 根据预测的数据计算valid smiles，correct mf和%found5个指标
    # 如果require_pkl=True，则需要保存pkl文件用于计算top-k
    results_complete = pd.concat(results_evaluated)
    results_complete["nn"] = batch_size * results_complete["block_id"] + results_complete["n"]
    results_complete["evaluation_set"] = eval_key

    results_complete["inchikey1_match"] = (  # retrieval smiles
            results_complete["inchikey1"] == results_complete["inchikey1_ref"]
    )
    results_complete["mf_match"] = (  # correct MF
            results_complete["mf"] == results_complete["mf_ref"]
    )
    results_complete["valid_smiles"] = (  # valid smiles
            results_complete["inchikey1"] != ''
    )
    results_per_candidate = results_complete.groupby(["nn"])
    results_summary = results_per_candidate[
        ["valid_smiles", "mf_match", "inchikey1_match"]
    ].aggregate(np.sum)

    results_summary_mean = results_summary.mean()
    results_summary_mean["valid_smiles"] = results_summary_mean["valid_smiles"] / beam_width
    results_summary_mean["mf_match"] = results_summary_mean["mf_match"] / beam_width

    results_summary_bool = results_summary >= 1
    # print(weight_path)
    print(results_summary_bool.mean())
    print(results_summary_mean[:2])



    results_bool = results_summary_bool.mean()
    results_mean = results_summary_mean[:2]

    new_names_bool = ["valid_smiles #>0", "correct_mf #>0", "%found"]
    new_names_mean = ["valid_smiles mean%", "correct_mf mean%"]
    results_bool = results_bool.rename(index=dict(zip(results_bool.index, new_names_bool)))
    results_mean = results_mean.rename(index=dict(zip(results_mean.index, new_names_mean)))


    metrics_5 = pd.concat([results_bool, results_mean], axis=0).to_frame().T
    if require_pkl:
        if not os.path.exists(pkl_folder):
            os.makedirs(pkl_folder)
        pickle_path = pkl_folder + "/eval_" + eval_key + ".pkl"
        pickle.dump(results_complete, open(pickle_path, "wb"))
    # csv_path = f"out_eval_fold{cv_fold}_bs{batch_size}.csv"
    # results_complete.to_csv(csv_path, index=False)
    # print("================================")
    return metrics_5


@torch.no_grad()
def calculate_metrics_fold(model_dict, fingerprinter, weight_path, args):
    model_transcode = model_dict["model_transcode"]
    model_encode = model_dict["model_encode"]
    model_decode = model_dict["model_decode"]
    model_transvae = model_dict["model_transvae"]

    model_transcode.cpu()
    model_encode.cpu()
    model_decode.cpu()
    print(f"load {weight_path}")
    if args.ctFlag:
        model_transvae.load_state_dict(state_dict=torch.load(weight_path))
        model_transcode.load_state_dict(state_dict=model_transvae.transcoder.state_dict())
        model_encode.load_state_dict(state_dict=model_transvae.transcoder.state_dict())
        model_decode.load_state_dict(state_dict=model_transvae.transcoder.state_dict())
    else:
        model_transcode.load_state_dict(state_dict=torch.load(weight_path))
        model_encode.load_state_dict(state_dict=torch.load(weight_path))
        model_decode.load_state_dict(state_dict=torch.load(weight_path))

    model_transcode.cuda().eval()
    model_encode.cuda().eval()
    model_decode.cuda().eval()

    steps = model_transcode.steps + 1  # 127
    batch_size = args.batch_size
    beam_width = args.beam_width

    if args.model_mode == "ms2smiles":
        model_beam = model_ms2smiles.beam_model.beam_search_decoder.BeamSearchDecoder(model_encode, model_decode,
                                                                                      steps=steps, n=batch_size,
                                                                                      k=beam_width, kk=beam_width,
                                                                                      config=mc.config, )
    elif args.model_mode == "msnovelist":
        model_beam = model_msnovelist.beam_model.beam_search_decoder.BeamSearchDecoder(model_encode, model_decode,
                                                                                       steps=steps, n=batch_size,
                                                                                       k=beam_width, kk=beam_width,
                                                                                       config=mc.config, )
    else:
        return
    cv_fold = os.path.basename(weight_path)[0]
    print(f"fold{cv_fold}")
    eval_key = f"fold{cv_fold}-{args.eval_set}"

    # 测试集中有数据的MF是None
    test_set = PklDataset("gnps", cv_fold=cv_fold)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    result_blocks = []
    reference_blocks = []
    for i, batch_data in enumerate(tqdm(test_loader)):
        # if i==1:
        #     break
        # if i!=281:
        #     continue
        fp, mf, gt, lab, smiles_str = batch_data
        fp = fp.cuda()
        mf = mf.cuda()
        gt = gt.cuda()

        data_X = {"FP": fp, "MF": mf}
        data_k = {key: x.repeat_interleave(beam_width, dim=0) for key, x in data_X.items()}
        # 最后一个batch的数据数目不一定是batch_size
        n_real = len(mf)
        if n_real != batch_size:
            if args.model_mode == "ms2smiles":
                model_beam = model_ms2smiles.beam_model.beam_search_decoder.BeamSearchDecoder(model_encode,
                                                                                              model_decode,
                                                                                              steps=steps, n=n_real,
                                                                                              k=beam_width,
                                                                                              kk=beam_width,
                                                                                              config=mc.config, )
            elif args.model_mode == "msnovelist":
                model_beam = model_msnovelist.beam_model.beam_search_decoder.BeamSearchDecoder(model_encode,
                                                                                               model_decode,
                                                                                               steps=steps, n=n_real,
                                                                                               k=beam_width,
                                                                                               kk=beam_width,
                                                                                               config=mc.config, )

        states_init = model_encode(data_k)
        sequences, y, scores = model_beam.decode_beam(states_init)
        seq, score, length = model_beam.beam_traceback(sequences, y, scores)
        smiles = model_beam.sequence_ytoc(seq)

        results_df = model_beam.format_results(smiles, score)
        result_blocks.append(results_df)
        reference_df = model_beam.format_reference(
            smiles_str,
            [d for d in fp.cpu().numpy()])
        reference_blocks.append(reference_df)

    if not args.require_pkl:  # 不生成pkl文件
        results_evaluated = []
        for block_, ref_, block_id in zip(tqdm(result_blocks),
                                          reference_blocks,
                                          range(len(result_blocks))):
            # Make a block with molecule, MF, smiles for candidates and reference
            block = db.process_df(block_, fingerprinter,
                                  construct_from="smiles",
                                  block_id=block_id)
            # false
            retain_single_duplicate = False
            if retain_single_duplicate:
                block.sort_values("score", ascending=False, inplace=True)
                block = block.groupby(["n", "inchikey1"]).first().reset_index()

            ref = db.process_df(ref_, fingerprinter,
                                construct_from="smiles",
                                block_id=block_id)

            # Match ref to predictions
            block = block.join(ref, on="n", rsuffix="_ref")

            results_evaluated.append(block)

    else:  # 生成pkl文件
        results_evaluated = []
        for block_, ref_, block_id in zip(tqdm(result_blocks),
                                          reference_blocks,
                                          range(len(result_blocks))):
            # Make a block with molecule, MF, smiles for candidates and reference
            block = db.process_df(block_, fingerprinter,
                                  construct_from="smiles",
                                  block_id=block_id)
            retain_single_duplicate_ = True
            if retain_single_duplicate_:
                block.sort_values("score", ascending=False, inplace=True)
                block = block.groupby(["n", "inchikey1"]).first().reset_index()

            ref = db.process_df(ref_, fingerprinter,
                                construct_from="smiles",
                                block_id=block_id)
            # Also actually compute the true fingerprint for the reference
            fingerprinter.process_df(ref,
                                     out_column="fingerprint_ref_true",
                                     inplace=True)

            # Match ref to predictions
            block = block.join(ref, on="n", rsuffix="_ref")
            # Keep only correct formula
            block_ok = block.loc[block["inchikey1"].notna()].loc[block["mf"] == block["mf_ref"]]
            # Now actually compute the fingerprints, only for matching MF
            fingerprinter.process_df(block_ok,
                                     inplace=True)
            block = block.merge(
                block_ok[["n", "k", "fingerprint"]],
                left_on=["n", "k"],
                right_on=["n", "k"],
                suffixes=["_ref", ""],
                how="left")

            results_evaluated.append(block)

    metrics_5 = cal_metrics_5(results_evaluated, eval_key, args)

    return metrics_5


def calculate_metrics_folds(model_dict, fingerprinter, args):
    # 获取权重文件路径列表
    weights_path_ls = get_all_file_paths(args.weights_dir)
    # weights_path_ls = [weights_path_ls[0]]
    require_pkl = args.require_pkl
    save_dir = args.pkl_folder

    metrics_5_folds = pd.DataFrame()
    metrics_all_folds = pd.DataFrame()
    for weight_path in weights_path_ls:
        cv_fold = os.path.basename(weight_path)[0]
        print(f"fold{cv_fold}, starting evaluation...")
        metrics_5 = calculate_metrics_fold(model_dict, fingerprinter, weight_path, args)
        metrics_5["weight_file"] = weight_path
        metrics_5["cv_fold"] = cv_fold
        metrics_5_folds = pd.concat([metrics_5_folds, metrics_5], ignore_index=True)

        if require_pkl:
            print(print(f"fold{cv_fold}, starting calculate top-k acc..."))

            results_bool, results_mean, results_topk = rank_score_metrics(cv_fold=cv_fold,
                                                                          weight_file=weight_path,
                                                                          args=args)
            new_names_bool = ["valid_smiles #>0", "correct_mf #>0", "%found"]
            new_names_mean = ["valid_smiles mean%", "correct_mf mean%"]
            results_bool = results_bool.rename(index=dict(zip(results_bool.index, new_names_bool)))
            results_mean = results_mean.rename(index=dict(zip(results_mean.index, new_names_mean)))

            combined_data = pd.concat([results_bool, results_mean, results_topk], axis=0).to_frame().T
            combined_data["weight_file"] = weight_path
            combined_data["cv_fold"] = cv_fold

            metrics_all_folds = pd.concat([metrics_all_folds, combined_data], ignore_index=True)

    # 把 top-k 之外的指标保存到 CSV 文件
    metrics_5_save_path = save_dir + '/metrics_5_output.csv'

    # 选择数值列
    numeric_columns = metrics_5_folds.select_dtypes(include=[float, int]).columns
    # 计算数值列的均值
    mean_values = metrics_5_folds[numeric_columns].mean()

    metrics_5_folds.loc[len(metrics_5_folds)] = mean_values
    last_index = len(metrics_5_folds) - 1
    # 将最后一行的 cv_fold 列的值设置为 "all"
    metrics_5_folds.at[last_index, 'cv_fold'] = 'all'

    metrics_5_folds.to_csv(metrics_5_save_path, index=False)
    if require_pkl:
        # 所有指标保存到 CSV 文件
        metrics_all_save_path = save_dir + '/metrics_all_output.csv'
        numeric_columns = metrics_all_folds.select_dtypes(include=[float, int]).columns
        mean_values = metrics_all_folds[numeric_columns].mean()

        metrics_all_folds.loc[len(metrics_all_folds)] = mean_values
        last_index = len(metrics_all_folds) - 1
        # 将最后一行的 cv_fold 列的值设置为 "all"
        metrics_all_folds.at[last_index, 'cv_fold'] = 'all'
        metrics_all_folds.to_csv(metrics_all_save_path, index=False)


def rank_score_metrics(cv_fold, weight_file, args):
    # eval_id = "fold0-gnps"  # fold0-casmi
    beam_width = args.beam_width
    eval_id = f"fold{cv_fold}-{args.eval_set}"
    eval_counter = 0
    pickle_id = eval_id

    # mc.config.setdefault('cv_fold', 0)
    eval_key = f"fold{cv_fold}-{args.eval_set}"

    evaluation_logger = utils.EvaluationLogger("topn", mc.config,
                                               eval_id, eval_counter, pickle_id)

    picklepath = {pickle_id: args.pkl_folder + "/eval_" + pickle_id + ".pkl"}
    print(f"load {picklepath}")

    def check_dict(v):
        if isinstance(v, dict):
            return v[eval_key]
        else:
            return v

    results_complete = {k: pickle.load(open(pp, 'rb')) for k, pp in picklepath.items()}
    results_complete = {k: check_dict(v) for k, v in results_complete.items()}
    results_complete = pd.concat([r[["nn", "mol", "mol_ref", "mf", "mf_ref",
                                     "fingerprint", "fingerprint_ref", "fingerprint_ref_true",
                                     "inchikey1", "inchikey1_ref",
                                     "score", "smiles", "smiles_ref"]].assign(source=k)
                                  for k, r in results_complete.items()])

    n_total_ = len(set(results_complete["nn"]))
    # kk = mc.config["eval_kk"]
    # k = mc.config["eval_k"]
    kk = beam_width
    k = beam_width
    # f1_cutoff = mc.config["f1_cutoff"]
    f1_cutoff = 0

    # ===========================results_ok:预测正确的MF对应的数据==========================================
    # 1、计算得分
    # 2、对每条数据的得分排序
    # 3、绘图分析
    # 删除nan的fingerprint, 只有预测正确的MF的fingerprint不是nan, results_ok中预测的MF都是正确的
    results_ok = results_complete.loc[results_complete["fingerprint"].notna()].copy()

    n_results_ok = len(results_ok)
    # logger.info(f"Scoring fingerprints for {n_results_ok} results with correct MF")

    # 1、计算得分
    fp_map = fpm.FingerprintMap(mc.config["fp_map"])
    scores = msc.get_candidate_scores()
    results_ok = msc.compute_candidate_scores(results_ok, fp_map,
                                              additive_smoothing_n=n_total_,
                                              f1_cutoff=f1_cutoff)
    # Add "native" scoring i.e. using the decoder output directly
    results_ok["score_decoder"] = results_ok["score"]
    scores.update({"score_decoder": None})

    # logger.info(f"Scoring fingerprints - done")

    # 2、对每条数据的得分排序
    for score in scores.keys():  # 对每种得分的候选smiles进行排序, rank_score为候选项的排名, (注意这里并没有确认候选项与smiles标签是否一致)
        results_ok["rank_" + score] = results_ok.groupby("nn")[score].rank(ascending=False, method="first")

    # 3、绘图分析
    n_rank_cols = ["nn"] + ["rank_" + score for score in scores.keys()]
    results_match_ranks = results_ok.loc[  # 选择预测正确的smiles, 这里对于同一个smiles可能有多个预测正确
        results_ok["inchikey1"] == results_ok["inchikey1_ref"]][n_rank_cols]

    # 对rank进行分组聚合, 如果beam-width有多个候选项预测正确, 则只选择排名高的
    results_top_rank = results_match_ranks.groupby("nn").aggregate(np.min)

    def ecdf(data):
        """ Compute ECDF """
        x = np.sort(data)
        n = x.size
        y = np.arange(1, n + 1) / n
        return (x, y)

    results_ranks_ecdf = {}
    for score in scores.keys():  # 从小到大对所有预测正确的smiles出现的位置进行排序
        results_ranks_ecdf.update({score: ecdf(results_top_rank["rank_" + score])})

    # # fig, ax = plt.subplot()
    # for score, data, i in zip(results_ranks_ecdf.keys(),
    #                           results_ranks_ecdf.values(),
    #                           range(len(results_ranks_ecdf))):
    #     plt.plot(data[0], data[1], color='C' + str(i))
    # plt.xscale("log")
    # plt.xlabel("rank inchikey1 hit")
    # plt.ylabel("ECDF")
    # plt.legend(scores.keys())
    # plt.show()
    # for axis in [ax.xaxis, ax.yaxis]:
    #     axis.set_major_formatter(ScalarFormatter())

    # ===========================计算评估指标==========================================
    results_complete["inchikey1_match"] = (  # retrieval smiles
            results_complete["inchikey1"] == results_complete["inchikey1_ref"]
    )
    results_complete["mf_match"] = (  # correct MF
            results_complete["mf"] == results_complete["mf_ref"]
    )
    results_complete["valid_smiles"] = (  # valid smiles
            results_complete["inchikey1"] != ''
    )
    # results_complete_lib是label, results_with_ranks是预测正确的数据
    results_complete_lib = results_complete[["nn", "mol_ref", "fingerprint_ref", "mf_ref"]].groupby("nn").first()
    results_with_ranks = results_complete_lib.join(results_match_ranks.set_index("nn"))  # 左外连接
    rank_cols = ["rank_" + score for score in scores.keys()]
    results_with_ranks[rank_cols] = results_with_ranks[rank_cols].fillna(value=kk + 1)  # 预测错误的位置填充beam_width+1

    results_per_candidate = results_complete.groupby(["nn"])
    results_summary = results_per_candidate[
        ["valid_smiles", "mf_match", "inchikey1_match"]
    ].aggregate(np.sum)
    # 计算valid smiles, correct MF # > 0
    results_summary_bool = results_summary >= 1
    # results_summary_bool.mean()
    results_summary_mean = results_summary.mean()
    results_summary_mean["valid_smiles"] = results_summary_mean["valid_smiles"] / kk
    results_summary_mean["mf_match"] = results_summary_mean["mf_match"] / kk

    # Add a table for valid and correct-MF SMILES per challenge

    results_mf_summary = results_summary.copy()
    results_mf_summary.reset_index(inplace=True)
    results_mf_summary.set_index("nn", inplace=True)
    all_res = results_complete.groupby("nn").first()[
        ["fingerprint", "fingerprint_ref", "fingerprint_ref_true",
         "mol_ref", "mf_ref"]]
    results_mf_summary = results_mf_summary.join(all_res)
    results_mf_summary = results_mf_summary.loc[results_mf_summary["fingerprint_ref_true"].notna()]
    # 计算spec-fp与对应的struct-fp相似度以及分子量
    results_mf_summary = msc.compute_fp_quality_mw(results_mf_summary, fp_map)

    results_mf_summary.reset_index(inplace=True)

    results_keys = ["valid_smiles", "mf_match"]
    for key in results_keys:
        results_mf_summary["rank"] = results_mf_summary[key].rank(  # 在n条test data中，对valid smiles数目进行排名
            ascending=False, method='first')
        results_mf_summary["value"] = results_mf_summary[key] / kk
        results_mf_summary["eval_score"] = key
        results_mf_summary["eval_metric"] = "rank"

        # evaluation_logger.append_csv(
        #     "valid_mf",
        #     results_mf_summary[["nn", "rank", "value", "eval_score",
        #                         "eval_metric", "predicted_fp_quality", "mol_weight"]]
        # )

    results_topk_ = {}
    results_top_rank = results_with_ranks[rank_cols]
    for rank in [1, 1.99, 5, 10, 20]:
        results_topk_.update({'top_' + str(rank): (results_top_rank <= rank).mean()})
    results_topk = pd.DataFrame(results_topk_)

    # results_topk_ = {}
    # results_top_rank = results_with_ranks[rank_cols]
    # grouped = results_top_rank.groupby(results_top_rank.index)
    # for rank in [1, 1.99, 5, 10, 20]:
    #     # 对每个组进行布尔判断，只要组中有一个数 <= rank，则该组为 True
    #     topk_counts = grouped.apply(lambda x: (x <= rank).any()).mean()
    #     results_topk_.update({'top_' + str(rank): topk_counts})
    # results_topk = pd.DataFrame(results_topk_)
    #########################
    # score_fivenum and tanimoto_fivenum indicators:
    # score_fivenum: five-number statistic of best scores reached per query
    # tanimoto_fivenum: five-number statistics of tanimoto similarity of
    #    top-scoring candidate per query,
    #    i.e. compound similarity of best match
    #########################
    score_fivenum_ = {}
    tanimoto_fivenum_ = {}

    for score in scores.keys():
        # get rank-1 candidates for each query  这里只有126条（总数127）？？？
        results_top1_score = results_ok.loc[results_ok["rank_" + score] == 1].copy()
        results_top1_score = results_top1_score.loc[results_top1_score["fingerprint_ref_true"].notna()]

        # get score_fivenum for rank-1 scandidates
        score_fivenum_.update({"fivenum_" + score: results_top1_score[score].describe()})
        # get fingerprints and calculate tanimoto for rank-1 candidates
        fingerprint_candidate_rank1 = np.concatenate(results_top1_score["fingerprint"].tolist())
        fingerprint_truematch = np.concatenate(results_top1_score["fingerprint_ref_true"].tolist())
        results_top1_score["match_tanimoto"] = list(map(
            lambda x: msc.fp_tanimoto(x[0], x[1]), zip(fingerprint_candidate_rank1, fingerprint_truematch)
        ))
        tanimoto_fivenum_.update({"tanimoto_" + score: results_top1_score["match_tanimoto"].describe()})

    score_fivenum = pd.DataFrame.from_records(score_fivenum_).transpose()
    tanimoto_fivenum = pd.DataFrame.from_records(tanimoto_fivenum_).transpose()
    print(score_fivenum)
    print(tanimoto_fivenum)

    #########################
    # Top-n evaluation: % spectra < rank n € {1, 1.99, 5, 10, 20}
    # for different scores.
    #########################

    # logger.info("Evaluation overall:")
    print(results_summary_bool.mean())
    # logger.info("Evaluation top-n ECDF:")
    print(results_topk)

    # evaluation_logger.append_txt(
    #     key='summary',
    #     data={'Evaluation set': eval_key,
    #           'Weights': weight_file,
    #           'Beam width': k,
    #           'Top-k': kk,
    #           # 'Pipeline': pipeline_encoder,
    #           'n': n_total_,
    #           'Total MF OK': n_results_ok,
    #           'Summary SMILES # > 0': results_summary_bool.mean(),
    #           'Summary SMILES % mean': results_summary_mean,  #
    #           'Results top-k ECDF': results_topk,  # results_topk.loc["rank_score_mod_platt"]
    #           'Best scores summary': score_fivenum,
    #           'Rank-1 tanimoto summary': tanimoto_fivenum
    #           })

    # evaluation_logger.append_csv("identity", pd.DataFrame(results_summary_bool.mean()).transpose())
    # evaluation_logger.append_csv("score_fivenum", score_fivenum)
    # evaluation_logger.append_csv("tanimoto_fivenum", tanimoto_fivenum)
    # evaluation_logger.append_csv("ranks", results_topk)
    results_ok_out = results_ok.copy()
    results_ok_out.drop(["fingerprint", "fingerprint_ref", "fingerprint_ref_true", "mol", "mol_ref"], axis=1,
                        inplace=True)
    results_ok_out["mf"] = results_ok_out.mf.apply(msc.formula_to_string)
    results_ok_out["mf_ref"] = results_ok_out.mf_ref.apply(msc.formula_to_string)
    # evaluation_logger.append_csv("results_ok_ranked", results_ok_out)

    if mc.config["eval_detail"]:
        for score in scores.keys():
            # get rank-1 candidates for each query
            results_top1_score = results_ok.loc[results_ok["rank_" + score] == 1].copy()
            results_top1_score = results_top1_score.loc[results_top1_score["fingerprint_ref_true"].notna()]
            results_top1_score = msc.compute_fp_quality_mw(results_top1_score, fp_map)
            # get score_fivenum for rank-1 scandidates
            results_top1_score["rank"] = results_top1_score[score].rank(ascending=False, method='first')
            results_top1_score["value"] = results_top1_score[score]
            results_top1_score["eval_score"] = score
            results_top1_score["eval_metric"] = "score"
            evaluation_logger.append_csv(
                "score_quantiles",
                results_top1_score[["nn", "rank", "value", "eval_score",
                                    "eval_metric", "predicted_fp_quality", "mol_weight"]],
            )

        for score in scores.keys():
            # get rank-1 candidates for each query
            results_top1_score = results_ok.loc[results_ok["rank_" + score] == 1].copy()
            results_top1_score = results_top1_score.loc[results_top1_score["fingerprint_ref_true"].notna()]
            results_top1_score = msc.compute_fp_quality_mw(results_top1_score, fp_map)
            # get score_fivenum for rank-1 scandidates
            results_top1_score["rank"] = results_top1_score["score_mod_platt"].rank(ascending=False, method='first')
            results_top1_score["value"] = results_top1_score["score_mod_platt"]
            results_top1_score["eval_score"] = score
            results_top1_score["eval_metric"] = "score"
            evaluation_logger.append_csv(
                "platt_quantiles",
                results_top1_score[["nn", "rank", "value", "eval_score",
                                    "eval_metric", "predicted_fp_quality", "mol_weight"]],
            )

        for score in scores.keys():
            # get rank-1 candidates for each query
            results_ref_mols = results_per_candidate.first()[["mol_ref", "fingerprint_ref_true", "fingerprint_ref"]]
            results_top1_score = results_top_rank.copy()
            results_top1_score = results_top1_score.join(results_ref_mols)
            results_top1_score = results_top1_score.loc[results_top1_score["fingerprint_ref_true"].notna()]

            results_top1_score.reset_index(inplace=True)
            results_top1_score = msc.compute_fp_quality_mw(results_top1_score, fp_map)

            # get score_fivenum for rank-1 scandidates
            results_top1_score["rank"] = results_top1_score["rank_" + score].rank(ascending=True, method='first')
            results_top1_score["value"] = results_top1_score["rank_" + score]
            results_top1_score["eval_score"] = score
            results_top1_score["eval_metric"] = "rank"
            evaluation_logger.append_csv(
                "rank_quantiles",
                results_top1_score[["nn", "rank", "value", "eval_score",
                                    "eval_metric", "predicted_fp_quality", "mol_weight"]],
            )

        # Spearman rank corr

        def weighted_spearman_loss(yhat, y):
            spearman_loss = np.square(yhat - y)
            weight = 1 / y
            return np.sum(weight * spearman_loss)

        results_ref_mols = results_per_candidate.first()[["mol_ref", "fingerprint_ref_true", "fingerprint_ref"]]
        results_ref_mols = results_ref_mols.loc[results_ref_mols["fingerprint_ref_true"].notna()]

        results_cor_group = results_ok.groupby("nn")[["rank_score_decoder", "rank_score_mod_platt"]]
        results_cor = results_cor_group.corr(weighted_spearman_loss).unstack().iloc[:, 1:2]
        results_cor.columns = ["value"]
        results_cor["eval_score"] = "weighted_spearman_loss"
        results_cor = results_cor.join(results_ref_mols)
        # results_cor = results_cor.loc[results_cor["fingerprint"].notna()]
        results_cor = results_cor.loc[results_cor["fingerprint_ref"].notna()]
        results_cor = results_cor.loc[results_cor["fingerprint_ref_true"].notna()]
        results_cor = results_cor.copy()
        results_cor = msc.compute_fp_quality_mw(results_cor, fp_map)

        evaluation_logger.append_csv(
            "rank_corr",
            results_cor
        )

        results_cor_group = results_ok.groupby("nn")[["rank_score_decoder", "rank_score_mod_platt"]]
        results_cor = results_cor_group.corr('spearman').unstack().iloc[:, 1:2]
        results_cor.columns = ["value"]
        results_cor["eval_score"] = "spearman_cor"
        results_cor = results_cor.join(results_ref_mols)
        # results_cor = results_cor.loc[results_cor["fingerprint"].notna()]
        results_cor = results_cor.loc[results_cor["fingerprint_ref"].notna()]
        results_cor = results_cor.loc[results_cor["fingerprint_ref_true"].notna()]
        results_cor = results_cor.copy()
        results_cor = msc.compute_fp_quality_mw(results_cor, fp_map)

        evaluation_logger.append_csv(
            "rank_corr",
            results_cor
        )

        results_ok_out = results_ok.copy()
        results_ok_out.drop(["fingerprint", "fingerprint_ref", "fingerprint_ref_true", "mol", "mol_ref"], axis=1,
                            inplace=True)
        results_ok_out["mf"] = results_ok_out.mf.apply(msc.formula_to_string)
        results_ok_out["mf_ref"] = results_ok_out.mf_ref.apply(msc.formula_to_string)
        evaluation_logger.append_csv("results_ok_ranked", results_ok_out)

    return results_summary_bool.mean(), results_summary_mean[:2], results_topk.loc["rank_score_mod_platt"]


def start_main():
    parser = argparse.ArgumentParser(description='Process some integers.')

    # 添加命令行参数
    parser.add_argument('--expmId', type=str, default="m2", help='Experiment ID')
    parser.add_argument('--beam_width', type=int, default=128, help='Beam width for decoding')
    # msnovelist ms2smiles
    parser.add_argument('--model_mode', type=str, default="msnovelist", help='Model mode')
    # parser.add_argument('--require_pkl', action='store_true', help='Require PKL flag')

    parser.add_argument('--deviceID', type=int, default=0, help='Device ID to use')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--eval_set', type=str, default="gnps", help='Evaluation set name')
    parser.add_argument('--cv_folds', type=int, default=10, help='Number of cross-validation folds')
    parser.add_argument('--retain_single_duplicate', action='store_true', help='Retain single duplicates')
    parser.add_argument('--base_folder', type=str, default='/home/sf123/ctMSNovelist', help='Base folder path')

    parser.add_argument('--weights_dir', type=str, help='Directory containing model weights')
    parser.add_argument('--pkl_folder', type=str, help='Directory containing model pkl results')
    parser.add_argument('--ctFlag', action='store_true', help='CT flag')

    args = parser.parse_args()
    args.require_pkl = True
    print(f"require_pkl: {args.require_pkl}")
    if args.expmId in ["m3", "m5", "m7"]:
        args.ctFlag = True
    else:
        args.ctFlag = False

    args.weights_dir = os.path.join(args.base_folder, "evaluation/weights_end", args.expmId)
    args.pkl_folder = os.path.join(args.base_folder, "evaluation/topk", args.expmId, args.beam_width.__str__())

    # device = "cuda"
    torch.cuda.set_device(f"cuda:{args.deviceID}")

    # 初始化指纹映射和其他配置
    fpr.Fingerprinter.init_instance(mc.config['fingerprinter_path'],
                                    mc.config['fingerprinter_threads'],
                                    capture=False,
                                    cache=mc.config['fingerprinter_cache'])
    fingerprinter = fpr.Fingerprinter.get_instance()
    decoder_input_size = 39
    decoder_output_size = 39

    # 根据model_mode选择不同的模型类
    if args.model_mode == "msnovelist":
        decoder_input_size += (256 + 11)
        model_encode = model_msnovelist.EncoderModel(config=mc.config, decoder_input_size=decoder_input_size,
                                                     out_size=decoder_output_size)
        model_decode = model_msnovelist.DecoderModel(config=mc.config, decoder_input_size=decoder_input_size,
                                                     out_size=decoder_output_size)
        model_transcode = model_msnovelist.TranscoderModel(config=mc.config, decoder_input_size=decoder_input_size,
                                                           out_size=decoder_output_size)
        model_transvae = model_msnovelist.TransVAEModel()
    elif args.model_mode == "ms2smiles":
        decoder_input_size += 512
        model_encode = model_ms2smiles.EncoderModel(config=mc.config, decoder_input_size=decoder_input_size,
                                                    out_size=decoder_output_size)
        model_decode = model_ms2smiles.DecoderModel(config=mc.config, decoder_input_size=decoder_input_size,
                                                    out_size=decoder_output_size)
        model_transcode = model_ms2smiles.TranscoderModel(config=mc.config, decoder_input_size=decoder_input_size,
                                                          out_size=decoder_output_size)
        model_transvae = model_ms2smiles.TransVAEModel()
    else:
        return
    model_dict = {
        "model_encode": model_encode,
        "model_decode": model_decode,
        "model_transcode": model_transcode,
        "model_transvae": model_transvae,
    }

    # 计算指标
    calculate_metrics_folds(model_dict, fingerprinter, args)


if __name__ == "__main__":
    start_main()
    print("h")
    print("h")

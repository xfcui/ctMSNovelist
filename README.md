项目结构
- MS_DATA (数据文件夹)
    - combined_0824_v44.db 训练与验证数据
    - complete_folds_smiles_holdout_me.pkl 测试数据, CANOPUS+GNPS+CASMI
    - csi_fingerid.csv 分子指纹映射 8925-->3609
- ctMSNovelist (工程文件夹)
  - evaluation
    - topk 存放测试结果
      - m2
      - m3
      - ...
      - m7
      - t1
      - t2
    - weights_end 存放模型参数文件
      - m2
      - m3
      - ...
      - m7
      - t1
      - t2
  - ... 

实验部分

|      |              |             |             |        |            | Params     |        | valid SMILES |        | correct MF |         |        |        | retrieval |
| ---- | ------------ | ----------- | ----------- | ------ | ---------- | ---------- | ------ | ------------ | ------ | ---------- | ------- | ------ | ------ | --------- |
|      | pre-training | fine-tuning | co-training | expmId | beam width | model_mode | mean % | # >0         | mean % | # > 0      | % found | top1   | top5   | top10     |
| m2   | ×            | √           | ×           | m2     | 128        | msnovelist | 31.40% | 99.00%       | 4.70%  | 60.00%     | 12.80%  | 8.70%  | 12.40% | 12.80%    |
| m3   | ×            | √           | √           | m3     | 128        | msnovelist | 45.90% | 99.70%       | 9.20%  | 83.40%     | 19.20%  | 12.00% | 18.10% | 18.70%    |
| m4   | √            | ×           | ×           | m4     | 128        | msnovelist | 78.30% | 100.00%      | 65.70% | 99.10%     | 40.70%  | 21.20% | 33.60% | 36.40%    |
| m5   | √            | ×           | √           | m5     | 128        | msnovelist | 79.90% | 100.00%      | 68.90% | 99.30%     | 40.90%  | 21.90% | 34.10% | 36.70%    |
| m6   | √            | √           | ×           | m6     | 128        | msnovelist | 78.10% | 100.00%      | 66.00% | 99.40%     | 45.30%  | 23.50% | 37.00% | 39.90%    |
| m7   | √            | √           | √           | m7     | 128        | msnovelist | 78.70% | 100.00%      | 60.50% | 99.50%     | 48.80%  | 25.40% | 40.30% | 43.80%    |
| m7.2 | √            | √           | √           | m7     | 16         | msnovelist | 83.52% | 99.85%       | 67.02% | 98.66%     | 37.87%  | 24.14% | 36.09% | 37.66%    |
| t1   | ×            | ×           | ×           | t1     | 128        | ms2smiles  | 80.11% | 100.00%      | 77.54% | 99.56%     | 42.23%  | 23.25% | 35.50% | 37.82%    |
| t2   | √            | √           | √           | t2     | 128        | ms2smiles  | 78.28% | 100.00%      | 73.10% | 99.66%     | 50.61%  | 26.11% | 41.75% | 45.17%    |

主实验：m7，m7.2

消融实验：m2~m7

副实验：t1，t2



运行命令

m2: python evaluate_loop_x.py --expmId m2

m3: python evaluate_loop_x.py --expmId m3

m4: python evaluate_loop_x.py --expmId m4

m5: python evaluate_loop_x.py --expmId m5

m6: python evaluate_loop_x.py --expmId m6

m7: python evaluate_loop_x.py --expmId m7

m7.2: python evaluate_loop_x.py --expmId m7 --beam_width=16

t1: python evaluate_loop_x.py --expmId t1 --model_mode ms2smiles

t2: python evaluate_loop_x.py --expmId t2 --model_mode ms2smiles
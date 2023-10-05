import sys
sys.path.append("../")
sys.path.append("../../")

from bert_pytorch.predict_log_modified import PredictorModified
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence
from drain3.template_miner import TemplateMiner
import torch
import numpy as np
import pandas as pd
from config import options

class Detector():
    def __init__(self):
        None

    def parsing(self, log_df):
        config = TemplateMinerConfig()
        persistence_handler = FilePersistence("drain3_state.json")  # 使用你之前保存状态的文件名
        template_miner = TemplateMiner(persistence_handler)

        # 获取第一个blk的值
        first_blk_value = log_df['blk'].iloc[0]

        # 提取第一个blk的所有日志
        first_blk_logs = log_df[log_df['blk'] == first_blk_value]['Content']

        logkeys = []

        # 使用template_miner.match解析这些日志
        for log in first_blk_logs:
            result = template_miner.match(log)
            if result is not None:
                # print(f"Log: {log}")
                # print(f"Matched Template ID: {result.cluster_id}")
                # print(f"Matched Template: {result.get_template()}\n")
                logkeys.append(result.cluster_id)
            else:
                print(f"No template matched for log: {log}\n")

        print(logkeys)

        return logkeys
    
    def detect(self, log_df):

        logkeys = self.parsing(log_df)

        predictor = PredictorModified(options)

        is_anomaly = predictor.predict_single_sequence(logkeys)

        return is_anomaly
    
    

if __name__ == "__main__":

    log_df = pd.read_csv('../output/hdfs/single_log_structured.csv', quotechar='"')

    detector = Detector()

    is_anomaly = detector.detect(log_df)

    print(f"The sequence is {'anomalous' if is_anomaly else 'normal'}.")
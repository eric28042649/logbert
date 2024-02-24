import sys
sys.path.append("../")
sys.path.append("../../")

from bert_pytorch.predict_log_batch import PredictorBatch
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence
from drain3.template_miner import TemplateMiner
import torch
import numpy as np
import pandas as pd

from option import options

class Detector():
    def __init__(self, options):
        self.options = options

    # Turn trace to logkeys format
    def process(self, log_df):
        config = TemplateMinerConfig()
        config.load('drain3.ini')
        config.drain_depth = 10
        persistence_handler = FilePersistence(self.options["model_dir"] + "/drain3_state.json")
        template_miner = TemplateMiner(persistence_handler, config)
        
        traceId = log_df['TraceId'].iloc[0]

        trace_logs = log_df[log_df['TraceId'] == traceId]['Content']

        logkeys = []
        logkeys_ignore = []
        
        for log in trace_logs:
            result = template_miner.match(log)
            if result is not None:
                logkeys.append(result.cluster_id)
                    
            else:
                print(f"No template matched for log: {log}\n")
                # test
                if log == "/smt-ecps-api/auth/GetCurrentUser":
                    log = "smt-ecps-api/Auth/GetCurrentUser"
                log = log.lstrip('/')
                result = template_miner.match(log)
                # print(f"Rematch...\n")
                if result is not None:
                    logkeys.append(result.cluster_id)
                    # print(f"Rematch success\n")
                else:
                    logkeys.append(0)
            
            if "api" in log and len(logkeys_ignore)<1:
                logkeys_ignore.append(len(logkeys)-1)

        # print(logkeys)
        # print(f"logkeys_ignore: {logkeys_ignore}")
        return logkeys, logkeys_ignore
    
    def detect_seq(self, log_df):
        grouped_logs = log_df.groupby('TraceId')

        logkeys = []
        logkeys_ignore = []
        trace_ids = []
        
        for trace_id, logs_df in grouped_logs:
            logkey, logkey_ignore = self.process(logs_df)
            logkeys.append(logkey)
            logkeys_ignore.append(logkey_ignore)
            trace_ids.append(trace_id)
            
        predictor = PredictorBatch(self.options)
        
        is_anomalies, test_results = predictor.predict_sequence(logkeys, logkeys_ignore)
    
        result_df = pd.DataFrame({'TraceId': trace_ids, 'is_anomaly': is_anomalies, 'result': test_results})
        return result_df

    
import sys
sys.path.append('../')

import sys
import os
import re
import json
import csv
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence
from itertools import islice
from collections import defaultdict

def parser(input_dir, output_dir, log_file):
    """
    对原始logs做parsing

    input: exported_data.csv
    output: log_structured.csv, log_templates.csv
    """
    filePersistence_file = os.path.join(output_dir, "../model/drain3_state.json")
    if os.path.exists(filePersistence_file):
        os.remove(filePersistence_file)
        
    config = TemplateMinerConfig()
    config.load('drain3.ini')
    # config.drain_sim_th = 0.1
    config.drain_depth = 10
    # config.max_children = 50
    
    persistence_handler = FilePersistence(filePersistence_file)
    
    template_miner = TemplateMiner(persistence_handler, config=config)
    
    log_structured_file = os.path.join(output_dir, "log_structured.csv")
    log_templates_file = os.path.join(output_dir, "log_templates.csv")

    if os.path.exists(log_structured_file):
        os.remove(log_structured_file)
    if os.path.exists(log_templates_file):
        os.remove(log_templates_file)
        
    with open(os.path.join(input_dir, log_file), 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        logs = list(reader)
    
    with open(log_templates_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["TemplateId", "LogTemplate"])
    
    with open(log_structured_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["LineId", "Timestamp", "TraceId", "SpanId", "ServiceName", "Content", "EventId"])
        
        for i, log_entry in enumerate(tqdm(logs, "Processing logs")):
            content = log_entry['Content']
            result = template_miner.add_log_message(content)
            
            if result["change_type"] in ["cluster_created", "cluster_template_changed"]:
                with open(log_templates_file, 'a', newline='', encoding='utf-8') as tmpl_file:
                    tmpl_writer = csv.writer(tmpl_file)
                    tmpl_writer.writerow([result['cluster_id'], result['template_mined']])
            
            writer.writerow([i + 1, log_entry['Timestamp'], log_entry['TraceId'], log_entry['SpanId'], log_entry['ServiceName'], content, result['cluster_id']])


def trace_sampling(log_file, output_file):
    """
    把logs切分成event sequence，基于相同的trace_id

    input: log_structured.csv
    output: trace_sequence.csv
    """
    print("Loading", log_file)
    df = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)

    data_dict = defaultdict(list)
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing logs"):
        data_dict[row['TraceId']].append(row["EventId"])

    data_dict = {k: v for k, v in data_dict.items()}

    data_df = pd.DataFrame(list(data_dict.items()), columns=['TraceId', 'EventSequence'])

    data_df.to_csv(output_file, index=False)
    print("Trace sampling done")


def generate_train_test(sequence_file, output_dir, n=None, ratio=0.3):
    """
    拆分training testing set

    input: trace_sequence.csv
    output: train, test_abnormal, test_normal
    """
    anomaly_label_file = os.path.join(output_dir, "anomaly_label.csv")
    if os.path.exists(anomaly_label_file):
        label_df = pd.read_csv(anomaly_label_file)
        label_dict = {row["TraceId"]: 1 if row["Label"] == "Anomaly" else 0 for _, row in label_df.iterrows()}
    else:
        label_dict = {}

    seq_df = pd.read_csv(sequence_file)
    if label_dict:
        seq_df["Label"] = seq_df["TraceId"].apply(lambda x: label_dict.get(x, 0))
    else:
        seq_df["Label"] = 0

    normal_seq = seq_df[seq_df["Label"] == 0]["EventSequence"]
    abnormal_seq = seq_df[seq_df["Label"] == 1]["EventSequence"] if label_dict else pd.Series()

    normal_seq = normal_seq.sample(frac=1, random_state=20)

    normal_len = len(normal_seq)
    train_len = n if n else int(normal_len * (1 - ratio))

    train = normal_seq.iloc[:train_len]
    test_normal = normal_seq.iloc[train_len:]
    test_abnormal = abnormal_seq

    df_to_file(train, os.path.join(output_dir, "train"))
    df_to_file(test_normal, os.path.join(output_dir, "test_normal"))
    df_to_file(test_abnormal, os.path.join(output_dir, "test_abnormal"))

    print(f"Training set size: {len(train)}")
    print(f"Test normal set size: {len(test_normal)}")
    print(f"Test abnormal set size: {len(test_abnormal)}")
    print("Generation of train and test data done.")


def df_to_file(df, file_name):
    with open(file_name, 'w') as f:
        for _, row in df.items():
            f.write(' '.join([str(ele) for ele in eval(row)]))
            f.write('\n')


def process(options):
    
    input_dir  = options["input_dir"]
    output_dir = options["output_dir"] + 'process/'
    log_file   = options["log_file"]

    log_structured_file = output_dir + "log_structured.csv"
    log_templates_file = output_dir + "log_templates.csv"
    log_sequence_file = output_dir + "log_sequence.csv"
    
    parser(input_dir, output_dir, log_file)
    trace_sampling(log_structured_file, log_sequence_file)
    generate_train_test(log_sequence_file, output_dir, ratio=0.2)

# if __name__ == "__main__":
    
#     # log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
    
#     parser(input_dir, output_dir, log_file)
    
#     trace_sampling(log_structured_file, log_sequence_file)
    
#     generate_train_test(log_sequence_file, output_dir, ratio=0.2)

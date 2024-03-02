import sys
sys.path.append("../")
sys.path.append("../../")

import os
import argparse
import schedule
import time
from datetime import datetime, timedelta
import pandas as pd
from option import options

import torch
from bert_pytorch.dataset import WordVocab
from bert_pytorch import Predictor, Trainer
from bert_pytorch.dataset.utils import seed_everything

from detector import Detector
from elasticsearch_client import ElasticsearchClient
from data_process_drain3 import process


if not os.path.exists(options['model_dir']):
    os.makedirs(options['model_dir'], exist_ok=True)

print("device", options["device"])
print("features logkey:{} time: {}\n".format(options["is_logkey"], options["is_time"]))
print("mask ratio", options["mask_ratio"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    es_client = ElasticsearchClient(options)
    
    subparsers = parser.add_subparsers()

    process_parser = subparsers.add_parser('process')
    process_parser.set_defaults(mode='process')

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(mode='train')

    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(mode='predict')
    predict_parser.add_argument("-m", "--mean", type=float, default=0)
    predict_parser.add_argument("-s", "--std", type=float, default=1)

    vocab_parser = subparsers.add_parser('vocab')
    vocab_parser.set_defaults(mode='vocab')
    vocab_parser.add_argument("-s", "--vocab_size", type=int, default=None)
    vocab_parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    vocab_parser.add_argument("-m", "--min_freq", type=int, default=1)

    detect_parser = subparsers.add_parser('detect')
    detect_parser.set_defaults(mode='detect')
    
    online_detect_parser = subparsers.add_parser('online_detect')
    online_detect_parser.set_defaults(mode='online_detect')

    deploy_parser = subparsers.add_parser('deploy')
    deploy_parser.set_defaults(mode='deploy')
    deploy_parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    deploy_parser.add_argument("-s", "--vocab_size", type=int, default=None)
    deploy_parser.add_argument("-m", "--min_freq", type=int, default=1)
    
    args = parser.parse_args()
    print("arguments", args)
    
    # Each round executes three actions: "query, detect, write back"
    def online_detect_job():
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=options["detect_period"])
        start_time = start_time.isoformat() + "Z"
        
        detector = Detector(options)
        log_df = es_client.query(last_timestamp=start_time) # query from es
        if not log_df.empty:
            result_df = detector.detect_seq(log_df) # detect logs
            es_client.write_result_to_es(result_df) # write the results back to es

    def train_job():
        print("\n========training start========\n")
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=options["retrain_period"])
        end_time = end_time.isoformat() + "Z"
        start_time = start_time.isoformat() + "Z"
        
        log_df = es_client.query(last_timestamp=start_time)
        log_df.to_csv(options["input_dir"] + "exported_data.csv", index=False)
        if not log_df.empty:
            process(options)
            with open(options["train_path"], "r", encoding=args.encoding) as f:
                texts = f.readlines()
            vocab = WordVocab(texts, max_size=args.vocab_size, min_freq=args.min_freq)
            vocab.save_vocab(options["vocab_path"])
            Trainer(options).train()
            es_client.update_last_train(end_time, datetime.utcnow().isoformat())
        print("\n========training finish========\n")
    
    if args.mode == 'process':
        process(options)
    
    elif args.mode == 'train':
        Trainer(options).train()

    elif args.mode == 'predict':
        Predictor(options).predict()

    elif args.mode == 'vocab':
        with open(options["train_path"], "r", encoding=args.encoding) as f:
            texts = f.readlines()
        vocab = WordVocab(texts, max_size=args.vocab_size, min_freq=args.min_freq)
        print("VOCAB SIZE:", len(vocab))
        print("save vocab in", options["vocab_path"])
        vocab.save_vocab(options["vocab_path"])

    elif args.mode == 'detect':
        print("\n========detecting start========\n")
        detector = Detector(options)
        log_df = pd.read_csv('../dataset/otlp/exported_data_1114-1120.csv')
        result_df = detector.detect_seq(log_df)
        result_df.to_csv(options["result_dir"] + "result_df_1114-1120.csv")
        print("\n========detecting finish========\n")
        

    elif args.mode == 'deploy':
        train_job()
        schedule.every(options["detect_period"]).minutes.at(":00").do(online_detect_job)
        schedule.every(options["retrain_period"]).days.at("00:00").do(train_job)
        while True:
            schedule.run_pending()
            time.sleep(1)


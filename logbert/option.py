import os
import torch
from elasticsearch import Elasticsearch

options = dict()
options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
options["input_dir"] = "../dataset/otlp_1209/export_data/"
options["output_dir"] = "../output/otlp_1209/"
options["log_file"]   = "exported_data_1209_noslash.csv"
options["model_dir"] = options["output_dir"] + "model/"
options["model_path"] = options["model_dir"] + "best_bert.pth"
options["result_dir"] = options["model_dir"] + "result/"
options["train_path"] = options["output_dir"] + "process/train"
options["vocab_path"] = options["model_dir"] + "vocab.pkl"

options["window_size"] = 128
options["adaptive_window"] = True
options["seq_len"] = 512
options["max_len"] = 512
options["min_len"] = 3
options["mask_ratio"] = 0.5
# sample ratio
options["train_ratio"] = 1
options["valid_ratio"] = 0.1
options["test_ratio"] = 1

# features
options["is_logkey"] = True
options["is_time"] = False

options["hypersphere_loss"] = True
options["hypersphere_loss_test"] = False

options["scale"] = None
options["scale_path"] = options["model_dir"] + "scale.pkl"

# model
options["hidden"] = 256
options["layers"] = 4
options["attn_heads"] = 4

options["epochs"] = 200
options["n_epochs_stop"] = 30
options["batch_size"] = 32

options["corpus_lines"] = None
options["on_memory"] = True
options["num_workers"] = 5
options["lr"] = 1e-3
options["adam_beta1"] = 0.9
options["adam_beta2"] = 0.999
options["adam_weight_decay"] = 0.00
options["with_cuda"]= True
options["cuda_devices"] = None
options["log_freq"] = None

# predict
options["num_candidates"] = 3
options["gaussian_mean"] = 0
options["gaussian_std"] = 1
options["seq_threshold"] = 0.2

# es
es_host = os.environ.get('ES_HOST', "log-monitor-es-http")
es_scheme = os.environ.get('ES_SCHEME', "http")
es_account = os.environ.get('ES_ACCOUNT', "elastic")
es_password = os.environ.get('ES_PASSWORD', "elastic")
service_name = os.environ.get('SERVICE_NAME', "service_api_ws1_smt-ecps-api")

es = Elasticsearch(
    [{'host': es_host, 'port': 9200, 'scheme': es_scheme}],
    basic_auth=(es_account, es_password),
    verify_certs=False
)
options["elasticsearch"] = es
options["service_name"] = service_name

options["read_index"] = 'traces-apm-default'
options["write_index"] = 'logbert_prediction'
options["state_index"] = 'logbert_state'

# schedule
options["detect_period"] = 5 # minites
options["retrain_period"] = 1 # days
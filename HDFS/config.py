import torch

options = {
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "output_dir": "../output/hdfs/",
    "model_dir": "../output/hdfs/bert/",
    "model_path": "../output/hdfs/bert/best_bert.pth",
    "train_vocab": "../output/hdfs/train",
    "vocab_path": "../output/hdfs/vocab.pkl",
    "window_size": 128,
    "adaptive_window": True,
    "seq_len": 512,
    "max_len": 512,
    "min_len": 10,
    "mask_ratio": 0.65,
    "train_ratio": 1,
    "valid_ratio": 0.1,
    "test_ratio": 1,
    "is_logkey": True,
    "is_time": False,
    "hypersphere_loss": True,
    "hypersphere_loss_test": False,
    "scale": None,
    "scale_path": "../output/hdfs/bert/scale.pkl",
    "hidden": 256,
    "layers": 4,
    "attn_heads": 4,
    "epochs": 200,
    "n_epochs_stop": 10,
    "batch_size": 32,
    "corpus_lines": None,
    "on_memory": True,
    "num_workers": 5,
    "lr": 0.001,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_weight_decay": 0.0,
    "with_cuda": True,
    "cuda_devices": None,
    "log_freq": None,
    "num_candidates": 6,
    "gaussian_mean": 0,
    "gaussian_std": 1
}
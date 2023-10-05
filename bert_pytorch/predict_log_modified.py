import torch
from bert_pytorch.predict_log import compute_anomaly
from bert_pytorch.predict_log import Predictor
from bert_pytorch.dataset import WordVocab
from bert_pytorch.dataset import LogDataset
from torch.utils.data import DataLoader
from bert_pytorch.dataset.sample import fixed_window
import numpy as np


class PredictorModified(Predictor):
    def __init__(self, options):
        super().__init__(options)
        # 加载模型
        self.model = torch.load(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        print('Model loaded from: {}'.format(self.model_path))
        self.vocab = WordVocab.load_vocab(self.vocab_path)
        self.scale = None
        self.error_dict = None
        
    def generate_test(self, output_dir, logs, window_size, adaptive_window, seq_len, min_len):
        """
        :return: log_seqs: num_samples x session(seq)_length, tim_seqs: num_samples x session_length
        """

        log_seq, tim_seq = fixed_window(logs, window_size,
                                                adaptive_window=adaptive_window,
                                                seq_len=seq_len, min_len=min_len)


        # sort seq_pairs by seq len
        log_seq = np.array(log_seq)
        tim_seq = np.array(tim_seq)

        # test_len = list(map(len, log_seq))
        # test_sort_index = np.argsort(-1 * np.array(test_len))

        # log_seq = log_seq[test_sort_index]
        # timtim_seq_seqs = tim_seq[test_sort_index]

        # print(f"{logs} size: {len(log_seq)}")
        # print(f"log_seq: {log_seq} tim_seq: {tim_seq}")
        return log_seq, tim_seq
    
    def helper(self, model, output_dir, log_seqs, vocab, scale, error_dict=None):
        total_results = []
        total_errors = []
        output_results = []
        total_dist = []
        output_cls = []
        log_string = " ".join(map(str, log_seqs))
        logkey_test, time_test = self.generate_test(output_dir, log_string, self.window_size, self.adaptive_window, self.seq_len, self.min_len)

        # use 1/10 test data
        if self.test_ratio != 1:
            num_test = len(logkey_test)
            rand_index = torch.randperm(num_test)
            rand_index = rand_index[:int(num_test * self.test_ratio)] if isinstance(self.test_ratio, float) else rand_index[:self.test_ratio]
            logkey_test = logkey_test[rand_index]


        seq_dataset = LogDataset(logkey_test, time_test, vocab, seq_len=self.seq_len,
                                 corpus_lines=self.corpus_lines, on_memory=self.on_memory, predict_mode=True, mask_ratio=self.mask_ratio)

        # use large batch size in test data
        data_loader = DataLoader(seq_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                 collate_fn=seq_dataset.collate_fn)

        for idx, data in enumerate(data_loader):
            data = {key: value.to(self.device) for key, value in data.items()}

            result = model(data["bert_input"], data["time_input"])

            # mask_lm_output, mask_tm_output: batch_size x session_size x vocab_size
            # cls_output: batch_size x hidden_size
            # bert_label, time_label: batch_size x session_size
            # in session, some logkeys are masked

            mask_lm_output, mask_tm_output = result["logkey_output"], result["time_output"]
            output_cls += result["cls_output"].tolist()

            # dist = torch.sum((result["cls_output"] - self.hyper_center) ** 2, dim=1)
            # when visualization no mask
            # continue

            # loop though each session in batch
            for i in range(len(data["bert_label"])):
                seq_results = {"num_error": 0,
                               "undetected_tokens": 0,
                               "masked_tokens": 0,
                               "total_logkey": torch.sum(data["bert_input"][i] > 0).item(),
                               "deepSVDD_label": 0
                               }

                mask_index = data["bert_label"][i] > 0
                num_masked = torch.sum(mask_index).tolist()
                seq_results["masked_tokens"] = num_masked

                if self.is_logkey:
                    num_undetected, output_seq = self.detect_logkey_anomaly(
                        mask_lm_output[i][mask_index], data["bert_label"][i][mask_index])
                    seq_results["undetected_tokens"] = num_undetected

                    output_results.append(output_seq)

                if self.hypersphere_loss_test:
                    # detect by deepSVDD distance
                    assert result["cls_output"][i].size() == self.center.size()
                    # dist = torch.sum((result["cls_fnn_output"][i] - self.center) ** 2)
                    dist = torch.sqrt(torch.sum((result["cls_output"][i] - self.center) ** 2))
                    total_dist.append(dist.item())

                    # user defined threshold for deepSVDD_label
                    seq_results["deepSVDD_label"] = int(dist.item() > self.radius)
                    #
                    # if dist > 0.25:
                    #     pass

                if idx < 10 or idx % 1000 == 0:
                    print(
                        " #time anomaly: {} # of undetected_tokens: {}, # of masked_tokens: {} , "
                        "# of total logkey {}, deepSVDD_label: {} \n".format(
                            seq_results["num_error"],
                            seq_results["undetected_tokens"],
                            seq_results["masked_tokens"],
                            seq_results["total_logkey"],
                            seq_results['deepSVDD_label']
                        )
                    )
                total_results.append(seq_results)
        return total_results, output_cls
    
    def predict_single_sequence(self, log_seqs):
        test_results, test_errors = self.helper(self.model, self.output_dir, log_seqs, self.vocab, self.scale, self.error_dict)

        seq_threshold = 0.4
        
        params = {
            "is_logkey": self.is_logkey, 
            "is_time": self.is_time, 
            "hypersphere_loss": self.hypersphere_loss,
            "hypersphere_loss_test": self.hypersphere_loss_test
        }
        
        is_anomaly = compute_anomaly(test_results, params, seq_threshold)
        
        return bool(is_anomaly)
import torch
from bert_pytorch.predict_log import Predictor
from bert_pytorch.dataset import WordVocab
from bert_pytorch.dataset import LogDataset
from torch.utils.data import DataLoader
from bert_pytorch.dataset.sample import fixed_window
import numpy as np

def compute_anomaly(results, params, seq_threshold=0.1):
    is_logkey = params["is_logkey"]
    is_time = params["is_time"]
    total_errors = 0
    is_anomalies = []
    for seq_res in results:
        # label pairs as anomaly when over half of masked tokens are undetected
        if (is_logkey and seq_res["undetected_tokens"] > seq_res["masked_tokens"] * seq_threshold) or \
                (is_time and seq_res["num_error"]> seq_res["masked_tokens"] * seq_threshold) or \
                (params["hypersphere_loss_test"] and seq_res["deepSVDD_label"]):
            total_errors += 1
            is_anomalies.append(True)
        else:
            is_anomalies.append(False)
    return is_anomalies

class PredictorBatch(Predictor):
    def __init__(self, options):
        super().__init__(options)
        self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.model.to(self.device)
        self.model.eval()
        print('Model loaded from: {}'.format(self.model_path))
        self.vocab = WordVocab.load_vocab(self.vocab_path)
        self.scale = None
        self.error_dict = None
        
    def generate_test(self, output_dir, log_seqs, window_size, adaptive_window, seq_len, min_len):
        """
        :return: log_seqs: num_samples x session(seq)_length, tim_seqs: num_samples x session_length
        """
        logkeys_test = []
        times_test = []
        
        for log_seq in log_seqs:
            log_string = " ".join(map(str, log_seq))
            logkey_test, time_test = fixed_window(log_string, window_size,
                                                adaptive_window=adaptive_window,
                                                seq_len=seq_len, min_len=min_len)
            logkeys_test += logkey_test
            times_test += time_test

        # sort seq_pairs by seq len
        logkeys_test = np.array(logkeys_test, dtype=object)
        times_test = np.array(times_test, dtype=object)

        # print(f"logkeys_test: {logkeys_test}")
        
        return logkeys_test, times_test
    
    def helper(self, model, output_dir, log_seqs, _log_ignore, vocab, scale, error_dict=None):
        total_results = []
        total_errors = []
        output_results = []
        total_dist = []
        output_cls = []
        # log_string = " ".join(map(str, log_seqs))
        # print(f"log_string: {log_seqs}")
        logkey_test, time_test = self.generate_test(output_dir, log_seqs, self.window_size, self.adaptive_window, self.seq_len, self.min_len)

        # use 1/10 test data
        # if self.test_ratio != 1:
        #     num_test = len(logkey_test)
        #     rand_index = torch.randperm(num_test)
        #     rand_index = rand_index[:int(num_test * self.test_ratio)] if isinstance(self.test_ratio, float) else rand_index[:self.test_ratio]
        #     logkey_test = logkey_test[rand_index]


        seq_dataset = LogDataset(logkey_test, time_test, vocab, seq_len=self.seq_len, log_ignore=_log_ignore,
                                 corpus_lines=self.corpus_lines, on_memory=self.on_memory, predict_mode=True, mask_ratio=self.mask_ratio)

        # print("seq_dataset: ", seq_dataset[0])
        
        # use large batch size in test data
        data_loader = DataLoader(seq_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                 collate_fn=seq_dataset.collate_fn)

        for idx, data in enumerate(data_loader):
            data = {key: value.to(self.device) for key, value in data.items()}
            # print(f"data: {data}")
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
        # print(f"total_results: {total_results}")
        # print(f"output_cls: {output_cls}")
        return total_results, output_cls
    
    def predict_sequence(self, log_seqs, logkeys_ignore=None):
        # print(f"logkeys_ignore: {logkeys_ignore}")
        # print(f"log_seqs: {log_seqs}")
        test_results, test_errors = self.helper(self.model, self.output_dir, log_seqs, logkeys_ignore, self.vocab, self.scale, self.error_dict)

        
        params = {
            "is_logkey": self.is_logkey, 
            "is_time": self.is_time, 
            "hypersphere_loss": self.hypersphere_loss,
            "hypersphere_loss_test": self.hypersphere_loss_test
        }
        
        is_anomalies = compute_anomaly(test_results, params, self.seq_threshold)
        
        # print(f"is_anomalies: {is_anomalies}")
        
        return is_anomalies, test_results
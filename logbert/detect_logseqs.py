# from detector import Detector
# import pandas as pd
# from elasticsearch import Elasticsearch
# from datetime import datetime
# import schedule
# import time
# import urllib3
# import warnings
# import argparse

# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# warnings.simplefilter(action='ignore', category=DeprecationWarning)

# def arg_parse():
#     parser = argparse.ArgumentParser(description='GLocalKD Arguments.')
#     parser.add_argument('--host', dest='host', default ='log-monitor.wneweb.com.tw', help='host for elasticsearch service.')
#     parser.add_argument('--port', dest='port', default ='32234', help='port for elasticsearch service.')
#     parser.add_argument('--account', dest='account', default ='elastic', help='Account for the elasticsearch service.')
#     parser.add_argument('--pwd', dest='pwd', default ='elastic', help='Password for the elasticsearch service.')
#     parser.add_argument('--index', dest='index', default ='apm-7.17.6-transaction-000001', help='Index to be searched.')
#     parser.add_argument('--modeldir', dest='modeldir', default ='model', help='Directory where model is located.')
#     parser.add_argument('--datadir', dest='datadir', default ='data', help='Directory where data is located.')

#     return parser.parse_args()

# args = arg_parse()

# es = Elasticsearch(
#     hosts=[{'host': args.host, 'port': args.port, 'scheme': 'https'}],
#     verify_certs=False,
#     http_auth=(args.account, args.pwd)
#     )

# def delete_index():
    
#     if es.indices.exists(index="analyzed_blks"):
#         es.indices.delete(index="analyzed_blks")
#     if es.indices.exists(index="model_prediction"):
#         es.indices.delete(index="model_prediction")

# def ensure_index_exists(es, index_name):

#     if not es.indices.exists(index=index_name):
#         es.indices.create(index=index_name)
#         print(f"Index {index_name} created.")
#     else:
#         print(f"Index {index_name} exists.")
    
# def get_analyzed_blks(es):

#     query = {
#         "query": {
#             "match_all": {}
#         }
#     }
#     result = es.search(index="analyzed_blks", body=query, size=1000)
#     analyzed_blks = [item['_source']['blk'] for item in result['hits']['hits']]
#     return analyzed_blks

# def get_new_logs(es, analyzed_blks):

#     query = {
#         "query": {
#             "bool": {
#                 "must_not": {
#                     "terms": {
#                         "blk": analyzed_blks
#                     }
#                 }
#             }
#         }
#     }
#     result = es.search(index="test_hdfs_data", body=query, size=1000)
#     new_logs = pd.DataFrame([item['_source'] for item in result['hits']['hits']])
#     return new_logs

# def group_logs_by_blk(log_df):

#     grouped_logs = dict(tuple(log_df.groupby('blk')))
#     return grouped_logs

# def write_result_to_es(es, blk, is_anomaly):

#     doc_data = {
#         "blk": blk,
#         "is_anomaly": is_anomaly,
#         "analyzed_time": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
#     }

#     es.index(index="model_prediction", body=doc_data)

# def mark_blk_as_analyzed(es, blk):

#     doc_data = {
#         "blk": blk,
#         "analyzed_time": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
#     }

#     es.index(index="analyzed_blks", body=doc_data)

# def detect_logseqs(options):
#     """
#     檢測所有之前沒檢測過的data
#     input:
#     output:
#     """
#     print("Job started...")

#     if not es.ping():
#         raise ValueError("Connection failed")

#     ensure_index_exists(es, "analyzed_blks")
#     ensure_index_exists(es, "model_prediction")

#     analyzed_blks = get_analyzed_blks(es)
#     print(f"Retrieved {len(analyzed_blks)} analyzed blks.")

#     new_logs = get_new_logs(es, analyzed_blks)
#     print(f"Retrieved {len(new_logs)} new logs.")

#     if new_logs.empty:
#         print("No new logs found.")
#         print("Job finished.\n")
#         return

#     grouped_logs = group_logs_by_blk(new_logs)
#     print(f"Grouped logs into {len(grouped_logs)} blks.")

#     detector = Detector(options)

#     for blk, logs_df in grouped_logs.items():
#         print(f"Analyzing blk: {blk}...")
#         is_anomaly = detector.detect(logs_df)
#         print(f"Anomaly detected: {is_anomaly}")
#         write_result_to_es(es, blk, is_anomaly)
#         mark_blk_as_analyzed(es, blk)
#         print(f"Results written and blk {blk} marked as analyzed.")
#     print("Job finished.\n")


# if __name__ == "__main__":
    
#     delete_index()

#     schedule.every(0.1).minutes.do(detect_logseqs)

#     while True:
#         schedule.run_pending()
#         time.sleep(1)
    
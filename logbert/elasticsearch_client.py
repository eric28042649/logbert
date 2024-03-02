from elasticsearch import Elasticsearch
import csv
import time
from datetime import datetime, timedelta
import urllib3
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ElasticsearchClient():
    def __init__(self, options):
        self.options = options
        self.es = self.options["elasticsearch"]
        
    def get_or_create_last_timestamp(self, index_name, doc_id, default_timestamp):
        try:
            result = self.es.get(index=index_name, id=doc_id)
            last_timestamp = result['_source']['last_timestamp']
        except Exception as e:
            last_timestamp = default_timestamp
            try:
                self.es.index(index=index_name, id=doc_id, body={'last_timestamp': last_timestamp})
                print(f"Created new timestamp document with timestamp: {last_timestamp}")
            except Exception as e:
                print(f"Error creating timestamp document: {e}")
                return None
        return last_timestamp

    def get_last_timestamp(self):
        try:
            result = self.es.get(index=self.options["state_index"], id="last_query")
            last_timestamp = result['_source']['last_timestamp']
            print(f"Retrieved last timestamp from Elasticsearch: {last_timestamp}")
            return last_timestamp
        except Exception as e:
            print(f"Error getting last timestamp: {e}")
            return None

    def update_last_query(self, timestamp):
        try:
            response = self.es.index(index=self.options["state_index"], id="last_query", body={"last_timestamp": timestamp})
            print(f"Updated last query to: {timestamp}, response: {response}")
        except Exception as e:
            print(f"Failed to update last query: {e}")
            
    def update_last_train(self, start_time, end_time):
        try:
            response = self.es.index(index=self.options["state_index"], id="last_train", body={"start_time": start_time, "end_time": end_time})
            print(f"Updated last train to: {start_time}, response: {response}")
        except Exception as e:
            print(f"Failed to update last train: {e}")

    def query(self, last_timestamp=None):
        if last_timestamp==None:
            last_timestamp = datetime.now() - timedelta(hours=24)
        query = {
            "size": 1000,
            "query": {
                "bool": {
                    "must_not": [
                        {"match": {"url.path": "/smt-ecps-api/metrics"}}
                    ],
                    "must": [
                        {"term": {"service.name": self.options["service_name"]}},
                    ],
                    "should": [
                        {"term": {"processor.event": "transaction"}},
                        {"term": {"processor.event": "span"}},
                        {"term": {"data_stream.type": "logs"}}
                    ],
                    "minimum_should_match": 1
                }
            },
            "sort": [
                {"@timestamp": {"order": "asc"}}
            ]
        }
        
        if last_timestamp:
            query['query']['bool']['filter'] = {
                "range": {
                    "@timestamp": {"gt": last_timestamp}
                }
            }

        all_docs = []
        page = self.es.search(index=self.options["read_index"], body=query, scroll='2m')
        scroll_id = page['_scroll_id']
        scroll_size = page['hits']['total']['value']
        
        while scroll_size > 0:
            print(f"Processing {scroll_size} more documents...")

            for hit in page['hits']['hits']:
                source = hit['_source']
                timestamp = source['@timestamp']
                trace_id = source.get('trace', {}).get('id', '')
                span_id = source.get('span', {}).get('id', '')
                service_name = source['service']['name']

                if not trace_id or not span_id:
                    continue

                processor_event = source.get('processor', {}).get('event', '')
                if processor_event == "transaction":
                    message = source.get('transaction', {}).get('name', '')
                elif processor_event == "span":
                    message = source.get('span', {}).get('name', '')
                elif source.get('data_stream', {}).get('type') == "logs":
                    message = source.get('message', '')
                else:
                    message = 'N/A'
                doc = {
                    "Timestamp": timestamp,
                    "TraceId": trace_id,
                    "SpanId": span_id,
                    "ServiceName": service_name,
                    "Content": message
                }
                all_docs.append(doc)

            page = self.es.scroll(scroll_id=scroll_id, scroll='2m')
            scroll_id = page['_scroll_id']
            scroll_size = len(page['hits']['hits'])

        self.es.clear_scroll(scroll_id=scroll_id)
        print("Data extraction completed.")

        if not page['hits']['hits']:
            last_timestamp = datetime.now().isoformat()
            self.update_last_query(last_timestamp)
            print(f"No new documents found.")
        else:
            last_timestamp = page['hits']['hits'][-1]['_source']['@timestamp']
            self.update_last_query(last_timestamp)

        return pd.DataFrame(all_docs)
    
    def write_result_to_es(self, result_df):
        index = self.options["write_index"]
        for _, row in result_df.iterrows():
            document = row.to_dict()
            document['timestamp'] = datetime.utcnow().isoformat()
            response = self.es.index(index=index, document=document)
            print(f"Document indexed in {index}: {response['_id']}")

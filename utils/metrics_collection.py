import pickle
import os
from collections import defaultdict

class metric_collection:
    def __init__(self, base_path):
        self.metrics = {'hits_10':{}, 'hits_3':{}, 'hits_1':{}, 'mrr':{}}
        self.base_path = base_path

    def update(self, time, mrr, hit_1, hit_3, hit_10):
        # import pdb; pdb.set_trace()
        self.metrics['hits_10'][time] = hit_10
        self.metrics['hits_3'][time] = hit_3
        self.metrics['hits_1'][time] = hit_1
        self.metrics['mrr'][time] = mrr

    def save(self):
        filename = os.path.join(self.base_path, "metrics-per-snapshot.pt")
        with open(filename, 'wb') as handle:
            pickle.dump(self.metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
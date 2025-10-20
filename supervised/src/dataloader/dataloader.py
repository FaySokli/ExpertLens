import torch
from indxr import Indxr
import random


class LoadTrainNQData(torch.utils.data.Dataset):
    def __init__(self, query_path, corpus_path, qrels, corpus_logits):
        self.query_path  = query_path
        self.corpus_path = corpus_path
        self.qrels = qrels
        self.corpus_logits = corpus_logits
        
        self.init_query()
        self.init_corpus()
        self.init_logits()
        
    def init_query(self):
        self.queries = Indxr(self.query_path, key_id='_id')
        
    def init_corpus(self):
        self.corpus = Indxr(self.corpus_path, key_id='_id')

    def init_logits(self):
        self.logits = Indxr(self.corpus_logits, key_id='_id')
        
    def __getitem__(self, idx):
        query = self.queries[idx]
        query_text = query['text'].lower()
        
        pos_ids = set(self.qrels[query['_id']])
        pos_id = str(random.choice(list(pos_ids)))
        pos_doc = self.corpus.get(pos_id)
        pos_doc_logits = self.logits.get(pos_id)
        
        return {
            'question': query_text,
            'pos_text': pos_doc.get('title', '').lower() + ' ' + pos_doc['text'].lower(),
            'pos_doc_logits': torch.tensor(pos_doc_logits['logits'], dtype=torch.float32)
        }
        
    def __len__(self):
        return len(self.queries)
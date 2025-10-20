import pandas as pd


#Computer Science Dataset Pre-processing
corpus_path = '../multi-domain/computer_science/collection.jsonl'
corpus = pd.read_json(corpus_path, lines=True)
corpus.rename(columns={'id': '_id'}, inplace=True)
corpus['_id'] = corpus['_id'].astype(str)
corpus.to_json(corpus_path, orient='records', lines=True)

q_path = '../multi-domain/computer_science/train/queries.jsonl'
queries_train = pd.read_json(q_path, lines=True)
queries_train.rename(columns={'id': '_id'}, inplace=True)
queries_train['_id'] = queries_train['_id'].astype(str)
queries_train.to_json(q_path, orient='records', lines=True)

query_ids = queries_train['_id']
rel_doc_ids = queries_train['rel_doc_ids']
rows = []
for query_id, doc_ids in zip(query_ids, rel_doc_ids):
    for doc_id in doc_ids:
        rows.append({'query-id': query_id, 'corpus-id': doc_id, 'score': 1})

qrels_df = pd.DataFrame(rows)
qrels_df.to_csv('../multi-domain/computer_science/train/qrels.tsv', sep='\t', index=False)

q_path = '../multi-domain/computer_science/test/queries.jsonl'
queries_test = pd.read_json(q_path, lines=True)
queries_test.rename(columns={'id': '_id'}, inplace=True)
queries_test['_id'] = queries_test['_id'].astype(str)
queries_test.to_json(q_path, orient='records', lines=True)


#Political Science Dataset Pre-processing
corpus_path = '../multi-domain/political_science/collection.jsonl'
corpus = pd.read_json(corpus_path, lines=True)
corpus.rename(columns={'id': '_id'}, inplace=True)
corpus['_id'] = corpus['_id'].astype(str)
corpus.to_json(corpus_path, orient='records', lines=True)

q_path = '../multi-domain/political_science/train/queries.jsonl'
queries_train = pd.read_json(q_path, lines=True)
queries_train.rename(columns={'id': '_id'}, inplace=True)
queries_train['_id'] = queries_train['_id'].astype(str)
queries_train.to_json(q_path, orient='records', lines=True)

query_ids = queries_train['_id']
rel_doc_ids = queries_train['rel_doc_ids']
rows = []
for query_id, doc_ids in zip(query_ids, rel_doc_ids):
    for doc_id in doc_ids:
        rows.append({'query-id': query_id, 'corpus-id': doc_id, 'score': 1})

qrels_df = pd.DataFrame(rows)
qrels_df.to_csv('../multi-domain/political_science/train/qrels.tsv', sep='\t', index=False)

q_path = '../multi-domain/political_science/test/queries.jsonl'
queries_test = pd.read_json(q_path, lines=True)
queries_test.rename(columns={'id': '_id'}, inplace=True)
queries_test['_id'] = queries_test['_id'].astype(str)
queries_test.to_json(q_path, orient='records', lines=True)  


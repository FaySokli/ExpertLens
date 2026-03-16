DATASET=computer-science
MODEL=tinybert

python3 1_train_new_moe.py model=$MODEL dataset=$DATASET testing=$DATASET model.adapters.use_adapters=True model.init.specialized_mode=densec3_top1
TESTING_DATASET=$DATASET
python3 2_create_embedding_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET model.adapters.use_adapters=True model.init.specialized_mode=densec3_top1
echo "$DATASET MoE"
python3 3_test_biencoder_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET model.adapters.use_adapters=True model.init.specialized_mode=densec3_top1
# Optional: add '+analysis.umap=true', '+analysis.deepview=true', '+analysis.text=true', '+analysis.corpus=true'
# Optional: '+analysis.n_sample=40', '+analysis.query_ids=["id1","id2"]'


DATASET=political-science
MODEL=contriever-base

python3 1_train_new_moe.py model=$MODEL dataset=$DATASET testing=$DATASET model.adapters.use_adapters=True model.init.specialized_mode=densec3_w
TESTING_DATASET=$DATASET
python3 2_create_embedding_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET model.adapters.use_adapters=True model.init.specialized_mode=densec3_w
echo "$DATASET MoE"
python3 3_test_biencoder_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET model.adapters.use_adapters=True model.init.specialized_mode=densec3_w
# Optional: add '+analysis.umap=true', '+analysis.deepview=true', '+analysis.text=true', '+analysis.corpus=true'
# Optional: '+analysis.n_sample=40', '+analysis.query_ids=["id1","id2"]'


DATASET=hotpotqa
MODEL=bert-base

python3 1_train_new_moe.py model=$MODEL dataset=$DATASET testing=$DATASET model.adapters.use_adapters=True model.init.specialized_mode=densec3_top1
TESTING_DATASET=$DATASET
python3 2_create_embedding_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET model.adapters.use_adapters=True model.init.specialized_mode=densec3_top1
echo "$DATASET MoE"
python3 3_test_biencoder_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET model.adapters.use_adapters=True model.init.specialized_mode=densec3_top1
# Optional: add '+analysis.umap=true', '+analysis.deepview=true', '+analysis.text=true', '+analysis.corpus=true'
# Optional: '+analysis.n_sample=40', '+analysis.query_ids=["id1","id2"]'


DATASET=nq-train
MODEL=colbert

python3 1_train_new_moe.py model=$MODEL dataset=$DATASET testing=$DATASET model.adapters.use_adapters=True model.init.specialized_mode=densec3_w
TESTING_DATASET=$DATASET
python3 2_create_embedding_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET model.adapters.use_adapters=True model.init.specialized_mode=densec3_w
echo "$DATASET MoE"
python3 3_test_biencoder_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET model.adapters.use_adapters=True model.init.specialized_mode=densec3_w
# Optional: add '+analysis.umap=true', '+analysis.deepview=true', '+analysis.text=true', '+analysis.corpus=true'
# Optional: '+analysis.n_sample=40', '+analysis.query_ids=["id1","id2"]'

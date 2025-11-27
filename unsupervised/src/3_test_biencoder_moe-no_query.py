import json
import logging
import os

import hydra
import torch
import tqdm
from indxr import Indxr
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer, AutoConfig
from indxr import Indxr

from model.models import MoEBiEncoder, DeepViewClassifier
from model.utils import seed_everything

from ranx import Run, Qrels, compare

import matplotlib
# matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.manifold import TSNE
import random
import numpy as np

# from dataloader.dataloader import LoadCorpus
from deepviewIR.DeepViewIR import DeepViewIR
from DeepView.deepview import DeepView
from DeepView.deepview.evaluate import leave_one_out_knn_dist_err


logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):


    os.makedirs(cfg.dataset.output_dir, exist_ok=True)
    os.makedirs(cfg.dataset.logs_dir, exist_ok=True)
    os.makedirs(cfg.dataset.model_dir, exist_ok=True)
    os.makedirs(cfg.dataset.runs_dir, exist_ok=True)

    logging_file = f"{cfg.model.init.doc_model.replace('/','_')}_testing_biencoder.log"
    logging.basicConfig(
        filename=os.path.join(cfg.dataset.logs_dir, logging_file),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    seed_everything(cfg.general.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.init.tokenizer)
    config = AutoConfig.from_pretrained(cfg.model.init.doc_model)
    config.num_experts = cfg.model.adapters.num_experts
    config.num_experts_to_use = cfg.model.adapters.num_experts_to_use
    config.adapter_residual = cfg.model.adapters.residual
    config.adapter_latent_size = cfg.model.adapters.latent_size
    config.adapter_non_linearity = cfg.model.adapters.non_linearity
    config.use_adapters = cfg.model.adapters.use_adapters
    doc_model = AutoModel.from_pretrained(cfg.model.init.doc_model)
    model = MoEBiEncoder(
        doc_model=doc_model,
        tokenizer=tokenizer,
        num_classes=cfg.model.adapters.num_experts,
        normalize=cfg.model.init.normalize,
        specialized_mode=cfg.model.init.specialized_mode,
        pooling_mode=cfg.model.init.aggregation_mode,
        use_adapters = cfg.model.adapters.use_adapters,
        device=cfg.model.init.device
    )

    if cfg.model.adapters.use_adapters:
        if cfg.model.init.specialized_mode == "variant_top1":
            model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-variant_top1.pt', weights_only=True))
            print(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-variant_top1.pt')
        elif cfg.model.init.specialized_mode == "variant_all":
            model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-variant_top1.pt',map_location=torch.device("cuda:0"), weights_only=True))
            print(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-variant_top1.pt')
        elif cfg.model.init.specialized_mode == "random":
            model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-random.pt', weights_only=True))
            print(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-random.pt')
    else:
        model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-ft.pt', weights_only=True))
        print(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-ft.pt')

    # Load the saved classifier and detect number of classes from it
    classifier_path = f'{cfg.dataset.output_dir}/dv_plots/deepview_cls.pt'
    state_dict = torch.load(classifier_path, map_location=cfg.model.init.device)
    # Detect num_classes from the saved model (check cls_5.bias shape)
    saved_num_classes = state_dict['cls_5.bias'].shape[0]
    print(f"Detected {saved_num_classes} classes from saved classifier")
    
    dv_cls = DeepViewClassifier(
        hidden_size=cfg.model.init.embedding_size,
        num_classes=saved_num_classes,
        device=cfg.model.init.device,
    )
    dv_cls.load_state_dict(state_dict)
    dv_cls = dv_cls.to(cfg.model.init.device)
    dv_cls.eval()



    # corpus = LoadCorpus(f"{cfg.dataset.data_dir}/corpus.jsonl")
    # print(corpus)

    ############################
    # Create directory for t-SNE and DeepView plots
    # qids = ["67316", "135802", "324585", "1051399", "1113256", "1127540", "1136962"]
    tsne_dir = os.path.join(cfg.dataset.output_dir, 'tsne_plots')
    os.makedirs(tsne_dir, exist_ok=True)

    dv_dir = os.path.join(cfg.dataset.output_dir, 'dv_plots')
    os.makedirs(dv_dir, exist_ok=True)

    # Load .npy embedding and label files



    np_data_dir = cfg.testing.embedding_dir
    prefix = "fullrank"

    # Load embeddings and expert_ids
    np_embedding_path = os.path.join(np_data_dir, f"doc_embeddings_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_{prefix}.npy")
    expert_ids_path = os.path.join(np_data_dir, f"expert_ids_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_{prefix}.npy")

    all_doc_embeddings_np = np.load(np_embedding_path)
    print("hereherhe")

    all_expert_ids = np.load(expert_ids_path)

    doc_sample = np.random.choice(len(all_doc_embeddings_np),1000,replace=False)
    X = all_doc_embeddings_np[doc_sample]
    # with torch.no_grad():
    #     probs = dv_cls(torch.tensor(X).to(cfg.model.init.device)).softmax(dim=1).cpu().numpy()
    y = all_expert_ids[doc_sample]

    def pred_wrapper(x):
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(cfg.model.init.device)
            pred = dv_cls(x).softmax(dim=1).cpu().numpy()
        return pred

    corpus = Indxr(f"{cfg.dataset.data_dir}/corpus.jsonl", key_id='_id')

    deepview_docs = np.array(corpus)[doc_sample]
    def see_text(sample, p, t):
        for s_id in sample:
            pos_doc = deepview_docs[s_id]
            text = pos_doc['text']
            print(text)


    # --- Deep View Parameters ----
    use_case = "nlp"
    classes = np.arange(cfg.model.adapters.num_experts)
    batch_size = cfg.training.batch_size
    max_samples = 1001  # including query
    data_shape = (cfg.model.init.embedding_size,)
    resolution = 100
    N = 10
    lam = 1
    cmap = 'tab10'
    metric = 'cosine'
    disc_dist = (
        False
        if lam == 1
        else True
    )

    # to make sure deepview.show is blocking,
    # disable interactive mode
    interactive = False
    my_title = "MOE Enhanced DRM - Deepview"

    print("here")
    deepview = DeepView(pred_wrapper, classes, max_samples, batch_size, data_shape,
                                            N, lam, resolution, cmap, interactive, my_title, metric=metric,
                                            disc_dist=disc_dist,use_selector=True,data_viz=see_text)


    deepview.add_samples(X, y)
    deepview.show()
    # ipdb.set_trace()

    # q_knn = leave_one_out_knn_dist_err(deepview.distances, deepview.y_pred)
    # print('Lambda: %.2f - Pred. Val. Q_kNN: %.3f' % (lam, q_knn))
    #
    # q_knn = leave_one_out_knn_dist_err(deepview.distances, deepview.y_true)
    # print('Lambda: %.2f - True Val. Q_kNN: %.3f' % (lam, q_knn))
    # ipdb.set_trace()
    # deepview.save_fig(os.path.join(dv_dir, f"deepview_query_{query_id}_experts{cfg.model.adapters.num_experts}.png"))

if __name__ == '__main__':
    main()

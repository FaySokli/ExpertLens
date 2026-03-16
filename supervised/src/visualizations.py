import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import torch
import umap

from deepview.DeepViewIR import DeepViewIR
from deepview.evaluate import leave_one_out_knn_dist_err, evaluate_umap, evaluate_inv_umap


def visualize_umap(query_embedding, top_doc_embeddings, top_doc_ids,
                   query_id, output_dir, experts_used, relevants,
                   use_adapters=True):

    all_embeddings = torch.cat([query_embedding, top_doc_embeddings], dim=0).cpu().numpy()
    relevant_set = set(relevants)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    embeddings_2d = reducer.fit_transform(all_embeddings)

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(
        embeddings_2d[0, 0],
        embeddings_2d[0, 1],
        c="black",
        marker="X",
        s=150,
        label="query",
        zorder=10,
    )

    color_map = plt.cm.tab10
    safe_colors = [color_map(i) for i in range(color_map.N) if i != 3]
    marker_list = ['o', '^', 's', 'D', 'v', 'P', '*', 'X', '<', '>']

    for i, (x, y) in enumerate(embeddings_2d[1:], start=1):
        doc_id = top_doc_ids[i - 1]
        is_relevant = doc_id in relevant_set

        if use_adapters:
            expert_id = experts_used[i - 1]
            color = safe_colors[expert_id % len(safe_colors)]
            marker = marker_list[expert_id % len(marker_list)]
        else:
            color = "blue"
            marker = "o"

        if is_relevant:
            ax.scatter(
                x, y,
                edgecolors="red",
                facecolors="none",
                s=120,
                linewidths=2,
                marker="o",
                label="relevant_docs" if i == 1 else "",
                zorder=9,
            )
        else:
            ax.scatter(
                x, y,
                c=[color],
                marker=marker,
                alpha=0.6,
                s=40,
            )

    handles = [
        mlines.Line2D([], [], color="black", marker="X",
                      linestyle="None", markersize=10, label="query"),
        mlines.Line2D([], [], color="red", marker="o",
                      markerfacecolor="none",
                      linestyle="None", markersize=10, label="relevant_docs"),
    ]

    if use_adapters:
        unique_experts = sorted(set(experts_used))
        for expert_id in unique_experts:
            handles.append(
                mlines.Line2D(
                    [], [],
                    color=safe_colors[expert_id % len(safe_colors)],
                    marker=marker_list[expert_id % len(marker_list)],
                    linestyle="None",
                    markersize=10,
                    label=f"Expert {expert_id + 1}",
                )
            )

    ax.legend(handles=handles, fontsize=20)
    ax.grid(True)
    plt.tight_layout()

    filename = (
        f"umap_query_{query_id}_experts.png"
        if use_adapters
        else f"umap_query_{query_id}_NO_experts.png"
    )
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=900)
    plt.close()
    print(f"Saved UMAP plot: {save_path}")


def visualize_deepview(
    query_embedding, top_doc_embeddings, relevant_indices,
    dv_cls, device, num_experts, batch_size, embedding_size,
    query_id, dataset_name, output_dir,
    true_expert_ids=None,
    lam=0.6, resolution=100, cmap='tab10', metric='cosine',
):
    all_embeddings = torch.cat([query_embedding, top_doc_embeddings], dim=0)

    with torch.no_grad():
        probs = dv_cls(all_embeddings.float().to(device)).softmax(dim=1).cpu().numpy()

    X = all_embeddings.detach().cpu().numpy()
    y = np.argmax(probs, axis=1)

    def _pred_wrapper(x):
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(device)
            return dv_cls(x).softmax(dim=1).cpu().numpy()

    classes = np.arange(num_experts)
    max_samples = len(X)
    data_shape = (embedding_size,)
    disc_dist = lam != 1
    N = 10

    deepview = DeepViewIR(
        _pred_wrapper, classes, max_samples, batch_size, data_shape,
        N, lam, resolution, cmap, False, "MOE Enhanced DRM - Deepview",
        metric=metric, disc_dist=disc_dist, relevant_docs=relevant_indices,
    )

    deepview.add_samples(X, y)
    deepview.show()

    fig = plt.gcf()
    save_path = os.path.join(
        output_dir,
        f"deepviewIR_lam{lam}_query_{query_id}_experts{num_experts}_{dataset_name}.png",
    )
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved DeepView plot: {save_path}")

    q_knn_pred = leave_one_out_knn_dist_err(deepview.distances, deepview.y_pred)
    q_knn_true = leave_one_out_knn_dist_err(deepview.distances, deepview.y_true)
    print(f"Lambda: {lam:.2f} - Pred. Val. Q_kNN: {q_knn_pred:.3f}")
    print(f"Lambda: {lam:.2f} - True Val. Q_kNN: {q_knn_true:.3f}")

    if true_expert_ids is not None:
        deepview.y_true = np.array(true_expert_ids)

    umap_results = evaluate_umap(
        deepview, return_values=True, compare_unsup=True, X=X, Y=deepview.y_true
    )
    print(f"evaluate_umap results: {umap_results}")

    inv_train_acc, inv_test_acc = evaluate_inv_umap(deepview, X, deepview.y_true)
    print(f"Inv UMAP - Train acc: {inv_train_acc:.1f}%  Test acc: {inv_test_acc:.1f}%")

    return q_knn_pred, q_knn_true, umap_results, inv_train_acc, inv_test_acc

import torch


def two_way_identification(true_labels, predict_prob, label_mapping=None, k=500):
    # true_labels: [B], predict_prob: [B, C]
    # label_mapping: {original_label: new_label}
    # k: number of negative classes to sample
    if predict_prob.dim() == 3 and predict_prob.size(-1) == 1:
        predict_prob = predict_prob.squeeze(-1)

    B, C = predict_prob.shape
    device = predict_prob.device
    # true_labels = torch.tensor([label_mapping[label.item()] for label in true_labels], device=device)
    true_prob = predict_prob.gather(dim=1, index=true_labels.view(-1, 1))  # [B, 1]

    # 采样负类索引: r ∈ [0, C-2], 形状 [B, k]
    r = torch.randint(low=0, high=C - 1, size=(B, k), device=device)
    y = true_labels.view(-1, 1)  # [B,1]
    neg_idx = r + (r >= y)  # [B,k]，将 r 映射到 [0, C-1] \ {y}
    # 取出负类得分：先扩展 predict_prob -> [B, C]；用 gather 取 [B, k]
    probs_neg = predict_prob.gather(dim=1, index=neg_idx)  # [B, k]

    # 比较：probs_true -> [B,1] 与 probs_neg -> [B,k] 对比
    wins = (true_prob > probs_neg).float()  # [B, k]，真类更大记 1
    per_sample_score = (wins.sum(dim=1) > (k // 2)).float()  # [B]
    return per_sample_score

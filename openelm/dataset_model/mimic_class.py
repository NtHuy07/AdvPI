import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from modules.models.factory import ModelFactory


model = ModelFactory.create_model(
    model_type='biomedclip',
    variant='base',
    pretrained=True,
)


medical_data_labeled_path = "datasets/mimic/data.csv"
img_dir = "datasets/mimic/images"

medical_samples = pd.read_csv(medical_data_labeled_path)

queries = []
for _, row in medical_samples.iterrows():
    queries.append((row['findings'], row['label']))

# Get good initialization queries (i.e. the performance of the model on these is initially good, so we can attack them)
init_id = np.array([1, 12, 77, 81, 100, 126, 147, 175, 201, 230, 256, 257, 258, 268, 275, 279, 287, 295, 312, 317, 324, 393, 419, 421, 430, 475, 519, 523, 552, 662, 685, 687, 699, 722, 772, 781, 785, 786, 787, 802, 856, 868, 890, 955, 962, 967, 979, 983, 989, 994, 1007, 1013, 1017, 1031, 1036, 1041, 1054, 1055, 1060, 1062, 1071, 1073, 1074, 1078, 1079, 1081, 1086, 1088, 1093, 1097, 1101, 1113, 1125, 1130, 1132, 1138, 1140, 1150, 1163, 1166, 1170, 1186, 1192, 1194, 1195, 1197, 1200, 1201, 1202, 1205, 1213, 1225, 1226, 1234, 1245, 1248, 1253, 1280, 1282, 1290, 1295, 1300, 1301, 1305, 1306, 1317, 1320, 1322, 1336, 1347, 1352, 1374, 1385, 1390, 1397, 1401, 1409, 1412, 1427, 1428, 1452, 1456, 1466, 1477, 1481, 1482, 1488, 1489, 1490, 1497, 1501, 1506, 1507, 1510, 1511, 1516, 1521, 1525, 1532, 1537, 1541, 1545, 1555, 1563, 1570, 1586, 1587, 1591, 1592, 1594, 1608, 1613, 1623, 1624, 1635, 1636, 1638, 1643, 1644, 1654, 1661, 1663, 1666, 1687, 1695, 1703, 1709, 1717, 1731, 1733, 1734, 1739, 1749, 1751, 1756, 1762, 1767, 1769, 1778, 1781, 1782, 1787, 1790, 1795, 1797, 1803, 1808, 1818, 1835, 1836, 1842, 1855, 1872, 1873, 1887, 1898, 1899, 1927, 1943, 1957, 1958, 1964, 1967, 1973, 1975, 1981, 1985, 1986])
init_queries = np.array(queries)[init_id]

print(len(queries))


_toTensor = transforms.ToTensor()


corpus_embeddings = []
corpus_labels = []
corpus_images = []
for _, row in tqdm(medical_samples.iterrows()):
    img_name = row['image_filename']
    img = Image.open(os.path.join(img_dir, img_name))
    corpus_images.append(img.convert("RGB"))
    img_attack = img.convert("RGB")
    label_id = row['label']
    img_attack_tensor = _toTensor(img_attack).unsqueeze(0).cuda()  # [1, C, H, W]
    with torch.no_grad():
        img_feats = model.encode_pretransform_image(img_attack_tensor)  # [1, D]

    corpus_embeddings.append(img_feats.cpu())
    corpus_labels.append(label_id)

corpus_embeddings = torch.cat(corpus_embeddings, dim=0).cuda()   # [N, D]
corpus_labels = torch.tensor(corpus_labels).cuda()               # [N]
print(corpus_embeddings.shape, corpus_labels.shape)



def get_rank(queries):
    k_list = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    NUM_RELEVANT = 1000 # each class has 1000 instances
    hit_counts = {k: 0 for k in k_list}
    precision_sums = {k: 0.0 for k in k_list}
    mrr_sum = 0.0

    num_queries = len(queries)

    for query, label_id in queries:
        label_id = torch.tensor(int(float(label_id)))

        with torch.no_grad():
            text_feat = model.encode_text(query)      # [1, D]
            sims = text_feat @ corpus_embeddings.T    # [1, N]
            sims = sims.squeeze(0)                    # [N]

        sorted_indices = torch.argsort(sims, descending=True)
        sorted_labels = corpus_labels[sorted_indices]

        # ================= MRR =================
        match_positions = (sorted_labels == label_id).nonzero(as_tuple=True)[0]
        if len(match_positions) > 0:
            rank = match_positions[0].item() + 1
            mrr_sum += 1.0 / rank

        # ================= Hit@k & Precision@k =================
        for k in k_list:
            topk_labels = sorted_labels[:k]

            # Hit@k
            if (topk_labels == label_id).any():
                hit_counts[k] += 1

            # Precision@k
            num_rel_in_topk = (topk_labels == label_id).sum().item()
            precision_sums[k] += num_rel_in_topk / k

    metrics = {}

    for k in k_list:
        metrics[f"Hit@{k}"] = hit_counts[k] / num_queries
        metrics[f"Precision@{k}"] = precision_sums[k] / num_queries

    metrics["MRR"] = mrr_sum / num_queries

    return rank, metrics
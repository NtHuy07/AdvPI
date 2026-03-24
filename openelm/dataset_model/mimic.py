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
idx = 0
for _, row in medical_samples.iterrows():
    queries.append((row['findings'], idx))
    idx += 1

# Get good initialization queries (i.e. the performance of the model on these is initially good, so we can attack them)
init_id = np.array([317,  324,  326,  347,  350,  352,  363,  378,  382,  384,  388, 390,  392,  413,  449,  461,  475,  497,  529,  530,  535,  536, 539,  549,  557,  558,  574,  602,  615,  622,  631,  637,  642, 670,  674,  679,  696,  702,  721,  722,  739,  741,  744,  758, 760,  784,  792,  799,  801,  803,  831,  835,  836,  880,  912, 916,  920,  941,  952,  957,  963,  971,  986,  991, 1002, 1018, 1023, 1029, 1077, 1104, 1113, 1115, 1212, 1272, 1335, 1360, 1373, 1438, 1486, 1487, 1520, 1540, 1546, 1554, 1561, 1587, 1593, 1623, 1632, 1659, 1704, 1785, 1830, 1855, 1861, 1880, 1928, 1929, 1969, 1988])
init_queries = np.array(queries)[init_id]

print(len(queries))

_toTensor = transforms.ToTensor()


corpus_embeddings = []
corpus_images = []
for _, row in tqdm(medical_samples.iterrows()):
    img_name = row['image_filename']
    img = Image.open(os.path.join(img_dir, img_name))
    corpus_images.append(img.convert("RGB"))
    img_attack = img.convert("RGB")
    img_attack_tensor = _toTensor(img_attack).unsqueeze(0).cuda()  # [1, C, H, W]
    with torch.no_grad():
        img_feats = model.encode_pretransform_image(img_attack_tensor)  # [1, D]

    corpus_embeddings.append(img_feats.cpu())

corpus_embeddings = torch.cat(corpus_embeddings, dim=0).cuda()   # [N, D]


def get_rank(queries):
    k_list = [1, 5, 10, 20, 50, 100, 200]

    hit_counts = {k: 0 for k in k_list}
    mrr_sum = 0.0

    num_queries = len(queries)

    for query, idx in queries:
        idx = torch.tensor(int(idx))

        with torch.no_grad():
            text_feat = model.encode_text(query)          # [1, D]
            sims = (text_feat @ corpus_embeddings.T).squeeze(0)  # [N]

        sorted_indices = torch.argsort(sims, descending=True)

        # ===== MRR =====
        match_pos = (sorted_indices == idx).nonzero(as_tuple=True)[0]
        if len(match_pos) > 0:
            rank = match_pos[0].item() + 1
            mrr_sum += 1.0 / rank

        # ===== Hit@K =====
        
        for k in k_list:
            if idx in sorted_indices[:k]:
                hit_counts[k] += 1

    metrics = {}

    for k in k_list:
        metrics[f"Hit@{k}"] = hit_counts[k] / num_queries

    metrics["MRR"] = mrr_sum / num_queries

    return rank, metrics
import os
import yaml
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

from modules.dataset.factory import DatasetFactory
from modules.models.factory import ModelFactory
from modules.utils.constants import DATA_ROOT


dataset = DatasetFactory.create_dataset(
    dataset_name='entrep',
    model_type='medclip',
    data_root=DATA_ROOT,
    transform=None
)

config_path = "configs/entrep_contrastive.yaml"
pretrained = "models/entrep_base_multi_modal_ssl_finetuning.pt"
data_path = "datasets/entrep/data.csv"
img_dir = "datasets/entrep/images"


with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
model_config = config.get('model', {})

model = ModelFactory.create_model(
    model_type='entrep',
    variant='base',
    checkpoint=None,
    pretrained=False,
    **{k: v for k, v in model_config.items() if k != 'model_type' and k != "pretrained" and k != "checkpoint"}
)

checkpoint = torch.load(pretrained)['model_state_dict']
not_matching_key = model.load_state_dict(checkpoint, strict=False)
print("Incabable key: ", not_matching_key)
model.eval()

_toTensor = transforms.ToTensor()


t2class_id = {
    'vc-open': 0,
    'vc-closed': 0,
    'nose-right': 1,
    'nose-left': 1,
    'ear-left': 2,
    'ear-right': 2,
    'throat': 3
}
class2class = {
    'vc-open': 'voice-throat',
    'vc-closed': 'voice-throat',
    'nose-right': 'nose',
    'nose-left': 'nose',
    'ear-left': 'ear',
    'ear-right': 'ear' ,
    'throat': 'throat'
}

df = pd.read_csv(data_path)
queries = []
img_names = []
for id, row in df.iterrows():
    img_name = row["path"].replace("image", "Image")
    img_names.append(img_name)
    caption = row["caption"]

    queries.append((caption, id))

# Get good initialization queries (i.e. the performance of the model on these is initially good, so we can attack them)
init_id = np.array([7, 12, 13, 26, 28, 29, 32, 38, 39, 43, 51, 54, 57, 59, 64, 69, 76, 79, 80, 83, 86, 88, 92, 93, 98, 99, 101, 114, 115, 117, 121, 123, 135, 149, 152, 156, 159, 163, 170, 174, 178, 180, 181, 182, 189, 202, 204, 209, 210, 212, 219, 224, 226, 227, 228, 234, 237, 245, 246, 250, 252, 266, 268, 269, 272, 274, 283, 285, 288, 291, 304, 315, 318, 320, 327, 334, 338, 344, 345, 346, 349, 351, 353, 356, 361, 363, 365, 370, 373, 377, 380, 382, 383, 390, 391, 392, 398, 399, 400, 402, 403, 423, 424, 425, 427, 428, 431, 435, 443, 445, 450, 452, 454, 464, 479, 480, 488, 505, 508, 512, 514, 515, 516, 519, 520, 522, 524, 525, 526, 531, 532, 533, 536, 538, 540, 542, 543, 544, 547, 548, 550, 554, 557, 558, 561, 564])
init_queries = np.array(queries)[init_id]

print("Len samples query: ", len(queries))


corpus_embeddings = []
corpus_images = []
for i, img_name in enumerate(tqdm(img_names)):
    img = Image.open(os.path.join(img_dir, img_name))
    corpus_images.append(img.convert("RGB"))
    img_attack = img.convert("RGB")
    img_attack_tensor = _toTensor(img_attack).unsqueeze(0).cuda()  # [1, C, H, W]
    with torch.no_grad():
        img_feats = model.encode_pretransform_image(img_attack_tensor)  # [1, D]

    corpus_embeddings.append(img_feats.cpu())

corpus_embeddings = torch.cat(corpus_embeddings, dim=0).cuda()   # [N, D]

print(corpus_embeddings.shape)

def get_rank(queries):

    k_list = [1, 5, 10, 20, 50, 100, 200, 500]

    hit_counts = {k: 0 for k in k_list}
    mrr_sum = 0.0

    num_queries = len(queries)

    for query, idx in queries:

        with torch.no_grad():
            text_feat = model.encode_text(query)          # [1, D]
            sims = (text_feat @ corpus_embeddings.T).squeeze(0)  # [N]

        sorted_indices = torch.argsort(sims, descending=True)

        # ===== MRR =====
        idx = torch.tensor(int(idx))
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



def get_similarity(attack_features, orig_features):

    sim = F.cosine_similarity(orig_features.cpu(), attack_features, dim=-1)

    return float(sim.detach().numpy()[0])


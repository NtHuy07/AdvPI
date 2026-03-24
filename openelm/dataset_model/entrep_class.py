import json
import yaml
import re
import torch
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from modules.models.factory import ModelFactory


config_path = "configs/entrep_contrastive.yaml"
pretrained = "models/entrep_base_multi_modal_ssl_finetuning.pt"
caption_path = "datasets/entrep/data.json"
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

with open(caption_path, "r", encoding="utf-8") as f:
    caption_data = json.load(f)

queries = []
img_infos = []
num_samples = np.zeros(4)
for sample in caption_data:

    desc = sample["DescriptionEN"]
    class_name = sample["Classification"]
    class_name_abstrast = class2class[class_name]
    class_id = t2class_id[class_name]
    num_samples[class_id] += 1
    img_name = sample['Path'].replace("image", "Image")
    img_infos.append((img_name, class_id))
    sentences = [s.strip() for s in re.split(r'[\r\n]+', desc) if s.strip()]
    sentences = [f"a photo of {class_name_abstrast}. " + s for s in sentences]
    for sent in sentences:
        queries.append((sent, class_id))

# Get good initialization queries (i.e. the performance of the model on these is initially good, so we can attack them)
init_id = np.array([1, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 56, 58, 61, 64, 66, 67, 69, 70, 71, 72, 74, 75, 78, 79, 80, 81, 82, 85, 86, 87, 88, 89, 90, 91, 92, 93, 96, 97, 99, 100, 101, 102, 103, 104, 106, 107, 109, 112, 115, 116, 118, 119, 121, 122, 123, 124, 126, 127, 128, 129, 130, 131, 132, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 146, 148, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 174, 175, 180, 181, 182, 183, 185, 186, 188, 190, 191, 193, 194, 195, 198, 199, 201, 204, 205, 206, 208, 209, 212, 214, 215, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 239, 240, 241, 242, 243, 246, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 270, 273, 275, 277, 278, 279, 280, 281, 282, 283, 284, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 306, 307, 308, 309, 311, 312, 313, 316, 318, 321, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 351, 352, 356, 363, 364, 365, 369, 370, 371, 372, 374, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 390, 391, 392, 393, 394, 397, 400, 401, 404, 406, 407, 408, 410, 411, 412, 415, 416, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 431, 432, 433, 434, 435, 436, 437, 439, 440, 443, 445, 446, 447, 448, 449, 451, 452, 453, 455, 457, 458, 459, 460, 461, 462, 465, 466, 468, 469, 470, 471, 472, 473, 476, 479, 480, 481, 482, 483, 486, 488, 490, 493, 494, 496, 497, 498, 499, 501, 502, 503, 504, 505, 506, 507, 508, 510, 511, 513, 514, 515, 516, 517, 518, 520, 521, 522, 525, 526, 527, 529, 531, 532, 533, 534, 536, 540, 543, 544, 545, 546, 547, 549, 550, 552, 555, 558, 559, 560, 564, 565, 567, 572, 573, 574, 576, 579, 580, 583, 584, 585, 586, 587, 588, 590, 591, 595, 597, 598, 599, 601, 602, 609, 612, 617, 620, 624, 625, 626, 627, 628, 630, 631, 634, 636, 638, 639, 640, 641, 642, 643, 644, 645, 647, 648, 649, 650, 651, 652, 654, 657, 658, 659, 660, 661, 664, 665, 666, 667, 668, 670, 671, 672, 673, 674, 675, 678, 681, 682, 683, 686, 688, 689, 692, 694, 697, 698, 699, 700, 701, 702, 703, 704, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 722, 723, 724, 726, 727, 728, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 741, 743, 744, 748, 749, 750, 751, 753, 754, 755, 756, 757, 758, 759, 760, 762, 763, 764, 765, 766, 767, 770, 771, 772, 775, 779, 780, 781, 782, 784, 785, 786, 787, 788, 789, 790, 792, 800, 801, 802, 806, 807, 808, 809, 810, 811, 812, 813, 816, 817, 819, 822, 823, 825, 828, 832, 833, 834, 835, 836, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 849, 850, 851, 853, 854, 857, 858, 859, 863, 864, 865, 868, 871, 872, 873, 875, 877, 880, 881, 882, 883, 884, 885, 886, 888, 896, 898, 899, 900])
init_queries = np.array(queries)[init_id]

print("Len samples query: ", len(queries))


corpus_embeddings = []
corpus_labels = []
corpus_images = []
for img_name, label_id in tqdm(img_infos):
    img_path = os.path.join(img_dir, img_name)
    img = Image.open(img_path)
    corpus_images.append(img)
    img_attack = img.convert("RGB")
    img_attack_tensor = _toTensor(img_attack).unsqueeze(0).cuda()  # [1, C, H, W]
    with torch.no_grad():
        img_feats = model.encode_pretransform_image(img_attack_tensor)  # [1, D]

    corpus_embeddings.append(img_feats.cpu())
    corpus_labels.append(label_id)

corpus_embeddings = torch.cat(corpus_embeddings, dim=0).cuda()   # [N, D]
corpus_labels = torch.tensor(corpus_labels).cuda()               # [N]
print(corpus_embeddings.shape, corpus_labels.shape)


def get_rank(queries):
    k_list = [1, 5, 10, 20, 50, 100, 200, 500]

    hit_counts = {k: 0 for k in k_list}
    precision_sums = {k: 0.0 for k in k_list} 
    mrr_sum = 0.0

    num_queries = len(queries)

    for query, label_id in queries:
        label_id = torch.tensor(int(label_id))

        with torch.no_grad():
            text_feat = model.encode_text(query)      # [1, D]
            sims = text_feat @ corpus_embeddings.T    # [1, N]
            sims = sims.squeeze(0)                    # [N]

        sorted_indices = torch.argsort(sims, descending=True)
        sorted_labels = corpus_labels[sorted_indices]

        # MRR 
        match_positions = (sorted_labels == label_id).nonzero(as_tuple=True)[0]
        if len(match_positions) > 0:
            rank = match_positions[0].item() + 1     
            mrr_sum += 1.0 / rank

        # Hit@k + "Precision@k"
        for k in k_list:
            topk_labels = sorted_labels[:k]

            if (topk_labels == label_id).any():
                hit_counts[k] += 1

            num_rel_in_topk = (topk_labels == label_id).sum().item()
            precision_sums[k] += num_rel_in_topk / k


    metrics = {}

    for k in k_list:
        hit_k = hit_counts[k] / num_queries
        rec_k = precision_sums[k] / num_queries

        metrics[f"Hit@{k}"] = hit_k
        metrics[f"Precision@{k}"] = rec_k  

    metrics["MRR"] = mrr_sum / num_queries

    return rank, metrics


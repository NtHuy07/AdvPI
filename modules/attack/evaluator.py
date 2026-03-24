import torch
import torch.nn as nn
from typing import List
from .util import pil_to_tensor, tensor_to_pillow, project_delta
from time import time
import math

class DCTDecoder:
    def __init__(self, f_ratio, device='cuda', dtype=torch.float32):
        self.f_ratio = f_ratio
        self.device = device
        self.dtype = dtype
    
    @torch.no_grad()
    def __call__(self, coeffs, W, H):   # coeffs: [pop, C, f, f]
        f = int(self.f_ratio * H)
        u = torch.arange(H, device=self.device, dtype=self.dtype).view(-1,1)
        v = torch.arange(W, device=self.device, dtype=self.dtype).view(-1,1)
        kH = torch.arange(f, device=self.device, dtype=self.dtype).view(1,-1)
        kW = torch.arange(f, device=self.device, dtype=self.dtype).view(1,-1)
        U = torch.cos(math.pi * (u + 0.5) * kH / H)   # [H,f]
        V = torch.cos(math.pi * (v + 0.5) * kW / W)   # [W,f]
        if coeffs.ndim == 3: coeffs = coeffs.unsqueeze(0)
        T = torch.einsum('hi,pcij->pchj', U, coeffs)   # [pop,C,H,f]
        X = torch.einsum('pchj,wj->pchw', T, V)      # [pop,C,H,W]
        return X

class EvaluatePerturbation:
    def __init__(
        self,
        model: nn.Module,
        class_prompts: List[str], # (NUM_CLASSES x D)
        mode: str="post_transform", # mode for transform
        decoder: DCTDecoder=None,
        eps: float=0.03,
        norm: str='linf'
    ):
        self.model = model
        self.decoder = decoder
        self.class_text_feats = self.extract_centroid_vector(class_prompts)
        self.mode = mode
        self.eps = eps
        self.norm = norm
        
    def set_data(self, image, clean_pred_id):
        self.img = image
        self.img_tensor = pil_to_tensor([image]).cuda()
        if self.decoder:
            _, C, img_W, img_H = self.img_tensor.shape
            self.img_W, self.img_H = img_W, img_H
            self.lq_shape = (1, 3, int(self.decoder.f_ratio * self.img_W), int(self.decoder.f_ratio * self.img_W))
        
        self.clean_pred_id = clean_pred_id
        
    
    @torch.no_grad() 
    def extract_centroid_vector(self, class_prompts): 
        class_features = [] 
        for class_name, item in class_prompts.items(): 
            text_feats = self.model.encode_text(item) 
            mean_feats = text_feats.mean(dim=0)
            class_features.append(mean_feats) 
            
        class_features = torch.stack(class_features) # NUM_ClASS x D 
        return class_features
            
    
    @torch.no_grad()
    def cal_l2(self, perturbations: torch.Tensor) -> torch.Tensor:
        return perturbations.view(perturbations.size(0), -1).norm(p=2, dim=1)
    
    @torch.no_grad()
    def evaluate_blackbox(self, perturbations: torch.Tensor):
        perturbations_ = perturbations.clone()
        if self.decoder:
            perturbations_ = self.decoder(perturbations, self.img_W, self.img_W)
            perturbations_ = project_delta(perturbations_, self.eps, self.norm)        
        adv_imgs = self.img_tensor + perturbations_
        adv_imgs = torch.clamp(adv_imgs, 0, 1)
                
        if self.mode == "post_transform":
            adv_feats = self.model.encode_posttransform_image(adv_imgs)  # (B, D)
        
        elif self.mode == "pre_transform":
            adv_feats = self.model.encode_pretransform_image(adv_imgs)  # (B, D)
        
        sims = adv_feats @ self.class_text_feats.T     # (B, NUM_CLASSES)
        # Correct class similarity
        correct_sim = sims[:, self.clean_pred_id].unsqueeze(-1)

        # Max of other classes
        mask = torch.ones_like(sims, dtype=bool)
        mask[:, self.clean_pred_id] = False
        other_max_sim = sims[mask].view(sims.size(0), -1).max(dim=1, keepdim=True).values  # (B, 1)
        margin = correct_sim - other_max_sim
        # margin = correct_sim - other_max_sim + 0.05

        
        # l2
        l2 = self.cal_l2(perturbations_)
        return margin, l2

    def evaluate_whitebox(self, perturbations: torch.Tensor):
        perturbations_ = perturbations.clone()
        if self.decoder:
            perturbations_ = self.decoder(perturbations, self.img_W, self.img_W)
            perturbations_ = project_delta(perturbations_, self.eps, self.norm)        
        adv_imgs = self.img_tensor + perturbations_
        adv_imgs = torch.clamp(adv_imgs, 0, 1)
                
        if self.mode == "post_transform":
            adv_feats = self.model.encode_posttransform_image(adv_imgs)  # (B, D)
        
        elif self.mode == "pre_transform":
            adv_feats = self.model.encode_pretransform_image(adv_imgs)  # (B, D)
        
        sims = adv_feats @ self.class_text_feats.T     # (B, NUM_CLASSES)
        # Correct class similarity
        correct_sim = sims[:, self.clean_pred_id].unsqueeze(-1)

        # Max of other classes
        mask = torch.ones_like(sims, dtype=bool)
        mask[:, self.clean_pred_id] = False
        other_max_sim = sims[mask].view(sims.size(0), -1).max(dim=1, keepdim=True).values  # (B, 1)
        margin = correct_sim - other_max_sim
        # margin = correct_sim - other_max_sim + 0.05

        
        # l2
        l2 = self.cal_l2(perturbations_)
        return margin, l2
    
  
    
    def take_adv_img(self, perturbation):
        adv_imgs = self.img_tensor + perturbation
        adv_imgs = torch.clamp(adv_imgs, 0, 1)
        pil_adv_imgs = tensor_to_pillow(adv_imgs) # pillow image
        return adv_imgs, pil_adv_imgs
    
    
# ----- HÀM TEST -----
def test_decoder_linear():
    H = W = 32
    C = 1
    pop = 1
    f_ratio = 0.5  # giữ 50% tần số đầu
    device = 'cpu'

    decoder = DCTDecoder(f_ratio, device=device)

    # tạo ngẫu nhiên hệ số DCT (latent)
    coeffs = torch.randn((pop, C, int(f_ratio*H), int(f_ratio*W)), device=device)
    alpha = 0.3  # hệ số co giãn

    # decode
    X1 = decoder(coeffs, W, H)
    X2 = decoder(alpha * coeffs, W, H)

    # kiểm tra tuyến tính
    a = torch.norm(X1)
    b = torch.norm(X2)
    print(b/a)

    # visualize 1 kênh để trực quan
    img1 = X1[0,0].cpu().numpy()
    img2 = X2[0,0].cpu().numpy()



# ----- CHẠY TEST -----
test_decoder_linear()

    

    
    

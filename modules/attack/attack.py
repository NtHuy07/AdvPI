import torch
import numpy as np
from typing import Any, Dict
from .util import clamp_eps, project_delta
from tqdm import tqdm
from time import time




class BaseAttack:

    def __init__(self, evaluator, eps=8/255, norm="l2", device=None):
        self.evaluator = evaluator
        self.eps = float(eps)
        self.norm = norm
        self.device = device if device is not None else next(self.evaluator.model.parameters()).device

    def evaluate_population(self, deltas: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            margins, l2s = self.evaluator.evaluate_blackbox(deltas)  # torch (pop,)
            return margins.clone(), l2s.clone()
        
    def is_success(self, margin):
        if margin < 0:
            return True
        return False
    
    def z_to_delta(self, z):
        s = torch.tanh(z)           # s in (-1,1)
        return self.eps * s

class ES_1_Lambda(BaseAttack):
    def __init__(self, evaluator, eps=8/255, norm="linf",
                 max_evaluation=10000, lam=64, c_inc=1.5, c_dec=0.9, device='cuda'):
        super().__init__(evaluator, eps, norm, device)
        # assert lam >= 2 and c_inc > 1.0 and 0.0 < c_dec < 1.0
        self.lam = int(lam)
        self.c_inc = float(c_inc)
        self.c_dec = float(c_dec)
        self.sigma = 1.1  # σ tuyệt đối
        self.max_evaluation = max_evaluation

    def run(self) -> Dict[str, Any]:
        sigma = self.sigma
        if self.evaluator.decoder:
            _, C, H, W = self.evaluator.lq_shape
        else:
            _, C, H, W = self.evaluator.img_tensor.shape
        
        m = torch.randn((1, C, H, W), device=self.device)
        delta_m = self.z_to_delta(m)
        delta_m = project_delta(delta_m, self.eps, self.norm)

        f_m, l2_m = self.evaluator.evaluate_blackbox(delta_m)
        history = [[float(f_m.item()), float(l2_m.item())]]

        num_evaluation = 1
        while num_evaluation < self.max_evaluation:
            noise = torch.randn((self.lam, C, H, W), device=self.device)
            X = m + sigma * noise
            X_delta = self.z_to_delta(X)
            X_delta = project_delta(X_delta, self.eps, self.norm)

            margins, l2s = self.evaluate_population(X_delta)
            num_evaluation += self.lam
            idx_best = torch.argmin(margins).item()
            x_best = X[idx_best].clone()
            f_best = float(margins[idx_best].item())
            l2_best = float(l2s[idx_best].item())
            x_delta_best = X_delta[idx_best].clone()
            if f_best < f_m:
                m = x_best.clone()
                delta_m = x_delta_best.clone()
                l2_m = l2_best
                f_m = f_best
                sigma *= self.c_inc
                # sigma = min(self.eps, self.sigma)
            else:
                sigma *= self.c_dec            
                # sigma = max(1e-6, sigma)     
            
            # print(f"[{num_evaluation} - attack phase] Best loss: ", f_m, " L2: ", l2_m )

            history.append([float(f_m), float(l2_m)])
            if self.is_success(f_m):
                break
            
        if self.evaluator.decoder:
            delta_m = self.evaluator.decoder(delta_m, self.evaluator.img_W, self.evaluator.img_H)
            delta_m = project_delta(delta_m, self.eps, self.norm)

            
        return {"best_delta": delta_m, "best_margin": f_m, "history": history, "num_evaluation": num_evaluation}


class RandomSearch(BaseAttack):
    def __init__(self,
                 evaluator,
                 eps=8/255,
                 norm="linf",
                 max_evaluation=10000,
                 lam=64,
                 device='cuda'):
        """
        Random Search thuần, nhưng dạng population-based cho giống ES_1_Lambda:

        - lam: số lượng mẫu (population size) mỗi vòng lặp.
        - max_evaluation: tổng số lần query tối đa (trên từng delta).
        """
        super().__init__(evaluator, eps, norm, device)
        self.max_evaluation = int(max_evaluation)
        self.lam = int(lam)

    def run(self) -> Dict[str, Any]:
        # Lấy shape ảnh
        if self.evaluator.decoder:
            _, C, H, W = self.evaluator.lq_shape
        else:
            _, C, H, W = self.evaluator.img_tensor.shape

        best_margin = float("inf")
        best_l2 = float("inf")
        best_delta = None
        history = []
        num_evaluation = 0

        while num_evaluation < self.max_evaluation:
            remaining = self.max_evaluation - num_evaluation
            batch_size = min(self.lam, remaining)

            z = torch.randn((batch_size, C, H, W), device=self.device)
            deltas = self.z_to_delta(z)
            deltas = project_delta(deltas, self.eps, self.norm)

            margins, l2s = self.evaluate_population(deltas)
            num_evaluation += batch_size

            idx_best = torch.argmin(margins).item()
            margin_batch_best = float(margins[idx_best].item())
            l2_batch_best = float(l2s[idx_best].item())
            delta_batch_best = deltas[idx_best:idx_best+1].clone()

            if margin_batch_best < best_margin:
                best_margin = margin_batch_best
                best_l2 = l2_batch_best
                best_delta = delta_batch_best

            history.append([best_margin, best_l2])

            if self.is_success(margin_batch_best):
                break

        if best_delta is None:
            best_delta = torch.zeros((1, C, H, W), device=self.device)

        if self.evaluator.decoder:
            best_delta = self.evaluator.decoder(
                best_delta, self.evaluator.img_W, self.evaluator.img_H
            )
            best_delta = project_delta(best_delta, self.eps, self.norm)

        return {
            "best_delta": best_delta,
            "best_margin": best_margin,
            "history": history,
            "num_evaluation": num_evaluation
        }



class NESAttack(BaseAttack):
    def __init__(self,
                 evaluator,
                 eps=8/255,
                 norm="linf",
                 max_evaluation=10000,
                 nes_samples=50,
                 nes_batch=32,
                 sigma=0.5/255,
                 alpha=1/255,
                 device='cuda'):
        """
        NES black-box attack trong không gian z (giống ES_1_Lambda):

        - nes_samples: số hướng nhiễu (số cặp + / -) cho mỗi vòng lặp outer.
        - nes_batch: batch size khi query (để không vỡ VRAM).
        - sigma: std cho smoothing NES.
        - alpha: step size cập nhật m.
        """
        super().__init__(evaluator, eps, norm, device)
        self.max_evaluation = int(max_evaluation)
        self.nes_samples = int(nes_samples)
        self.nes_batch = int(nes_batch)
        self.sigma = float(sigma)
        self.alpha = float(alpha)

    def run(self) -> Dict[str, Any]:
        if self.evaluator.decoder:
            _, C, H, W = self.evaluator.lq_shape
        else:
            _, C, H, W = self.evaluator.img_tensor.shape

        device = self.device

        # Khởi tạo m (latent) = 0 -> delta ~ 0 (giống PGD bắt đầu từ ảnh gốc)
        m = torch.zeros((1, C, H, W), device=device)

        # Delta hiện tại
        delta_m = self.z_to_delta(m)
        delta_m = project_delta(delta_m, self.eps, self.norm)

        # Đánh giá ban đầu
        margins, l2s = self.evaluate_population(delta_m)
        f_m = float(margins[0].item())
        l2_m = float(l2s[0].item())

        best_margin = f_m
        best_l2 = l2_m
        best_delta = delta_m.clone()

        history = [[f_m, l2_m]]
        num_evaluation = 1  # đã query 1 lần

        # ================ Vòng lặp NES ================
        while num_evaluation < self.max_evaluation and not self.is_success(f_m):

            remaining = self.max_evaluation - num_evaluation
            max_samples_by_budget = max(1, remaining // 2)
            total_samples = min(self.nes_samples, max_samples_by_budget)

            grad_accum = torch.zeros_like(m, device=device)
            used = 0
            s = 0

            while s < total_samples and num_evaluation < self.max_evaluation:
                bsz = min(self.nes_batch, total_samples - s)

                z = torch.randn((bsz, C, H, W), device=device)  # [B,C,H,W]

                m_pos = m + self.sigma * z
                m_neg = m - self.sigma * z

                delta_pos = self.z_to_delta(m_pos)
                delta_neg = self.z_to_delta(m_neg)
                delta_pos = project_delta(delta_pos, self.eps, self.norm)
                delta_neg = project_delta(delta_neg, self.eps, self.norm)

                deltas = torch.cat([delta_pos, delta_neg], dim=0)  # [2B,C,H,W]

                margins_batch, _ = self.evaluate_population(deltas)
                num_evaluation += 2 * bsz

                loss_all = margins_batch.view(-1)  # [2B]
                loss_pos = loss_all[:bsz].view(bsz, 1, 1, 1)
                loss_neg = loss_all[bsz:].view(bsz, 1, 1, 1)

                # grad ≈ 1/(2σ) * Σ (f_pos - f_neg) * z
                grad_chunk = ((loss_pos - loss_neg) * z).sum(dim=0, keepdim=True)  # [1,C,H,W]
                grad_accum = grad_accum + grad_chunk

                used += bsz
                s += bsz

            grad_est = grad_accum / (2.0 * self.sigma * max(1, used))

            m = m - self.alpha * grad_est.sign()
            delta_m = self.z_to_delta(m)
            delta_m = project_delta(delta_m, self.eps, self.norm)

            # Đánh giá current point
            margins, l2s = self.evaluate_population(delta_m)
            num_evaluation += 1

            f_m = float(margins[0].item())
            l2_m = float(l2s[0].item())
            history.append([f_m, l2_m])

            # Cập nhật best theo margin
            if f_m < best_margin:
                best_margin = f_m
                best_l2 = l2_m
                best_delta = delta_m.clone()

        # Decode nếu dùng decoder (giống ES_1_Lambda)
        if self.evaluator.decoder:
            best_delta = self.evaluator.decoder(best_delta,
                                                self.evaluator.img_W,
                                                self.evaluator.img_H)
            best_delta = project_delta(best_delta, self.eps, self.norm)

        return {
            "best_delta": best_delta,
            "best_margin": best_margin,
            "history": history,
            "num_evaluation": num_evaluation,
        }



class ES_1_Lambda_visual(BaseAttack):
    def __init__(self, evaluator, eps=8/255, norm="linf", max_evaluation=10000, _bs_steps=20, additional_eval=200,
                 lam=64, c_inc=1.5, c_dec=0.9, device='cuda'):
        super().__init__(evaluator, eps, norm, device)
        # assert lam >= 2 and c_inc > 1.0 and 0.0 < c_dec < 1.0
        self.lam = int(lam)
        self.c_inc = float(c_inc)
        self.c_dec = float(c_dec)
        self.sigma = 1.1  # σ tuyệt đối
        self._bs_steps = _bs_steps
        self.visual_interval = 5
        self.max_evaluation = max_evaluation
        self.additional_eval = additional_eval

    def optimize_visual(self, m, delta_m, f_m, l2_m):

        left, right = 0.0, 1.0
        best_m = m.clone()
        best_margin = float(f_m)
        best_l2 = l2_m
        best_delta_m = delta_m.clone()
        num_evaluation = 0
        for _ in range(self._bs_steps):
            alpha = 0.5 * (left + right)
            m_try = alpha * m
            m_delta_try = self.z_to_delta(m_try)
            m_delta_try = project_delta(m_delta_try, self.eps, self.norm)
            margin_try, l2_try = self.evaluator.evaluate_blackbox(m_delta_try)
            num_evaluation += 1
            if self.is_success(margin_try):
                right = alpha
                best_m = m_try
                best_margin = margin_try
                best_delta_m = m_delta_try
                best_l2 = l2_try
            else:
                left = alpha

        return best_m, best_delta_m, best_margin, num_evaluation, best_l2
            


    def run(self) -> Dict[str, Any]:
        sigma = self.sigma
        if self.evaluator.decoder:
            _, C, H, W = self.evaluator.lq_shape
        else:
            _, C, H, W = self.evaluator.img_tensor.shape
            
        m = torch.randn((1, C, H, W), device=self.device)
        delta_m = self.z_to_delta(m)
        delta_m = project_delta(delta_m, self.eps, self.norm)

        f_m, l2_m = self.evaluator.evaluate_blackbox(delta_m)
        history = [[float(f_m.item()), float(l2_m.item())]]
    
        success = False
        num_evaluation = 1
        while num_evaluation < self.max_evaluation:
            if success == True and num_evaluation > stop_num_evaluation:
                break
                
            noise = torch.randn((self.lam, C, H, W), device=self.device)
            X = m + sigma * noise
            X_delta = self.z_to_delta(X)
            X_delta = project_delta(X_delta, self.eps, self.norm)

            margins, l2s = self.evaluate_population(X_delta)
            num_evaluation += self.lam
            idx_best = torch.argmin(margins).item()
            x_best = X[idx_best].clone()
            f_best = float(margins[idx_best].item())
            l2_best = float(l2s[idx_best].item())
            x_delta_best = X_delta[idx_best].clone()

            if f_best < f_m:
                m = x_best.clone()
                delta_m = x_delta_best.clone()
                f_m = f_best
                l2_m = l2_best
                sigma *= self.c_inc
                # sigma = min(self.eps, sigma)
            else:
                sigma *= self.c_dec       
                # sigma = max(1e-6, sigma)     
            
            history.append([f_m, l2_m])
            
            # print(f"[{num_evaluation} - attack phase] Best loss: ", f_m, " L2: ", l2_m )
            
            if self.is_success(f_m): # neus lần đầu success
                m, m_delta, f_m, visual_evaluation, l2_m = self.optimize_visual(m, delta_m, f_m, l2_m)
                delta_m = self.z_to_delta(m)
                delta_m = project_delta(delta_m, self.eps, self.norm)
                num_evaluation += visual_evaluation
                if success == False:
                    stop_num_evaluation = num_evaluation + self.additional_eval # chạy thêm 50 dòng nữa
                    success = True
                
                
        if self.evaluator.decoder:
            delta_m = self.evaluator.decoder(delta_m, self.evaluator.img_W, self.evaluator.img_H)
            delta_m = project_delta(delta_m, self.eps, self.norm)

            
        return {"best_delta": delta_m, "best_margin": f_m, "history": history, "num_evaluation": num_evaluation}


class PGDAttack(BaseAttack):
    def __init__(self, eps, alpha, norm, steps, evaluator):
        self.eps = eps
        self.alpha = alpha
        self.norm = norm
        self.steps = steps
        self.evaluator = evaluator
    
    def run(self):
        delta = torch.zeros_like(self.evaluator.img_tensor).to(self.evaluator.img_tensor.device)
        delta.requires_grad = True

        for step in range(self.steps):
            margin, _ = self.evaluator.evaluate_whitebox(delta)
            loss = margin.mean()
            print("Loss: ", loss)
            loss.backward()
            if loss < 0:
                break
            with torch.no_grad():
                if self.norm == "linf":
                    delta.data = delta - self.alpha * delta.grad.sign()
                    delta.data = clamp_eps(delta.data, self.eps, norm="linf")
                elif self.norm == "l2":
                    grad_norm = torch.norm(delta.grad.view(delta.size(0), -1), dim=1).view(-1, 1, 1, 1)
                    scaled_grad = delta.grad / (grad_norm + 1e-10)
                    delta.data = delta - self.alpha * scaled_grad
                    delta.data = clamp_eps(delta.data, self.eps, norm="l2")
                delta.grad.zero_()

        final_margin, _ = self.evaluator.evaluate_whitebox(delta)
        return {
            "best_delta": delta.detach(),
            "best_margin": float(final_margin.item()),
            "history": None,
            "num_evaluation": step
        }
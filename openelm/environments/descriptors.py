import textwrap
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

import lpips
from PIL import Image
from PIL import Image, ImageDraw, ImageFont


class FastPerplexity:
    """
    Optimized perplexity computation with model caching and compilation.
    """
    def __init__(self, model_name="gpt2", device=None, compile_model=True):
        """
        Initialize and cache model (do this once).
        
        Args:
            model_name: HuggingFace model name
            device: "cuda" or "cpu" (auto-detected if None)
            compile_model: Use torch.compile for 2-3x speedup (PyTorch 2.0+)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,  # FP16 for speed
        ).to(self.device)
        self.model.eval()
        
        # Torch compile for ~2-3x speedup (PyTorch 2.0+)
        if compile_model and hasattr(torch, 'compile'):
            print("Compiling model (first run will be slow, then much faster)...")
            self.model = torch.compile(self.model)
        
        self.max_length = getattr(self.model.config, 'n_positions', 1024)
    
    @torch.no_grad()
    def __call__(self, text, stride=512):
        """
        Compute perplexity (fast, after first compilation).
        
        Args:
            text: Input text string or list of strings
            stride: Stride for sliding window (larger = faster but less accurate)
        """
        if isinstance(text, list):
            return [self._compute_single(t, stride) for t in text]
        return self._compute_single(text, stride)
    
    def _compute_single(self, text, stride):
        # Tokenize
        encodings = self.tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encodings.input_ids.to(self.device)
        seq_len = input_ids.size(1)
        
        # Handle short sequences (no sliding window needed)
        if seq_len <= self.max_length:
            target_ids = input_ids.clone()
            outputs = self.model(input_ids, labels=target_ids)
            return torch.exp(outputs.loss).item()
        
        # Sliding window for long sequences
        nlls = []
        prev_end_loc = 0
        
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            
            input_batch = input_ids[:, begin_loc:end_loc]
            target_ids = input_batch.clone()
            target_ids[:, :-trg_len] = -100
            
            outputs = self.model(input_batch, labels=target_ids)
            nlls.append(outputs.loss * trg_len)
            
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        
        ppl = torch.exp(torch.stack(nlls).sum() / (seq_len - 1))
        return ppl.item()


def compute_perplexity_batch(texts, model_name="gpt2", device=None, max_length=1024):
    """
    Batch processing for multiple texts (even faster for many texts).
    
    Args:
        texts: List of text strings
        model_name: HuggingFace model name
        device: "cuda" or "cpu"
        max_length: Maximum sequence length
    
    Returns:
        List of perplexity values
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    
    # Batch tokenization
    encodings = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=max_length
    )
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        
        # Compute per-sample perplexity
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        ).view(shift_labels.size())
        
        # Mask padding tokens
        mask = attention_mask[..., 1:].contiguous()
        loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)
        perplexities = torch.exp(loss).cpu().numpy()
    
    return perplexities.tolist()





# -----------------------------
# Resizing helper
# -----------------------------
def resize_to_match(img1, img2):
    h = max(img1.shape[0], img2.shape[0])
    w = max(img1.shape[1], img2.shape[1])

    def resize(img):
        return np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))

    return resize(img1), resize(img2)


# -----------------------------
# NEW: wrap tokens according to visual width
# -----------------------------
def wrap_tokens_by_width(tokens, font, max_width):
    """
    Returns:
        lines: list[str] rendered lines
        token_lines: list[list[int]] token indices for each line
    """
    lines = []
    token_lines = []

    current_tokens = []
    current_indices = []
    current_width = 0

    space_width = font.getlength(" ")

    for i, tok in enumerate(tokens):
        tok_width = font.getlength(tok)

        add_width = tok_width if not current_tokens else space_width + tok_width

        # if adding token exceeds width → start new line
        if current_width + add_width > max_width and current_tokens:
            lines.append(" ".join(current_tokens))
            token_lines.append(current_indices)

            current_tokens = []
            current_indices = []
            current_width = 0

        current_tokens.append(tok)
        current_indices.append(i)
        current_width += add_width

    # flush last line
    if current_tokens:
        lines.append(" ".join(current_tokens))
        token_lines.append(current_indices)

    return lines, token_lines


# -----------------------------
# Render text1 → discover line mapping
# -----------------------------
def render_token_text(tokens, font_path="DejaVuSans.ttf", size=32, max_width=500, margin=10):
    font = ImageFont.truetype(font_path, size)

    # wrap according to width
    lines, token_lines = wrap_tokens_by_width(tokens, font, max_width)

    bbox = font.getbbox("Ay")
    line_height = (bbox[3] - bbox[1]) + 6

    img_height = line_height * len(lines) + 2 * margin
    img_width = max_width + 2 * margin

    img = Image.new("L", (img_width, img_height), 255)
    draw = ImageDraw.Draw(img)

    y = margin
    for line in lines:
        draw.text((margin, y), line, font=font, fill=0)
        y += line_height

    return np.array(img), token_lines


# -----------------------------
# Render text2 using SAME line mapping
# -----------------------------
def render_tokens_from_line_mapping(tokens, token_lines, font_path="DejaVuSans.ttf", size=32, max_width=500, margin=10):
    font = ImageFont.truetype(font_path, size)

    lines = []
    for idxs in token_lines:
        line_tokens = [tokens[i] for i in idxs]
        lines.append(" ".join(line_tokens))

    bbox = font.getbbox("Ay")
    line_height = (bbox[3] - bbox[1]) + 6

    img_height = line_height * len(lines) + 2 * margin
    img_width = max_width + 2 * margin

    img = Image.new("L", (img_width, img_height), 255)
    draw = ImageDraw.Draw(img)

    y = margin
    for line in lines:
        draw.text((margin, y), line, font=font, fill=0)
        y += line_height

    return np.array(img)


# -----------------------------
# Your existing helpers
# -----------------------------
def crop_to_content(img, threshold=250):
    mask = img < threshold
    coords = np.argwhere(mask)

    if coords.shape[0] == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]


def pad_min_size(img, min_size=64):
    h, w = img.shape
    new_h = max(h, min_size)
    new_w = max(w, min_size)

    canvas = Image.new("L", (new_w, new_h), 255)
    canvas.paste(Image.fromarray(img), ((new_w - w)//2, (new_h - h)//2))
    return np.array(canvas)


def to_lpips_tensor(img):
    img = Image.fromarray(img).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    img = img * 2 - 1
    return img


# -----------------------------
# LPIPS model
# -----------------------------
lpips_model = lpips.LPIPS(net='alex')


# -----------------------------
# MAIN FUNCTION (token-aligned LPIPS)
# -----------------------------
def lpips_distance(tokens1, tokens2, font_path="DejaVuSans.ttf"):

    assert len(tokens1) == len(tokens2), "token lists must have same length"

    # text1 decides wrapping & token grouping
    img1, token_lines = render_token_text(tokens1, font_path=font_path)

    # text2 forced to follow same grouping
    img2 = render_tokens_from_line_mapping(tokens2, token_lines, font_path=font_path)

    # postprocess as before
    img1 = crop_to_content(img1)
    img2 = crop_to_content(img2)

    img1 = pad_min_size(img1)
    img2 = pad_min_size(img2)

    img1, img2 = resize_to_match(img1, img2)

    # show_pair(img1, img2)

    t1 = to_lpips_tensor(img1)
    t2 = to_lpips_tensor(img2)

    return float(lpips_model(t1, t2))


def show_pair(img1, img2, title1="Image 1", title2="Image 2"):
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(img1, cmap="gray")
    plt.title(title1)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(img2, cmap="gray")
    plt.title(title2)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


class TokenSim:
    def __init__(self, orig_tokens, model_name="bert-base-uncased"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        self.bert.eval()

        self.orig_tokens = orig_tokens
        self.emb1, self.mask1 = self.get_input_embeddings(orig_tokens)

    def get_input_embeddings(self, texts):
        """
        texts: list of strings
        returns: tensor [B, L, D], attention_mask
        """

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        emb = self.bert.get_input_embeddings()(encoded["input_ids"])

        return emb, encoded["attention_mask"]

    def avg_token_embedding_shift(self, adv_tokens):
        """
        Compute mean cosine distance per pair in batch.
        """

        assert len(self.orig_tokens) == len(adv_tokens)


        # get embeddings for whole batches
        emb2, mask2 = self.get_input_embeddings(adv_tokens)

        L1 = self.mask1.size(1)
        L2 = mask2.size(1)

        max_len = max(L1, L2)

        # pad masks
        mask1 = F.pad(self.mask1, (0, max_len - L1))
        mask2 = F.pad(mask2, (0, max_len - L2))

        # pad embeddings too
        new_emb1 = F.pad(self.emb1, (0, 0, 0, max_len - L1))
        new_emb2 = F.pad(emb2, (0, 0, 0, max_len - L2))

        mask = mask1 & mask2

        # normalize embeddings
        emb1_n = F.normalize(new_emb1, dim=-1)
        emb2_n = F.normalize(new_emb2, dim=-1)

        # cosine similarity per token
        cos = (emb1_n * emb2_n).sum(dim=-1)  # [B, L]

        # distance = 1 - similarity
        dist = 1 - cos

        # mask invalid padded tokens
        dist = dist * mask

        # sum and divide by valid tokens
        per_pair_mean = dist.sum(dim=-1) / mask.sum(dim=-1)

        overall_mean = per_pair_mean.mean().item()

        return (
            overall_mean,
            per_pair_mean.tolist()
        )
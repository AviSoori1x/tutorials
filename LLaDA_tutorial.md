 ** code‑first tutorial** to help you deeply understand the LLaDA paper (**“Large Language Diffusion Models”**)


---

## Table of contents

1. **Big picture in plain English** (with analogies)
2. **A quick probability & MLE primer** (gentle)
3. **Two model families**: Autoregressive vs. Masked Diffusion
4. **LLaDA’s formulation** (forward & reverse masking)
5. **Training objective** and why the factor **1/t** matters
6. **Supervised fine‑tuning (SFT)** as conditional modeling
7. **Inference (sampling)**: low‑confidence re‑masking (core idea)
8. **Likelihood estimation** you’ll actually use on benchmarks
9. **Any‑order AR connection** (why LLaDA reasons both ways)
10. **Classifier‑Free Guidance** (optional boost)
11. **Code, step‑by‑step** (tiny PyTorch demo)
12. **End‑to‑end runnable file**
13. **Checks, tips, and exercises** to own the ideas

---

## 1) Big picture in plain English

**What makes LLaDA different?**
Instead of predicting the next token **left→right** like classic LLMs, LLaDA trains a **masked diffusion** model:

* **Forward process:** take a sentence, **mask** (hide) each token independently with probability (t). Bigger (t) = more masks. (Think: **scratch‑off card**—the more silver paint, the less text you see.)
* **Reverse process:** a Transformer (no causal mask) learns to **fill in the masked tokens** given the visible context, and we **repeat** fill→re‑mask in steps from (t{=}1) (everything masked) down to (t{=}0) (nothing masked).
* Same recipe works for **pre‑training** and **SFT** (instruction tuning)—see **Figure 2**, panels (a) and (b), and **sampling** in panel (c). (p. 3) 

**Why this is exciting:** LLaDA shows that many “LLM superpowers”—**scaling**, **in‑context learning**, **instruction‑following**—come from **generative modeling principles** (MLE) + Transformers, **not only** from left‑to‑right factorization. (pp. 1–2) 

**Scale results:** an 8B model, **2.3T** pre‑training tokens, **4.5M** SFT pairs—competitive with strong AR baselines across tasks and especially strong on some **reversal reasoning** tests (Tables 1–2 and Table 4; Fig. 3 shows scaling). (pp. 2, 6–9) 

---

## 2) A quick probability & MLE primer

**Goal:** learn a model (p_\theta(x)) that matches the true data (p_{\text{data}}(x)).

**Equation (1)** (p. 1):
[
\max_\theta \mathbb{E}*{x\sim p*{\text{data}}}[\log p_\theta(x)]
\quad\Longleftrightarrow\quad
\min_\theta \mathrm{KL}!\left(p_{\text{data}},|,p_\theta\right).
]
**Plain meaning:** maximize average log‑likelihood of real sentences (or minimize how different your model’s distribution is from reality).

**Analogy:** imagine a **histogram** over sentences; MLE adjusts your model’s bars to match the real bars.

**Pythonic pseudocode**

```python
for x in dataloader:           # x ~ p_data
    loss = -model.log_prob(x)  # maximize log p = minimize negative log p
    loss.backward()
    opt.step()
```

(Equation (1) and the MLE view are emphasized in Section 1.) 

---

## 3) Two model families

### 3.1 Autoregressive (AR) models (Eq. (2))

**Factorization** (p. 1–2):
[
p_\theta(x_1,\ldots,x_L) = p_\theta(x_1)\prod_{i=2}^L p_\theta(x_i\mid x_1,\ldots,x_{i-1}).
]
Train by **next‑token prediction**. Great, but strictly **left→right** at inference.

### 3.2 Masked diffusion (LLaDA)

LLaDA (a **masked diffusion model**, MDM) replaces “predict next token” with “**predict the masked tokens anywhere**,” using a forward mask process and a learned reverse process. (Sec. 2, App. A, pp. 3, 19–22; Fig. 2.) 

**Analogy:** solving a **crossword**: some squares blank (masked), some filled (visible). You fill blanks using everything visible (both left & right).

---

## 4) LLaDA’s formulation—forward & reverse masking

### 4.1 Forward process (q_{t|0}) (Eqs. (7)–(8), p. 19)

Mask each token independently with probability (t):
[
q_{t|0}(x_t|x_0)=\prod_{i=1}^L q_{t|0}(x_t^i|x_0^i),\quad
q_{t|0}(x_t^i=x_0^i)=1-t,;;q_{t|0}(x_t^i=M)=t.
]
**Analogy:** flip a **biased coin** (heads (=M)) for each word.

**Mini example** (5‑token sentence, (t=0.4)): expect 2 out of 5 masked on average.

### 4.2 Reverse process (q_{s|t}) (Eqs. (9)–(11), pp. 20–21)

Go from more masked to less masked (t\to s<t). The key term is **data prediction**:
[
q_{0|t}(x_s^i\mid x_t) = p_{\text{data}}(x_0^i \mid x_t^{UM})\quad\text{for masked positions}.
]
**Time‑free insight (Eq. (11))**: the optimal predictor **doesn’t need (t) as input**—just the partially masked sequence (x_t). (p. 20) 

---

## 5) Training objective (why the (1/t) factor matters)

**Core loss (Eq. (3), p. 3; Eq. (12), p. 21):**
[
\mathcal{L}(\theta) ;=;
-\mathbb{E}*{t,x_0,x_t}\Big[
\frac{1}{t}\sum*{i=1}^L \mathbf{1}{x_t^i=M};\log p_\theta(x_0^i\mid x_t)
\Big].
]

* Cross‑entropy **only on masked positions** (we don’t “learn” what we already see).
* Scale by **(1/t)**. This is crucial: it normalizes for the expected number of masked tokens (about (tL)) and, more importantly, is what makes the loss a **principled MLE‑style objective**.

**Bound (Eq. (4)/(13))**:
[
-\mathbb{E}*{x_0\sim p*{\text{data}}}\big[\log p_\theta(x_0)\big] ;\le; \mathcal{L}(\theta).
]
So minimizing (\mathcal{L}) **upper‑bounds the negative log‑likelihood** (equivalently, **lower‑bounds the log‑likelihood**). (pp. 3, 21) 

**BERT/MaskGIT vs LLaDA:** BERT uses a **fixed** mask ratio and **no (1/t)**; MaskGIT also misses (1/t)—so it lacks the clean link to MLE that LLaDA has. (Sec. 2.1, p. 4) 

---

## 6) Supervised Fine‑Tuning (SFT) as conditional modeling

For prompt–response ((p_0,r_0)), LLaDA trains the same way **but masks only the response** (Fig. 2b, Eq. (5)):
[
-\mathbb{E}*{t,p_0,r_0,r_t}\Big[
\frac{1}{t}\sum*{i=1}^{L'} \mathbf{1}{r_t^i=M};\log p_\theta(r_0^i\mid p_0,r_t)
\Big].
]
They pad short pairs with (|\text{EOS}|) inside batches, treat (|\text{EOS}|) as a normal token during training, and **remove it** at generation time—this lets the model **learn** when to stop. (Sec. 2.3 and App. B.1, pp. 4–5, 22) 

---

## 7) Inference (sampling) with **low‑confidence re‑masking**

Discretize time (t_1{=}1 \to t_0{=}0) (say (N) steps). At each step:

1. **Fill all masks** using (p_\theta(\cdot\mid x_t)) (e.g., greedy per position).
2. **Re‑mask** some tokens to move from mask‑ratio (t) to (s<t).
3. Repeat to (t=0). (Sec. 2.4; App. A.3; **Algorithm 5** p. 21) 

**Low‑confidence re‑masking:** Among current tokens, keep the **highest‑confidence** ones unmasked so the number of unmasked tokens matches the new step; **re‑mask** the rest (lowest confidence). This beats random re‑masking (Table 9). (pp. 5, 24–25) 

**Speed/quality dial:** fewer steps = faster, more steps = higher quality; see **Fig. 5** (throughput vs accuracy). (p. 26) 

**Flexible strategies:** LLaDA also supports **AR**, **block diffusion**, and **block diffusion LLaDA** (semi‑AR). Pure diffusion is usually best (Fig. 4; Tables 7–8). (pp. 24–25) 

---

## 8) Likelihood estimation you’ll actually use

For multiple‑choice tasks you need (\log p_\theta(r_0\mid p_0)). Use the **lower‑variance** Monte‑Carlo estimator (Eq. (6), p. 5; also Eq. (14), p. 21):
[
-\mathbb{E}*{l,r_0,r_l}\Big[\frac{L}{l}
\sum*{i=1}^L \mathbf{1}{r_l^i=M};\log p_\theta(r_0^i\mid p_0,r_l)\Big].
]
Here you **mask exactly (l)** tokens (without replacement). Why better? Because (l) controls the masked **count** deterministically, reducing variance relative to Bernoulli masks. (pp. 5, 21) 

---

## 9) Any‑order AR connection (why LLaDA can “think both ways”)

App. A.2 shows the AO‑ARM loss (train over **all permutations** of token order) equals LLaDA’s masked loss (Eq. (12)). (Eq. (15), pp. 21–22)
**Plain meaning:** LLaDA implicitly **trains on many conditioning directions**, which helps **reversal reasoning** (Table 4 shows large gains vs strong ARMs). (pp. 8–9, 21–22) 

---

## 10) Classifier‑Free Guidance (CFG) (optional boost)

LLaDA is compatible with CFG; not used in main head‑to‑head tables for fairness, but ablations show **consistent improvements** (Table 6). The (unsupervised) CFG form (Eq. (16), p. 22):
[
\tilde p_\theta(r_0\mid p_0,r_t);\propto;
\frac{p_\theta(r_0\mid p_0,r_t)^{1+w}}{p_\theta(r_0\mid m,r_t)^{,w}},
]
with (m) a masked prompt and (w) a scaling knob. 

---

# 11) Code (step‑by‑step) — tiny PyTorch demo

> **Goal:** Implement the mechanics (forward mask, LLaDA loss (1/t), pre‑train + SFT loops, diffusion sampling with low‑confidence re‑masking, likelihood estimator), and show them **clearly** on a toy corpus you can run on CPU.

We’ll proceed in **small snippets** with explanations, then give you a single **end‑to‑end file** (§12).

---

### 11.1 Setup & toy tokenizer

```python
import math, random, itertools
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
rng = random.Random(0)
torch.manual_seed(0)
```

* Fix seeds so results are reproducible—critical when debugging sampling.

```python
SPECIAL = ["[PAD]", "[BOS]", "[EOS]", "[SEP]", "[MASK]"]
PAD, BOS, EOS, SEP, MASK = range(len(SPECIAL))

class TinyTokenizer:
    def __init__(self, texts: List[List[str]]):
        words = set(itertools.chain.from_iterable(texts))
        self.itos = SPECIAL + sorted(list(words - set(SPECIAL)))
        self.stoi = {w:i for i,w in enumerate(self.itos)}
    def encode(self, toks: List[str]): return [self.stoi[w] for w in toks]
    def decode(self, ids: List[int]):  return [self.itos[i] for i in ids]
    def __len__(self): return len(self.itos)
```

* We include a **dedicated mask token** `[MASK]` = (M) in Eqs. (7–8), (12). (App. A, pp. 19–21) 

---

### 11.2 Tiny corpora: pre‑train + SFT pairs

```python
pretrain_texts = [
    ["I","love","pizza"],
    ["I","love","sushi"],
    ["You","love","pizza"],
    ["Math","is","fun"],
    ["Transformers","are","powerful"],
]

sft_pairs = [
    (["Solve",":","2","+","2","="], ["4"]),
    (["Translate",":","hello","->","Spanish"], ["hola"]),
    (["What","is","the","capital","of","France","?"], ["Paris"]),
]

tok = TinyTokenizer(pretrain_texts + [p+r for p,r in sft_pairs])
V = len(tok)
```

* In the paper: **2.3T** tokens pretraining, **4.5M** SFT pairs; we’re just **mimicking the pipeline**. (Sec. 2.2–2.3, pp. 4–5) 

---

### 11.3 Forward masking (q_{t|0}) (Eqs. (7–8))

```python
def mask_sequence(x_ids: List[int], t: float, allow_indices=None):
    """
    Mask each allowed position independently with probability t.
    Returns masked ids and a boolean mask (True where we masked).
    """
    masked, mask_pos = [], []
    for i, tid in enumerate(x_ids):
        can = (allow_indices is None) or (i in allow_indices)
        if can and rng.random() < t:
            masked.append(MASK); mask_pos.append(True)
        else:
            masked.append(tid);   mask_pos.append(False)
    return masked, mask_pos
```

* **SFT detail** (later): we’ll pass `allow_indices` to mask **only the response** (Fig. 2b; Eq. (5)). 

**Analogy:** “fog‑of‑war” in games—each step reveals or hides squares.

---

### 11.4 The mask predictor (Transformer encoder; **no causal mask**)

```python
@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    max_len: int = 128
    dropout: float = 0.1

class MaskPredictor(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_len, config.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config.n_layers)
        self.lm_head = nn.Linear(config.d_model, vocab_size)

    def forward(self, x_ids: torch.Tensor, attn_mask: torch.Tensor):
        B, L = x_ids.shape
        pos = torch.arange(L, device=x_ids.device).unsqueeze(0).expand(B, L)
        h = self.tok_emb(x_ids) + self.pos_emb(pos)
        key_padding_mask = (attn_mask == 0)  # 0=PAD; don’t attend to PAD
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return self.lm_head(h)  # [B, L, V]
```

* Matches the **time‑free** predictor (p_\theta(\cdot\mid x_t)) (no (t) input), consistent with Eq. (11). (p. 20; Sec. 2.1) 

---

### 11.5 The LLaDA loss with the **(1/t)** factor (Eq. (3)/(12))

```python
def masked_xent_loss(logits, targets, mask_bool, t_scalar):
    """
    Cross-entropy computed only on masked positions, scaled by 1/t.
    """
    B, L, V = logits.shape
    logits  = logits.reshape(B*L, V)
    targets = targets.reshape(B*L)
    mask    = mask_bool.reshape(B*L)

    if mask.sum() == 0:
        # edge case: nothing was masked in this example
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    loss = F.cross_entropy(logits[mask], targets[mask])
    return loss / max(t_scalar, 1e-6)
```

* This implements Eq. (3)/(12). Why the (1/t)? It produces the **MLE‑linked bound** (Eq. (4)/(13)) and normalizes for the expected mask count (tL). (pp. 3–4, 21) 

---

### 11.6 Pre‑training data & loop (Algorithm 1)

**Batch maker** (sample (t\sim U(0,1]); mask all non‑PAD positions): (Alg. 1, p. 19) 

```python
def make_pt_batch(batch_size=16, max_len=16):
    xs, xmask, ys, ymask = [], [], [], []
    for _ in range(batch_size):
        sent = rng.choice(pretrain_texts)
        ids  = [BOS] + tok.encode(sent) + [EOS]

        L = min(len(ids), max_len)
        ids = ids[:L] + [PAD]*(max_len - L)

        t = max(rng.random(), 1e-3)  # avoid exactly 0
        valid_positions = [i for i,u in enumerate(ids) if u != PAD]

        masked_ids, mask_pos = mask_sequence(ids, t, allow_indices=valid_positions)
        xs.append(masked_ids); ys.append(ids)
        xmask.append([1 if u != PAD else 0 for u in ids])
        ymask.append(mask_pos)

    return (torch.tensor(xs,    dtype=torch.long, device=DEVICE),
            torch.tensor(xmask, dtype=torch.long, device=DEVICE),
            torch.tensor(ys,    dtype=torch.long, device=DEVICE),
            torch.tensor(ymask, dtype=torch.bool, device=DEVICE),
            t)
```

**Training loop**

```python
config = ModelConfig()
model  = MaskPredictor(V, config).to(DEVICE)
opt    = torch.optim.AdamW(model.parameters(), lr=2e-3)

for step in range(400):  # tiny run
    x, attn, y, m, t = make_pt_batch(batch_size=32, max_len=16)
    loss = masked_xent_loss(model(x, attn), y, m, t)
    opt.zero_grad(); loss.backward(); opt.step()
    if (step+1) % 100 == 0:
        print(f"[PT] step={step+1} loss={loss.item():.3f}")
```

**Paper detail:** they train with fixed 4096 seq length and **1% random shorter** sequences for variable‑length robustness. (Sec. 2.2) We reflect the spirit with random lengths. 

---

### 11.7 SFT data & loop (Eq. (5); Algorithm 2)

**Batch maker** (mask **only the response**): (Fig. 2b; Eq. (5); Alg. 2 p. 19) 

```python
def make_sft_batch(batch_size=8, max_len=24):
    xs, xmask, ys, ymask = [], [], [], []
    t = max(rng.random(), 1e-3)
    for _ in range(batch_size):
        p, r = rng.choice(sft_pairs)
        full = [BOS] + tok.encode(p) + [SEP] + tok.encode(r) + [EOS]

        L = min(len(full), max_len)
        full = full[:L] + [PAD]*(max_len - L)

        sep_idx   = full.index(SEP) if SEP in full else 0
        resp_idxs = [i for i,u in enumerate(full) if (i > sep_idx and u != PAD)]

        masked_ids, mask_pos = mask_sequence(full, t, allow_indices=resp_idxs)
        xs.append(masked_ids); ys.append(full)
        xmask.append([1 if u != PAD else 0 for u in full])
        ymask.append(mask_pos)

    return (torch.tensor(xs,    dtype=torch.long, device=DEVICE),
            torch.tensor(xmask, dtype=torch.long, device=DEVICE),
            torch.tensor(ys,    dtype=torch.long, device=DEVICE),
            torch.tensor(ymask, dtype=torch.bool, device=DEVICE),
            t)
```

**Loop**

```python
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
for step in range(200):
    x, attn, y, m, t = make_sft_batch(batch_size=16, max_len=24)
    loss = masked_xent_loss(model(x, attn), y, m, t)
    opt.zero_grad(); loss.backward(); opt.step()
    if (step+1) % 50 == 0:
        print(f"[SFT] step={step+1} loss={loss.item():.3f}")
```

* App. B.1 describes (|\text{EOS}|) padding for equal batch lengths; **treat EOS as normal token** in training; remove it at sampling. (pp. 5, 22) 

---

### 11.8 Diffusion sampling with **low‑confidence re‑masking** (Alg. 5)

At each step (t\to s):

* Fill all masks greedily to get predictions + **confidence** (max prob).
* Compute target **unmasked count** ( \text{nun}=\lfloor L(1-s)\rfloor ).
* **Keep the top‑confidence tokens** so exactly nun are unmasked; **re‑mask the rest**.
  This aligns reverse sampling with the forward process; it outperforms random re‑masking (Table 9). (App. A.3, p. 21; Sec. 2.4, p. 5; App. B.4, p. 25) 

```python
@torch.no_grad()
def sample_diffusion(prompt_tokens: List[str], out_len=8, steps=12, trace=False):
    prompt_ids = [BOS] + tok.encode(prompt_tokens) + [SEP]
    # Start at t=1: fully masked response plus an EOS sentinel at end
    rt = [MASK]*out_len + [EOS]
    seq = prompt_ids + rt
    attn = torch.tensor([[1]*len(seq)], device=DEVICE)
    seq  = torch.tensor([seq], dtype=torch.long, device=DEVICE)

    resp_start = len(prompt_ids)
    resp_end   = resp_start + out_len
    for k in range(steps, 0, -1):
        t = k/steps; s = (k-1)/steps

        logits = model(seq, attn)[0]
        probs  = F.softmax(logits, dim=-1)
        conf   = torch.ones(seq.shape[1], device=DEVICE)

        # Fill masks and record confidence
        for i, tid in enumerate(seq[0]):
            if tid.item() == MASK:
                topv, topi = probs[i].max(dim=-1)
                seq[0, i] = topi
                conf[i]   = topv

        # Keep highest-confidence tokens so nun = floor(out_len*(1-s)) remain unmasked
        nun = int(out_len*(1 - s))
        resp_conf = conf[resp_start:resp_end]
        idxs = torch.argsort(resp_conf, descending=True).tolist()
        keep = set(idxs[:nun])

        # Re-mask the rest (lowest confidence)
        for j, idx in enumerate(range(resp_start, resp_end)):
            if j not in keep:
                seq[0, idx] = MASK

        if trace:
            decoded = tok.decode(seq[0].tolist()[resp_start:resp_end])
            print(f"step {k:02d} -> nun={nun:02d}  state:", decoded)

    # Decode response (stop at EOS if present)
    final = tok.decode(seq[0].tolist()[resp_start:resp_end])
    out = []
    for w in final:
        if w == tok.itos[EOS]: break
        out.append(w)
    return out
```

> **Tip (App. B.4):** with lots of EOS in SFT data, you may **down‑weight EOS confidence** to avoid “early quitting”. They set EOS confidence to zero in some Instruct experiments. (p. 25) 

Try it:

```python
print(">>> diffusion sample:",
      sample_diffusion(["Translate",":","hello","->","Spanish"], out_len=4, steps=10, trace=True))
```

---

### 11.9 Conditional log‑likelihood (Eq. (6)) & multiple choice

This is the estimator you’ll use to **score candidate answers** (lower variance than straight Bernoulli masking). (Sec. 2.4; App. A.2) 

```python
@torch.no_grad()
def cond_loglik(prompt_tokens: List[str], resp_tokens: List[str], nmc=64):
    p_ids = [BOS] + tok.encode(prompt_tokens) + [SEP]
    r_ids = tok.encode(resp_tokens) + [EOS]
    L = len(r_ids)
    if L == 0: return 0.0

    total = 0.0
    for _ in range(nmc):
        l = rng.randint(1, L)
        idxs = rng.sample(range(L), l)  # choose l unique positions to mask
        masked = r_ids[:]
        for i in idxs: masked[i] = MASK

        x    = torch.tensor([p_ids + masked], dtype=torch.long, device=DEVICE)
        attn = torch.tensor([[1]*x.shape[1]], device=DEVICE)
        logp = F.log_softmax(model(x, attn)[0], dim=-1)

        off = len(p_ids)
        s = sum(logp[off+i, r_ids[i]].item() for i in idxs)
        total += (L / l) * s
    return total / nmc

def choose(prompt, choices):
    scores = [cond_loglik(prompt, c.split()) for c in choices]
    best = max(range(len(choices)), key=lambda i: scores[i])
    return choices[best], scores
```

Try it:

```python
prompt  = ["What","is","the","capital","of","France","?"]
choices = ["Paris","London","Berlin"]
print(">>> choose:", choose(prompt, choices))
```

---

### 11.10 (Optional) Block diffusion & AR sampling

* **Block diffusion:** do diffusion **inside a fixed‑length block**, move **left→right** across blocks (semi‑AR).
* **Block diffusion LLaDA:** a fixed‑length variant used in the paper; can help math/code in Instruct. (Fig. 4; Tables 7–8) 

**Pythonic pseudocode**

```python
# Split response into blocks of Lb
for b in range(num_blocks):
    # Freeze previous blocks; only current block can change
    for substep in range(K):               # small number of diffusion steps
        fill_masks_in_block_greedily()
        remask_low_conf_in_block_to_match_schedule()
# Concatenate blocks to get final text
```

(They still find **pure diffusion** best overall on their suite.) 

---

### 11.11 (Optional) Classifier‑Free Guidance (Eq. (16))

Compute two logits at the same masked positions:

* **Conditional**: (\log p_\theta(\cdot\mid p_0,r_t))
* **Unconditional** (masked prompt): (\log p_\theta(\cdot\mid m,r_t))

Then combine in log‑space:

```python
def cfg_logits(logits_cond, logits_uncond, w):
    logp_c = F.log_softmax(logits_cond, dim=-1)
    logp_u = F.log_softmax(logits_uncond, dim=-1)
    return (1+w)*logp_c - w*logp_u
```

(See Eq. (16) and Table 6.) 

---

## 12) End‑to‑end runnable file

> You’ve seen the pieces. Here is a single **copy‑paste** script that includes: tokenizer, masking, model, LLaDA loss with (1/t), pre‑training loop, SFT loop, diffusion sampling with **trace** printing, and conditional likelihood / multiple‑choice utilities.

```python
import math, random, itertools
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
rng = random.Random(0)
torch.manual_seed(0)

SPECIAL = ["[PAD]", "[BOS]", "[EOS]", "[SEP]", "[MASK]"]
PAD, BOS, EOS, SEP, MASK = range(len(SPECIAL))

class TinyTokenizer:
    def __init__(self, texts: List[List[str]]):
        words = set(itertools.chain.from_iterable(texts))
        self.itos = SPECIAL + sorted(list(words - set(SPECIAL)))
        self.stoi = {w:i for i,w in enumerate(self.itos)}
    def encode(self, toks: List[str]): return [self.stoi[w] for w in toks]
    def decode(self, ids: List[int]):  return [self.itos[i] for i in ids]
    def __len__(self): return len(self.itos)

pretrain_texts = [
    ["I","love","pizza"],
    ["I","love","sushi"],
    ["You","love","pizza"],
    ["Math","is","fun"],
    ["Transformers","are","powerful"],
]
sft_pairs = [
    (["Solve",":","2","+","2","="], ["4"]),
    (["Translate",":","hello","->","Spanish"], ["hola"]),
    (["What","is","the","capital","of","France","?"], ["Paris"]),
]
tok = TinyTokenizer(pretrain_texts + [p+r for p,r in sft_pairs])
V = len(tok)

def mask_sequence(x_ids: List[int], t: float, allow_indices=None):
    masked, mask_pos = [], []
    for i, tid in enumerate(x_ids):
        can = (allow_indices is None) or (i in allow_indices)
        if can and rng.random() < t:
            masked.append(MASK); mask_pos.append(True)
        else:
            masked.append(tid);   mask_pos.append(False)
    return masked, mask_pos

@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    max_len: int = 128
    dropout: float = 0.1

class MaskPredictor(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_len, config.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config.n_layers)
        self.lm_head = nn.Linear(config.d_model, vocab_size)
    def forward(self, x_ids: torch.Tensor, attn_mask: torch.Tensor):
        B, L = x_ids.shape
        pos = torch.arange(L, device=x_ids.device).unsqueeze(0).expand(B, L)
        h = self.tok_emb(x_ids) + self.pos_emb(pos)
        key_padding_mask = (attn_mask == 0)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        return self.lm_head(h)

def masked_xent_loss(logits, targets, mask_bool, t_scalar):
    B, L, V = logits.shape
    logits  = logits.reshape(B*L, V)
    targets = targets.reshape(B*L)
    mask    = mask_bool.reshape(B*L)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    loss = F.cross_entropy(logits[mask], targets[mask])
    return loss / max(t_scalar, 1e-6)

def make_pt_batch(batch_size=16, max_len=16):
    xs, xmask, ys, ymask = [], [], [], []
    for _ in range(batch_size):
        sent = rng.choice(pretrain_texts)
        ids  = [BOS] + tok.encode(sent) + [EOS]
        L = min(len(ids), max_len)
        ids = ids[:L] + [PAD]*(max_len-L)
        t = max(rng.random(), 1e-3)
        valid_positions = [i for i,u in enumerate(ids) if u != PAD]
        masked_ids, mask_pos = mask_sequence(ids, t, allow_indices=valid_positions)
        xs.append(masked_ids); ys.append(ids)
        xmask.append([1 if u!=PAD else 0 for u in ids]); ymask.append(mask_pos)
    return (torch.tensor(xs, dtype=torch.long, device=DEVICE),
            torch.tensor(xmask, dtype=torch.long, device=DEVICE),
            torch.tensor(ys, dtype=torch.long, device=DEVICE),
            torch.tensor(ymask, dtype=torch.bool, device=DEVICE),
            t)

def make_sft_batch(batch_size=8, max_len=24):
    xs, xmask, ys, ymask = [], [], [], []
    t = max(rng.random(), 1e-3)
    for _ in range(batch_size):
        p, r = rng.choice(sft_pairs)
        full = [BOS] + tok.encode(p) + [SEP] + tok.encode(r) + [EOS]
        L = min(len(full), max_len)
        full = full[:L] + [PAD]*(max_len-L)
        sep_idx   = full.index(SEP) if SEP in full else 0
        resp_idxs = [i for i,u in enumerate(full) if (i>sep_idx and u!=PAD)]
        masked_ids, mask_pos = mask_sequence(full, t, allow_indices=resp_idxs)
        xs.append(masked_ids); ys.append(full)
        xmask.append([1 if u!=PAD else 0 for u in full]); ymask.append(mask_pos)
    return (torch.tensor(xs, dtype=torch.long, device=DEVICE),
            torch.tensor(xmask, dtype=torch.long, device=DEVICE),
            torch.tensor(ys, dtype=torch.long, device=DEVICE),
            torch.tensor(ymask, dtype=torch.bool, device=DEVICE),
            t)

# Pre-train (tiny demo)
config = ModelConfig()
model  = MaskPredictor(V, config).to(DEVICE)
opt    = torch.optim.AdamW(model.parameters(), lr=2e-3)
for step in range(400):
    x, attn, y, m, t = make_pt_batch(batch_size=32, max_len=16)
    loss = masked_xent_loss(model(x, attn), y, m, t)
    opt.zero_grad(); loss.backward(); opt.step()
    if (step+1) % 100 == 0:
        print(f"[PT] step={step+1} loss={loss.item():.3f}")

# SFT (tiny demo)
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
for step in range(200):
    x, attn, y, m, t = make_sft_batch(batch_size=16, max_len=24)
    loss = masked_xent_loss(model(x, attn), y, m, t)
    opt.zero_grad(); loss.backward(); opt.step()
    if (step+1) % 50 == 0:
        print(f"[SFT] step={step+1} loss={loss.item():.3f}")

@torch.no_grad()
def sample_diffusion(prompt_tokens: List[str], out_len=8, steps=12, trace=False):
    prompt_ids = [BOS] + tok.encode(prompt_tokens) + [SEP]
    rt = [MASK]*out_len + [EOS]
    seq = prompt_ids + rt
    attn = torch.tensor([[1]*len(seq)], device=DEVICE)
    seq  = torch.tensor([seq], dtype=torch.long, device=DEVICE)

    resp_start = len(prompt_ids)
    resp_end   = resp_start + out_len

    for k in range(steps, 0, -1):
        t = k/steps; s = (k-1)/steps
        logits = model(seq, attn)[0]
        probs  = F.softmax(logits, dim=-1)
        conf   = torch.ones(seq.shape[1], device=DEVICE)

        for i, tid in enumerate(seq[0]):
            if tid.item() == MASK:
                topv, topi = probs[i].max(dim=-1)
                seq[0,i] = topi; conf[i] = topv

        nun = int(out_len*(1 - s))
        resp_conf = conf[resp_start:resp_end]
        idxs = torch.argsort(resp_conf, descending=True).tolist()
        keep = set(idxs[:nun])
        for j, idx in enumerate(range(resp_start, resp_end)):
            if j not in keep:
                seq[0, idx] = MASK

        if trace:
            decoded = tok.decode(seq[0].tolist()[resp_start:resp_end])
            print(f"step {k:02d} -> nun={nun:02d}  state:", decoded)

    decoded = tok.decode(seq[0].tolist()[resp_start:resp_end])
    out = []
    for w in decoded:
        if w == tok.itos[EOS]: break
        out.append(w)
    return out

@torch.no_grad()
def cond_loglik(prompt_tokens: List[str], resp_tokens: List[str], nmc=64):
    p_ids = [BOS] + tok.encode(prompt_tokens) + [SEP]
    r_ids = tok.encode(resp_tokens) + [EOS]
    L = len(r_ids)
    if L == 0: return 0.0
    total = 0.0
    for _ in range(nmc):
        l = rng.randint(1, L)
        idxs = rng.sample(range(L), l)
        masked = r_ids[:]
        for i in idxs: masked[i] = MASK
        x = torch.tensor([p_ids + masked], dtype=torch.long, device=DEVICE)
        attn = torch.tensor([[1]*x.shape[1]], device=DEVICE)
        logp = F.log_softmax(model(x, attn)[0], dim=-1)
        off = len(p_ids)
        s = sum(logp[off+i, r_ids[i]].item() for i in idxs)
        total += (L / l) * s
    return total / nmc

def choose(prompt, choices):
    scores = [cond_loglik(prompt, c.split()) for c in choices]
    best = max(range(len(choices)), key=lambda i: scores[i])
    return choices[best], scores

if __name__ == "__main__":
    print(">>> sample:", sample_diffusion(
        ["Translate",":","hello","->","Spanish"], out_len=4, steps=10, trace=True))
    print(">>> choose:", choose(
        ["What","is","the","capital","of","France","?"],
        ["Paris","London","Berlin"]))
```

---

## 13) Checks, tips, and exercises

### A) Visualize the reverse process (recommended)

Run `trace=True` in `sample_diffusion` to **print each step**. You’ll see how nun grows from 0 to (L) as (t) decreases, and how low‑confidence tokens are re‑masked—this mirrors **Algorithm 5** and **Fig. 2(c)**. (pp. 3, 21) 

### B) Swap re‑masking strategies

Replace low‑confidence with random re‑masking and compare outputs/accuracy—**Table 9** shows why confidence helps. (p. 25) 

### C) Try block diffusion

Implement a block sampler (left→right blocks with inner diffusion). Compare on math prompts; see **Fig. 4** and **Tables 7–8**. (pp. 24–25) 

### D) Play with the speed/quality dial

Vary `steps` to trade speed vs quality; see **Fig. 5** (throughput curves). (p. 26) 

### E) Likelihood scoring sanity check

For a prompt with obvious correct choice (e.g., France→Paris), verify the **fixed‑(l)** estimator (Eq. (6)) picks the right option with reasonable (nmc) (e.g., 64). (p. 5) 

---

## Recap (what to remember)

* **MLE view** (Eq. (1)) underpins both AR and diffusion. LLaDA is a **principled** generative model because its loss is an **upper bound** on negative log‑likelihood (Eq. (4)/(13)). (pp. 1, 3, 21) 
* **Forward masking** (Eqs. (7–8)) + **time‑free reverse prediction** (Eq. (11))—the model just sees partially masked text. (pp. 19–21) 
* **Critical detail:** the **(1/t)** factor in the loss (Eq. (3)/(12)); this is the key difference from BERT/MaskGIT and what ties LLaDA to MLE. (p. 4) 
* **SFT** = same recipe but **mask only the response** (Eq. (5)); handle EOS as in App. B.1. (pp. 4–5, 22) 
* **Sampling** = fill→re‑mask schedule; **low‑confidence re‑masking** yields better results (Alg. 5; Table 9). (pp. 5, 21, 25) 
* **Likelihood estimator** (Eq. (6)) for scoring answers is **lower variance**. (p. 5; App. A.2) 
* **Any‑order AR link** explains bidirectional strengths (Eq. (15); Sec. 3.3). (pp. 8–9, 21–22) 
* **Results**: LLaDA scales (Fig. 3), competes broadly (Tables 1–2), and shines on reversal reasoning (Table 4). (pp. 6–9) 

---

### Where to look in the paper (fast map)

* **Fig. 2** overview; **Eq. (3–5)** losses; **Eq. (6)** estimator; **Algs. 1–5** training & sampling. (pp. 3–5, 19–21) 
* **Time‑free predictor** (Eq. (11)). (p. 20) 
* **Sampling strategies & ablations** (Fig. 4; Tables 7–10; EOS tip). (pp. 24–25) 
* **Scaling & benchmarks** (Fig. 3; Tables 1–2). (pp. 6–8) 

---

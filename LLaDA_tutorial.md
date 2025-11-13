# LLaDA (Large Language Diffusion Models) — a Code‑First, Beginner‑Friendly Tutorial

**Audience:** You’re good at Python and high‑school math, but you want the **why** and the **how** without the heavy formalism.
**Goal:** Understand LLaDA deeply enough to (a) follow the math, (b) code a toy version, and (c) explain how it differs from GPT‑style autoregressive training.

---

## Table of Contents

1. [Plain‑English overview](#plain-english-overview)
2. [Gentle probability crash course](#gentle-probability-crash-course)
3. [Two families: Autoregressive vs Masked Diffusion](#two-families)
4. [Forward masking process (q_{t\mid0})](#forward-masking)
5. [Reverse process & time‑free mask predictor](#reverse-process)
6. [Main training loss with (1/t) and **why** it’s there](#main-loss)
7. [Log‑likelihood connection (bound)](#likelihood-bound)
8. [Supervised Fine‑Tuning (SFT): prompt→response](#sft)
9. [Sampling (generation): low‑confidence re‑masking](#sampling)
10. [Scoring answers: conditional likelihood estimator](#likelihood-estimator)
11. [Why LLaDA can “think both ways”: Any‑order AR idea](#any-order-ar)
12. [Optional: Classifier‑Free Guidance (CFG)](#cfg)
13. [Code, step‑by‑step](#code-step-by-step)
14. [Full end‑to‑end script (copy‑paste)](#full-script)
15. [Sanity checks, exercises, and troubleshooting](#exercises)

---

## 1) Plain‑English overview <a id="plain-english-overview"></a>

* **GPT‑style (autoregressive):** predict the **next** token given all **previous** tokens. One direction: left → right.

* **LLaDA (masked diffusion):**

  1. **Forward**: randomly **mask** tokens in a sentence with probability (t) (more masking as (t) goes from 0→1).
  2. **Reverse**: a Transformer **mask predictor** sees the partially masked sentence and tries to **fill the blanks**.
  3. Training uses a **special loss**: cross‑entropy **only on masked positions**, scaled by **(1/t)**.
  4. For **SFT** (instruction tuning), keep the **prompt** visible and mask only the **response**.
  5. For **generation**, start with the response fully masked and iteratively “fill then re‑mask low‑confidence bits” until you’re done.

**Analogy:** Imagine a crossword or a sentence with sticky notes covering some words. The model looks at all visible words (both left and right) and fills in the covered ones. If it’s not confident, it covers them again and tries another fill as the “fog” clears.

---

## 2) Gentle probability crash course <a id="gentle-probability-crash-course"></a>

**Generative modeling** wants a model distribution (p_\theta(x)) that matches the true data distribution (p_{\text{data}}(x)). The fundamental training objective is **maximum likelihood**:

$$
\textbf{(Eq. 1)}\quad
\max_\theta; \mathbb{E}*{x\sim p*{\text{data}}}\big[\log p_\theta(x)\big]
;\Longleftrightarrow;
\min_\theta; \mathrm{KL}!\big(p_{\text{data}};|;p_\theta\big).
$$

* **Left side:** maximize average log‑probability your model assigns to real sentences.
* **Right side:** equivalent to minimizing the **KL divergence** (a mismatch measure) from real world to model.

**Logarithms** turn products into sums and stabilize training; **expectation** (\mathbb{E}) means “average over data.”

**Pythonic pseudocode**

```python
for x in dataset:                      # x ~ p_data
    loss = - model.log_prob(x)         # maximize log p  <=>  minimize -log p
    loss.backward()
    opt.step()
```

---

## 3) Two families: Autoregressive vs Masked Diffusion <a id="two-families"></a>

### 3.1 Autoregressive (AR) — GPT world

Any joint (p(x_1,\dots,x_L)) can be **factorized** by the chain rule:

$$
\textbf{(Eq. 2)}\quad
p_\theta(x_1,\dots,x_L) =
p_\theta(x_1)\prod_{i=2}^L p_\theta(x_i \mid x_1,\dots,x_{i-1}).
$$

Train by **next‑token prediction** (cross‑entropy at every position): show the prefix (x_1,\dots,x_{i-1}), predict (x_i).

**Limitation:** at inference, you **must** go left→right. It can be awkward for tasks that benefit from bidirectional context or “reversal” logic.

---

### 3.2 LLaDA — Masked Diffusion

Instead of always predicting the **next** token, LLaDA randomly **masks** tokens and learns to **fill** them. It uses a **forward** process (masking) and a **reverse** process (filling).

**Analogy:** scratch‑off ticket → you gradually scratch (unmask) and write in the numbers you’re confident about.

---

## 4) Forward masking process (q_{t\mid0}) <a id="forward-masking"></a>

Let (x_0 = (x_0^1,\dots,x_0^L)) be a clean sentence, and (M) be the **mask token** (like `[MASK]`).

**Independent masking per token**:

$$
\textbf{(Eq. 7–8)}\quad
q_{t\mid0}(x_t \mid x_0) = \prod_{i=1}^L q_{t\mid0}(x_t^i \mid x_0^i),
$$

with

$$
q_{t\mid0}(x_t^i = x_0^i \mid x_0^i) = 1-t,
\qquad
q_{t\mid0}(x_t^i = M \mid x_0^i) = t,
\quad t\in[0,1].
$$

* At (t=0): nothing is masked.
* At (t=1): **everything** is masked.

**Analogy:** a “fog slider” from 0 (clear) to 1 (opaque). Each word is independently hidden with probability (t).

**Pseudocode**

```python
def forward_mask(x0_tokens, t):
    xt = []
    for tok in x0_tokens:
        if random.random() < t:
            xt.append("[MASK]")
        else:
            xt.append(tok)
    return xt
```

---

## 5) Reverse process & time‑free mask predictor <a id="reverse-process"></a>

The reverse process (conceptually (q_{s\mid t}(x_s\mid x_t))) “unmasks” from heavy mask ((t)) to light mask ((s<t)). Its core is the **data prediction** idea:

> The best prediction at a masked position uses the **visible (unmasked)** tokens; it doesn’t need the exact scalar (t).

Formally:

$$
\textbf{(Eq. 11)}\quad
q_{0\mid t}(x_s^i \mid x_t) = p_{\text{data}}(x_0^i \mid x_t^{UM})
\quad\text{(for positions where }x_t^i=M\text{)},
$$

where (x_t^{UM}) denotes the **unmasked** tokens in (x_t).

**Consequence:** The neural mask predictor can be **time‑free**: a Transformer encoder that takes (x_t) (with some masks) and outputs a distribution for **every** position.

**Analogy:** If I give you a sentence with some words covered, you don’t need to know *when* I covered them; you only use the words you can see.

---

## 6) Main training loss with (1/t) and **why** it’s there <a id="main-loss"></a>

The core loss used to train the mask predictor is:

$$
\textbf{(Eq. 3)}\quad
\mathcal{L}(\theta)
= -\mathbb{E}*{t, x_0, x_t}\left[
\frac{1}{t}\sum*{i=1}^{L}
\mathbf{1}{x_t^i = M},
\log p_\theta(x_0^i \mid x_t)
\right].
$$

* (\mathbf{1}{x_t^i = M}) means “only masked positions contribute to the loss.”
* (\log p_\theta(x_0^i \mid x_t)) is the log‑probability the model assigns to the **correct** original token at position (i), given the current masked sequence (x_t).
* The **(1/t)** factor is crucial.

### Why (1/t)?

On average, a fraction (t) of tokens are masked, i.e. (\mathbb{E}[#\text{masked}]=tL).

* If (t) is small, only a few tokens are masked; without scaling, such examples would contribute tiny loss.
* Dividing by (t) normalizes each example’s contribution (and, as the paper shows, makes the loss a clean surrogate for maximum likelihood).

**Analogy:** If I grade your test but only show you a few questions this time and many questions next time, I should **scale** scores so each test counts comparably.

---

## 7) Log‑likelihood connection (bound) <a id="likelihood-bound"></a>

The loss above isn’t arbitrary; it **upper‑bounds** the true negative log‑likelihood:

$$
\textbf{(Eq. 4)}\quad
-\mathbb{E}*{x_0 \sim p*{\text{data}}}[\log p_\theta(x_0)]
;\le;
\mathcal{L}(\theta).
$$

So minimizing (\mathcal{L}(\theta)) is a principled way to do maximum‑likelihood‑style training.
(BERT/MaskGIT don’t include the (1/t) term, so they don’t get this tight ML connection across mask ratios.)

---

## 8) Supervised Fine‑Tuning (SFT): prompt→response <a id="sft"></a>

For instruction tuning, we model a **conditional** distribution (p_\theta(r_0 \mid p_0)) (response given prompt). The SFT loss mirrors Eq. (3) but masks **only the response**:

$$
\textbf{(Eq. 5)}\quad
-\mathbb{E}*{t, p_0, r_0, r_t}\left[
\frac{1}{t}\sum*{i=1}^{L'}
\mathbf{1}{r_t^i = M},
\log p_\theta(r_0^i \mid p_0, r_t)
\right].
$$

Implementation trick:

* Build sequence `[BOS]  p_0  [SEP]  r_0  [EOS]`.
* For SFT batches, mask **only** positions after `[SEP]`.
* Treat `EOS` as a normal token in training; during generation, you cut off at the first `EOS`.

---

## 9) Sampling (generation): low‑confidence re‑masking <a id="sampling"></a>

**Goal:** Start from a fully masked response and gradually unmask it in a few steps.

**Schedule:** pick (N) steps; define (t_k = k/N) for (k=N,\dots,0). We move from (t) to (s=(k-1)/N).

**At each step:**

1. **Fill** every `[MASK]` position with the model’s **argmax token** and compute its **confidence** (that argmax probability).
2. Decide how many **unmasked** response tokens we want at the new time (s):

[
\text{nun} = \big\lfloor L_{\text{resp}}(1-s)\big\rfloor.
]

3. In the response block, **keep the top‑confidence nun** tokens **unmasked** and set the rest **back to `[MASK]`**.
4. Repeat to the next step (smaller (t)).

This is **low‑confidence re‑masking**: “lock in what you’re sure about; revisit what you’re unsure about.”

**Why it helps:** Compared with random re‑masking, it systematically focuses compute on uncertain tokens.

**Analogy:** Writing in pencil: keep the sentences you’re confident about; erase and rewrite the doubtful ones as you refine.

---

## 10) Scoring answers: conditional likelihood estimator <a id="likelihood-estimator"></a>

For multiple‑choice evaluation, you want (\log p_\theta(r_0 \mid p_0)) for each candidate (r_0). The paper uses a **lower‑variance Monte‑Carlo** estimator:

$$
\textbf{(Eq. 6)}\quad
-\mathbb{E}*{l, r_0, r_l}\left[
\frac{L}{l}
\sum*{i=1}^L \mathbf{1}{r_l^i = M},
\log p_\theta(r_0^i \mid p_0, r_l)
\right],
$$

where:

* (L) = response length,
* (l\in{1,\dots,L}) is uniformly sampled,
* (r_l) is made by masking **exactly (l)** random positions in (r_0) (no replacement).

**Why multiply by (L/l)?** You sample a subset of positions, estimate the sum by scaling the average—classic Monte‑Carlo trick.

---

## 11) Why LLaDA can “think both ways”: Any‑order AR idea <a id="any-order-ar"></a>

Appendix‑level result (you don’t need the full proof): the LLaDA loss equals training an **any‑order autoregressive** model that considers **many permutation orders** of tokens. Intuition:

* AR trains only on **one** direction (left→right).
* LLaDA’s masked objective effectively trains on **many** conditioning patterns (like predicting token (i) from *any* subset of the others).
* That’s why it’s naturally **bidirectional** and often better at “reversal” tasks (e.g., going from “A is B” to “B is A”).

---

## 12) Optional: Classifier‑Free Guidance (CFG) <a id="cfg"></a>

CFG blends a **conditional** model (with the prompt) and an **unconditional** model (prompt masked out). In log‑space:

$$
\log \tilde p = (1+w),\log p_\theta(r_0\mid p_0,r_t) - w,\log p_\theta(r_0\mid m,r_t),
$$

where (w) is a guidance weight and (m) is a masked‑prompt input of the same length.
**Effect:** Often boosts quality; you need two forward passes (cond & uncond) and combine logits.

---

## 13) Code, step‑by‑step <a id="code-step-by-step"></a>

We’ll build a **toy** LLaDA in PyTorch you can run on CPU:

* Tiny tokenizer and datasets
* Forward masking
* Transformer **encoder** mask predictor (no causal mask)
* LLaDA loss with (1/t)
* Pre‑training loop
* SFT loop (mask only response)
* Diffusion sampling with low‑confidence re‑masking
* Conditional likelihood estimator + multiple‑choice helper

> **Note:** To keep code simple, we sample **one** (t) per batch. The paper samples a (t) per example; you can extend it if you like.

---

### 13.1 Setup & tokenizer

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

```python
SPECIAL = ["[PAD]", "[BOS]", "[EOS]", "[SEP]", "[MASK]"]
PAD, BOS, EOS, SEP, MASK = range(len(SPECIAL))

class TinyTokenizer:
    def __init__(self, texts: List[List[str]]):
        words = set(itertools.chain.from_iterable(texts))
        self.itos = SPECIAL + sorted(list(words - set(SPECIAL)))
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, toks: List[str]): return [self.stoi[w] for w in toks]
    def decode(self, ids: List[int]):  return [self.itos[i] for i in ids]
    def __len__(self): return len(self.itos)
```

**Explanation:**
We keep things at **word level** to stay simple. We include a dedicated `[MASK]` token and a `[SEP]` to split prompt/response.

---

### 13.2 Tiny datasets (pre‑train + SFT)

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

**Explanation:**
Just enough data to exercise the pipeline. Real models use subword tokenization and huge corpora.

---

### 13.3 Forward masking function (q_{t\mid0})

```python
def mask_sequence(x_ids: List[int], t: float, allow_indices=None):
    masked, mask_pos = [], []
    for i, tid in enumerate(x_ids):
        can = (allow_indices is None) or (i in allow_indices)
        if can and rng.random() < t:
            masked.append(MASK); mask_pos.append(True)
        else:
            masked.append(tid);   mask_pos.append(False)
    return masked, mask_pos
```

**Explanation:**
We can restrict **which** positions are allowed to be masked (e.g., only the response during SFT).

---

### 13.4 Mask predictor (Transformer encoder)

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
        key_padding_mask = (attn_mask == 0)       # ignore PAD in attention
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)  # no causal mask
        return self.lm_head(h)                    # [B, L, V]
```

**Explanation:**
Encoder (not decoder) → it can “look both ways” (left & right). It outputs a distribution at **every** position (we’ll use only masked ones for loss).

---

### 13.5 LLaDA loss (Eq. 3) with (1/t)

```python
def masked_xent_loss(logits, targets, mask_bool, t_scalar):
    B, L, V = logits.shape
    logits  = logits.reshape(B*L, V)
    targets = targets.reshape(B*L)
    mask    = mask_bool.reshape(B*L)

    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    loss = F.cross_entropy(logits[mask], targets[mask])  # only masked positions
    return loss / max(t_scalar, 1e-6)                    # scale by 1/t
```

**Explanation:**
Exactly matches the math: (\frac{1}{t}\sum_{i:,x_t^i=M}\text{CE}(\cdot)).

---

### 13.6 Pre‑training batch + loop

```python
def make_pt_batch(batch_size=16, max_len=16):
    xs, xmask, ys, ymask = [], [], [], []
    for _ in range(batch_size):
        sent = rng.choice(pretrain_texts)
        ids  = [BOS] + tok.encode(sent) + [EOS]
        L = min(len(ids), max_len)
        ids = ids[:L] + [PAD]*(max_len-L)

        t = max(rng.random(), 1e-3)  # sample t ~ U(0,1]; avoid exactly 0
        valid_positions = [i for i,u in enumerate(ids) if u != PAD]
        masked_ids, mask_pos = mask_sequence(ids, t, allow_indices=valid_positions)

        xs.append(masked_ids); ys.append(ids)
        xmask.append([1 if u != PAD else 0 for u in ids])
        ymask.append(mask_pos)

    return (
        torch.tensor(xs, dtype=torch.long, device=DEVICE),
        torch.tensor(xmask, dtype=torch.long, device=DEVICE),
        torch.tensor(ys, dtype=torch.long, device=DEVICE),
        torch.tensor(ymask, dtype=torch.bool, device=DEVICE),
        t
    )
```

```python
config = ModelConfig()
model  = MaskPredictor(V, config).to(DEVICE)
opt    = torch.optim.AdamW(model.parameters(), lr=2e-3)

for step in range(400):                # tiny toy run
    x, attn, y, m, t = make_pt_batch(batch_size=32, max_len=16)
    logits = model(x, attn)
    loss   = masked_xent_loss(logits, y, m, t)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if (step+1) % 100 == 0:
        print(f"[PT] step={step+1} loss={loss.item():.3f}")
```

**Explanation:**
Samples (t), makes a masked version (x_t), computes the LLaDA loss, updates weights.

---

### 13.7 SFT batch + loop (mask **only** the response)

```python
def make_sft_batch(batch_size=8, max_len=24):
    xs, xmask, ys, ymask = [], [], [], []
    t = max(rng.random(), 1e-3)
    for _ in range(batch_size):
        p, r = rng.choice(sft_pairs)
        full = [BOS] + tok.encode(p) + [SEP] + tok.encode(r) + [EOS]
        L = min(len(full), max_len)
        full = full[:L] + [PAD]*(max_len-L)

        sep_idx   = full.index(SEP) if SEP in full else 0
        resp_idxs = [i for i,u in enumerate(full) if (i > sep_idx and u != PAD)]

        masked_ids, mask_pos = mask_sequence(full, t, allow_indices=resp_idxs)

        xs.append(masked_ids); ys.append(full)
        xmask.append([1 if u != PAD else 0 for u in full])
        ymask.append(mask_pos)

    return (
        torch.tensor(xs, dtype=torch.long, device=DEVICE),
        torch.tensor(xmask, dtype=torch.long, device=DEVICE),
        torch.tensor(ys, dtype=torch.long, device=DEVICE),
        torch.tensor(ymask, dtype=torch.bool, device=DEVICE),
        t
    )
```

```python
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
for step in range(200):
    x, attn, y, m, t = make_sft_batch(batch_size=16, max_len=24)
    logits = model(x, attn)
    loss   = masked_xent_loss(logits, y, m, t)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if (step+1) % 50 == 0:
        print(f"[SFT] step={step+1} loss={loss.item():.3f}")
```

**Explanation:**
Same loss, but masking is **restricted** to the response part (after `[SEP]`).

---

### 13.8 Diffusion sampling (low‑confidence re‑masking)

```python
@torch.no_grad()
def sample_diffusion(prompt_tokens: List[str], out_len=8, steps=12, trace=False):
    prompt_ids = [BOS] + tok.encode(prompt_tokens) + [SEP]
    rt = [MASK]*out_len + [EOS]       # response block + EOS sentinel
    seq = prompt_ids + rt

    attn = torch.tensor([[1]*len(seq)], device=DEVICE)
    seq  = torch.tensor([seq], dtype=torch.long, device=DEVICE)

    resp_start = len(prompt_ids)
    resp_end   = resp_start + out_len

    for k in range(steps, 0, -1):
        t = k/steps
        s = (k-1)/steps

        logits = model(seq, attn)[0]           # [L, V]
        probs  = F.softmax(logits, dim=-1)
        conf   = torch.ones(seq.shape[1], device=DEVICE)

        # Fill masked spots; record confidence
        for i, tid in enumerate(seq[0]):
            if tid.item() == MASK:
                topv, topi = probs[i].max(dim=-1)
                seq[0, i] = topi
                conf[i]   = topv

        # Keep nun = floor(out_len*(1 - s)) best-confident tokens in response
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
            print(f"step {k:02d} nun={nun:02d}  state:", decoded)

    # Decode response up to EOS
    decoded = tok.decode(seq[0].tolist()[resp_start:resp_end])
    out = []
    for w in decoded:
        if w == tok.itos[EOS]: break
        out.append(w)
    return out
```

**Explanation:**
At each step, “fill all masks → keep the most confident subset → re‑mask the rest.” Over steps, fewer are re‑masked, so you **converge**.

---

### 13.9 Conditional likelihood estimator + multiple‑choice helper

```python
@torch.no_grad()
def cond_loglik(prompt_tokens: List[str], resp_tokens: List[str], nmc=64):
    p_ids = [BOS] + tok.encode(prompt_tokens) + [SEP]
    r_ids = tok.encode(resp_tokens) + [EOS]
    L = len(r_ids)
    if L == 0: return 0.0

    total = 0.0
    for _ in range(nmc):
        l = rng.randint(1, L)            # choose how many positions to mask
        idxs = rng.sample(range(L), l)   # choose exactly l unique positions
        masked = r_ids[:]
        for i in idxs:
            masked[i] = MASK

        x    = torch.tensor([p_ids + masked], dtype=torch.long, device=DEVICE)
        attn = torch.tensor([[1]*x.shape[1]], device=DEVICE)
        logp = F.log_softmax(model(x, attn)[0], dim=-1)

        off = len(p_ids)
        s = sum(logp[off+i, r_ids[i]].item() for i in idxs)
        total += (L / l) * s             # scale average by L/l

    return total / nmc

def choose(prompt_tokens: List[str], choices: List[str]):
    scores = [cond_loglik(prompt_tokens, c.split()) for c in choices]
    best_i = max(range(len(choices)), key=lambda i: scores[i])
    return choices[best_i], scores
```

**Explanation:**
Masks exactly (l) positions (lower variance than Bernoulli masking), sums log‑probs on those positions, scales by (L/l), averages over trials.

---

## 14) Full end‑to‑end script (copy‑paste) <a id="full-script"></a>

> If you want *everything in one file*, copy all the snippets in §13 into a single Python file (e.g., `llada_toy.py`) **in the same order**.
> Then add this at the bottom to run a tiny demo:

```python
if __name__ == "__main__":
    # Pre-train
    config = ModelConfig()
    model  = MaskPredictor(V, config).to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for step in range(400):
        x, attn, y, m, t = make_pt_batch(batch_size=32, max_len=16)
        loss = masked_xent_loss(model(x, attn), y, m, t)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step+1) % 100 == 0:
            print(f"[PT] step={step+1} loss={loss.item():.3f}")

    # SFT
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    for step in range(200):
        x, attn, y, m, t = make_sft_batch(batch_size=16, max_len=24)
        loss = masked_xent_loss(model(x, attn), y, m, t)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step+1) % 50 == 0:
            print(f"[SFT] step={step+1} loss={loss.item():.3f}")

    # Sampling demo
    print(">>> sample:",
        sample_diffusion(["Translate",":","hello","->","Spanish"],
                         out_len=4, steps=10, trace=True))

    # Multiple-choice demo
    prompt  = ["What","is","the","capital","of","France","?"]
    choices = ["Paris","London","Berlin"]
    print(">>> choose:", choose(prompt, choices))
```

---

## 15) Sanity checks, exercises, and troubleshooting <a id="exercises"></a>

### A. Sanity checks

* **Training loss goes down** (both phases).
* **Sampling** returns plausible short outputs (e.g., “hola” for the translate demo).
* **Multiple‑choice** picks “Paris” for the France question.

### B. Exercises (to *own* the ideas)

1. **Trace the reverse process:** set `trace=True` and watch how the response evolves at each step—see the mask schedule unfold.
2. **Random re‑masking:** replace low‑confidence with random re‑masking; see why confidence helps.
3. **Per‑example (t):** modify the batch code to sample a **different** (t) for each example and divide each example’s loss by **its** (t).
4. **Block diffusion (semi‑AR):** split the response into fixed‑size blocks; do 2–3 diffusion sub‑steps per block left→right. Compare outputs.
5. **CFG:** implement conditional and unconditional passes and combine logits with a weight (w).
6. **Longer data:** add more sentences and SFT pairs; observe better generations.

### C. Troubleshooting

* If **equations don’t render**, make sure they’re in **Markdown math** blocks: `$...$` or `$$...$$` (not inside triple‑backtick code fences).
* If outputs look random, run more **training steps** or reduce model size to debug.
* If sampling ends too early, consider **down‑weighting EOS** confidence when choosing “top‑confidence” tokens so EOS doesn’t get locked in too soon.

---

### Final mental model (30‑second recap)

* **Goal:** learn (p_\theta) close to (p_{\text{data}}) (Eq. 1).
* **AR:** predict next token from the left.
* **LLaDA:** mask randomly → predict masked tokens from **both sides**.
* **Loss:** masked cross‑entropy **divided by (1/t)** (Eq. 3) → **upper‑bounds** negative log‑likelihood (Eq. 4).
* **SFT:** same idea, but mask **only the response** (Eq. 5).
* **Sampling:** iteratively **fill then re‑mask low‑confidence** tokens until done.
* **Scoring:** use the **fixed‑(l)** estimator (Eq. 6) to compute (\log p_\theta(r_0\mid p_0)).
* **Why it works:** effectively trains on **many conditioning directions**, not just left→right → strong at “reversal” and bidirectional reasoning.

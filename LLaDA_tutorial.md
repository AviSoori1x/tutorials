# LLaDA (Large Language Diffusion Models) — A Code-First, Student-Friendly Tutorial

I’ll walk you through the LLaDA paper step by step, **with equations rendered in Markdown**, lots of **analogies**, and **pythonic pseudocode + PyTorch snippets**.

This is meant so that a smart high-schooler with good Python but shaky probability can follow.

Paper: *Large Language Diffusion Models* (Nie et al., 2025). 

---

## 0. Bird’s-eye view

### What is LLaDA?

Classic LLMs (GPT-style) are **autoregressive (AR)**:

> Predict the next token given all previous tokens (left → right).

LLaDA instead is a **masked diffusion model**:

1. **Forward process**: randomly **mask** tokens in a sentence with probability (t \in [0,1]).
2. **Reverse process**: a Transformer **mask predictor** sees the partially masked sentence and tries to **fill in the masked tokens**.
3. Train this mask predictor with a special loss that:

   * only uses masked positions, and
   * scales by **(1/t)** (this is important).
4. For **SFT** (instruction tuning), only the **response** part is masked, the prompt stays visible.
5. For **generation**, start with the response fully masked and repeatedly:

   * fill in the masks,
   * re-mask low-confidence tokens,
   * move from (t=1) (all masked) down to (t=0) (unmasked).

**Key claim of the paper**:
Most LLM “magic” — scalability, in-context learning, instruction following — comes from **generative modeling principles (maximum likelihood)** + Transformers, *not* specifically from left-to-right AR factorization. 

---

## 1. Generative modeling & maximum likelihood (Eq. (1))

The paper starts with:

```math
\max_\theta \mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)]
\;\Longleftrightarrow\;
\min_\theta \mathrm{KL}\big(p_{\text{data}}(x) \,\|\, p_\theta(x)\big).
```

* (p_{\text{data}}(x)): the **real** distribution over sentences (“how often sentence (x) appears in the world”).
* (p_\theta(x)): the **model**’s distribution (Transformer with weights (\theta)).

### 1.1 Maximum likelihood in simple terms

You have a dataset (x^{(1)}, \dots, x^{(N)}).
Maximum likelihood says: choose (\theta) to make

```math
\frac{1}{N} \sum_{n=1}^N \log p_\theta(x^{(n)})
```

as large as possible.

Equivalently, **minimize negative log-likelihood**:

```python
for x in dataloader:           # each x is a sentence from the dataset
    logp = model.log_prob(x)   # log p_theta(x)
    loss = -logp               # we minimize this
    loss.backward()
    opt.step()
```

This is what “maximum likelihood estimation” (MLE) means.

### 1.2 Why is this equivalent to minimizing KL?

KL divergence:

```math
\mathrm{KL}(p \,\|\, q) = \sum_x p(x) \log \frac{p(x)}{q(x)}.
```

Let (p = p_{\text{data}}), (q = p_\theta). Then:

```math
\mathrm{KL}(p_{\text{data}} \,\|\, p_\theta)
= \sum_x p_{\text{data}}(x)\log p_{\text{data}}(x)
 - \sum_x p_{\text{data}}(x)\log p_\theta(x).
```

* First term = entropy of data (independent of (\theta)).
* Second term = (\mathbb{E}*{p*{\text{data}}}[\log p_\theta(x)]).

So:

```math
\mathrm{KL}(p_{\text{data}} \,\|\, p_\theta)
= \text{constant} - \mathbb{E}_{p_{\text{data}}}[\log p_\theta(x)].
```

Therefore:

* **Minimize KL**
  (\Longleftrightarrow)
* **Maximize (\mathbb{E}*{p*{\text{data}}}[\log p_\theta(x)])**.

That’s Eq. (1).

---

## 2. Autoregressive factorization vs masked diffusion

### 2.1 Autoregressive (Eq. (2)) — the GPT world

Classical factorization:

```math
p_\theta(x_1,\dots,x_L)
= p_\theta(x_1)\prod_{i=2}^L p_\theta(x_i \mid x_1,\dots,x_{i-1}).
```

Train by **next-token prediction**:

* At each position (i), model sees (x_1,\dots,x_{i-1})
* Must predict (x_i).

Pseudocode:

```python
for sentence in dataset:
    for i in range(1, len(sentence)):
        context = sentence[:i]       # prefix
        target  = sentence[i]        # next token
        pred_dist = model(context)   # distribution over vocab
        loss_i = cross_entropy(pred_dist, target)
        total_loss += loss_i
```

**Limitation**: generation is **strictly left→right**. Harder to “fill in the middle” or reason symmetrically (e.g., “B is A” from “A is B”) — see the “reversal curse” discussion in the paper. 

---

### 2.2 LLaDA: masked diffusion instead

LLaDA chooses a **different parameterization**:

> Mask tokens randomly (forward process) and train a model to recover the original tokens (reverse process).

This is a **discrete diffusion model** specialized to masking; the paper calls it a **masked diffusion model (MDM)**. 

Analogy:

* Imagine a **sentence on a whiteboard**.
* You gradually **paint over words** with black squares (masks) as time (t) moves from 0 to 1.
* The reverse process learns how to **reconstruct** the original sentence from partially painted versions.

This is very close in spirit to BERT (masked LM), but with:

* Continuous mask level (t\in[0,1]) instead of fixed 15%.
* A special **(1/t)** weighting in the loss.
* A proper **forward–reverse diffusion formulation** with a likelihood bound.

---

## 3. Forward process (q_{t|0}): mathematically & intuitively

### 3.1 Formal definition (Eqs. (7)–(8))

For a sentence (x_0 = (x_0^1,\dots,x_0^L)), define (x_t) at time (t\in[0,1]) by:

```math
q_{t|0}(x_t \mid x_0)
= \prod_{i=1}^{L} q_{t|0}(x_t^i \mid x_0^i),
```

where

```math
q_{t|0}(x_t^i = x_0^i \mid x_0^i) = 1 - t,
\qquad
q_{t|0}(x_t^i = M \mid x_0^i) = t.
```

* (M) is the special **mask token** (`[MASK]` in our code).
* Each token **independently** survives with prob (1-t) or becomes `[MASK]` with prob (t).
* At (t=0): nothing masked.
  At (t=1): everything masked.

**Analogy**: you have a **fog slider** from 0 to 1.

* At (t=0): scene is clear (all words visible).
* At (t=1): total fog (all words hidden).
* In between: each word is independently fogged with prob (t).

### 3.2 Pythonic pseudocode

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

## 4. Reverse process & the mask predictor

The reverse process moves from **more masked** to **less masked** sequences.

Formally they define a reverse kernel (q_{s|t}(x_s \mid x_t)) (Eqs. (9)–(10)), but the important practical piece is the **data-prediction term** (q_{0|t}(x_s^i \mid x_t)). 

### 4.1 Time-free data prediction (Eq. (11))

They show:

```math
q_{0|t}(x_s^i \mid x_t) = p_{\text{data}}(x_0^i \mid x_t^{UM})
\quad\text{for masked positions},
```

where (x_t^{UM}) is the set of **unmasked** tokens in (x_t).

**Key consequence**:

* The optimal predictor of a masked token only depends on the **visible tokens**, not explicitly on the scalar (t).
* So the **mask predictor** can be **time-free**:

```math
p_\theta(x_0^i \mid x_t)
```

No need to feed (t) into the neural net.

**Analogy**: if I give you a sentence with some words blacked out, and ask you to fill them, you don’t care *how long* I waited before blacking them out—you only care what words are visible.

---

## 5. The training loss (Eq. (3)) and the (1/t) trick

The central training loss (Eq. (3)) is:

```math
\mathcal{L}(\theta)
= -\mathbb{E}_{t,x_0,x_t}\left[
  \frac{1}{t}\sum_{i=1}^L \mathbf{1}\{x_t^i = M\}
  \log p_\theta(x_0^i \mid x_t)
  \right].
```

Pieces:

* (\mathbf{1}{x_t^i = M}): we only compute loss where the token **was masked**.
* (\log p_\theta(x_0^i \mid x_t)): log-probability of the **true original token** at position (i), given the partially masked sentence (x_t).
* The **(1/t)** factor is critical.

### 5.1 Why (1/t)?

On average, a fraction (t) of tokens are masked, so the expected number of masked tokens is (tL).
Rough intuition:

* If (t) is small, only a few tokens are masked; without scaling, those examples would contribute tiny loss.
* Dividing by (t) roughly normalizes so that **each example** contributes similar weight, regardless of (t).

The paper goes further and proves:

```math
-\mathbb{E}_{x_0 \sim p_{\text{data}}}[\log p_\theta(x_0)]
\;\le\;
\mathcal{L}(\theta),
```

i.e. (\mathcal{L}(\theta)) is an **upper bound** on the **negative log-likelihood**, so minimizing (\mathcal{L}) is a **principled MLE-style objective** (Eq. (4)/(13)). 

**Important difference**: BERT / MaskGIT do **not** include this (1/t) term, so they don’t get this tight ML link at all mask ratios.

### 5.2 Our implementation of that loss

```python
def masked_xent_loss(logits, targets, mask_bool, t_scalar):
    """
    logits: [B, L, V] model predictions
    targets: [B, L]     original tokens x0
    mask_bool: [B, L]   True where x_t was MASK
    t_scalar: float     sampled t in (0,1]
    """
    B, L, V = logits.shape
    logits  = logits.reshape(B*L, V)
    targets = targets.reshape(B*L)
    mask    = mask_bool.reshape(B*L)

    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Cross-entropy ONLY on masked tokens
    loss = F.cross_entropy(logits[mask], targets[mask])

    # Scale by 1/t
    return loss / max(t_scalar, 1e-6)
```

That’s Eq. (3) in code.

---

## 6. SFT objective: modeling (p_\theta(r_0 \mid p_0)) (Eq. (5))

For prompt–response pairs ((p_0,r_0)), LLaDA trains:

```math
-\mathbb{E}_{t,p_0,r_0,r_t}\left[
\frac{1}{t}\sum_{i=1}^{L'} \mathbf{1}\{r_t^i=M\}
\log p_\theta(r_0^i \mid p_0, r_t)\right].
```

Key differences versus pre-training:

* The input is the concatenation `[BOS] p_0 [SEP] r_t [EOS]`.
* We **only mask** tokens in the **response** (r_0); the prompt stays intact.
* Everything else (sampling (t), (1/t) scaling, cross-entropy on masked positions) is identical.

In code, we create:

```python
full = [BOS] + tok.encode(prompt) + [SEP] + tok.encode(response) + [EOS]
```

Find indices **after** `[SEP]` and restrict masking to those indices.

---

## 7. Likelihood bound (Eq. (4)/(13)) — what it buys you

The paper proves (using results from Shi et al., Sahoo et al., Ou et al.):

```math
-\mathbb{E}_{x_0 \sim p_{\text{data}}}[\log p_\theta(x_0)] \le \mathcal{L}(\theta).
```

So LLaDA:

* Is a **proper generative model** (not just some heuristic).
* Has a **clear link** to maximum likelihood estimation.
* Improves the log-likelihood as you minimize the masked loss.

This is what justifies scaling to **8B parameters & 2.3T tokens** just like ARMs. 

---

## 8. Supervised fine-tuning: practical details

During SFT, the 8B model:

* Trains on **4.5M prompt–response pairs**.
* Pads shorter pairs with EOS so all sequences in a batch have equal length.
* Treats EOS as a **normal token** during training; removes it during sampling.
* Uses a similar LR schedule but with much smaller learning rates. 

Our toy code mirrors the **mask-only-response** part and EOS padding idea.

---

## 9. Inference (sampling) — pure diffusion with low-confidence re-masking

We discretize the reverse process into (N) steps:

* (t_1 = 1, t_2 = \frac{N-1}{N}, \dots, t_N = \frac{1}{N}, t_0=0).

At sampling:

1. Build the sequence

   ```text
   [BOS] prompt_tokens [SEP] [MASK, MASK, ..., MASK] [EOS]
   ```

   where the response block has length `out_len`.

2. For (k) from (N) down to 1:

   * Let (t = k/N), (s = (k-1)/N).

   * Pass current sequence into model.

   * For each `[MASK]` position:

     * take argmax token and probability (confidence).

   * Compute target “unmasked” count in the response block:

     ```math
     \text{nun} = \lfloor L_{\text{resp}} (1 - s) \rfloor.
     ```

   * Keep the **nun highest-confidence** tokens unmasked; set the rest back to `[MASK]`.

This is exactly **Algorithm 5** (low-confidence re-masking) from the appendix. 

### 9.1 Our implementation

```python
@torch.no_grad()
def sample_diffusion(prompt_tokens, out_len=8, steps=12, trace=False):
    prompt_ids = [BOS] + tok.encode(prompt_tokens) + [SEP]
    rt = [MASK]*out_len + [EOS]         # response block + EOS sentinel
    seq = prompt_ids + rt

    attn = torch.tensor([[1]*len(seq)], device=DEVICE)
    seq  = torch.tensor([seq], dtype=torch.long, device=DEVICE)

    resp_start = len(prompt_ids)
    resp_end   = resp_start + out_len

    for k in range(steps, 0, -1):
        t = k/steps
        s = (k-1)/steps

        logits = model(seq, attn)[0]         # [L, V]
        probs  = F.softmax(logits, dim=-1)
        conf   = torch.ones(seq.shape[1], device=DEVICE)

        # 1) Fill all masks greedily, record confidence
        for i, tid in enumerate(seq[0]):
            if tid.item() == MASK:
                topv, topi = probs[i].max(dim=-1)
                seq[0, i] = topi
                conf[i]   = topv

        # 2) Determine how many tokens in response should stay unmasked
        nun = int(out_len * (1 - s))

        resp_conf = conf[resp_start:resp_end]
        idxs = torch.argsort(resp_conf, descending=True).tolist()
        keep = set(idxs[:nun])   # indexes (within resp block) to keep

        # 3) Re-mask the rest = low-confidence positions
        for j, idx in enumerate(range(resp_start, resp_end)):
            if j not in keep:
                seq[0, idx] = MASK

        if trace:
            decoded = tok.decode(seq[0].tolist()[resp_start:resp_end])
            print(f"step {k:02d} nun={nun:02d}  state:", decoded)

    # Decode final response (up to EOS)
    decoded = tok.decode(seq[0].tolist()[resp_start:resp_end])
    out = []
    for w in decoded:
        if w == tok.itos[EOS]: break
        out.append(w)
    return out
```

Try it:

```python
print(
    ">>> diffusion sample:",
    sample_diffusion(
        ["Translate", ":", "hello", "->", "Spanish"],
        out_len=4,
        steps=10,
        trace=True
    )
)
```

You should see something like `["hola"]` (depending on random seeds and training).

---

## 10. Conditional log-likelihood estimation (Eq. (6))

For tasks like MMLU / ARC-C etc., you have:

* a prompt (p_0),
* several candidate answers (r_0),
* you want to pick the one with highest (\log p_\theta(r_0 \mid p_0)).

Exact (\log p_\theta) is hard to compute in diffusion models. LLaDA uses this lower-variance Monte-Carlo estimator (Eq. (6)):

```math
-\mathbb{E}_{l,r_0,r_l}\left[
\frac{L}{l}
\sum_{i=1}^L \mathbf{1}\{r_l^i = M\}
\log p_\theta(r_0^i \mid p_0, r_l)
\right],
```

where:

* (L) = length of response (r_0),
* (l) is uniformly drawn from ({1,\dots,L}),
* (r_l) is obtained by masking **exactly (l)** random positions in (r_0) (without replacement).

**Intuition analogy**:

* To estimate the total of (L) numbers, you sample (l) of them, take their average, and multiply by (L).
* Here, the “numbers” are the per-position log probabilities.

### 10.1 Implementation

```python
@torch.no_grad()
def cond_loglik(prompt_tokens, resp_tokens, nmc=64):
    p_ids = [BOS] + tok.encode(prompt_tokens) + [SEP]
    r_ids = tok.encode(resp_tokens) + [EOS]
    L = len(r_ids)
    if L == 0: return 0.0

    total = 0.0
    for _ in range(nmc):
        l = rng.randint(1, L)
        idxs = rng.sample(range(L), l)   # positions to mask
        masked = r_ids[:]
        for i in idxs:
            masked[i] = MASK

        x    = torch.tensor([p_ids + masked], dtype=torch.long, device=DEVICE)
        attn = torch.tensor([[1]*x.shape[1]], device=DEVICE)
        logp = F.log_softmax(model(x, attn)[0], dim=-1)

        off = len(p_ids)
        s = 0.0
        for i in idxs:
            s += logp[off+i, r_ids[i]].item()
        total += (L / l) * s

    return total / nmc
```

### 10.2 Multiple-choice helper

```python
def choose(prompt_tokens, choices_strs):
    scores = [cond_loglik(prompt_tokens, c.split()) for c in choices_strs]
    best_i = max(range(len(choices_strs)), key=lambda i: scores[i])
    return choices_strs[best_i], scores
```

Example:

```python
prompt  = ["What","is","the","capital","of","France","?"]
choices = ["Paris", "London", "Berlin"]
print(">>> choose:", choose(prompt, choices))
```

The “Paris” option should get the highest score.

---

## 11. Any-order AR connection — why LLaDA is bidirectional

Appendix A.2 shows that the LLaDA loss is equivalent to training an **any-order autoregressive model (AO-ARM)** that considers **all permutations** of token order. (Eq. (15)) 

AO-ARM loss:

```math
-\mathbb{E}_{x_0, \pi}\left[
\sum_{i=1}^L \log p_\theta\big(x_0^{\pi(i)} \mid x_0^{\pi(<i)}; \pi\big)
\right].
```

This means:

* LLaDA is implicitly learning to predict tokens given **many different conditioning directions**, not just left→right.
* That explains why it works better on **reversal tasks** (e.g., “B is A” given “A is B”) and **reversal poems** than pure left-to-right ARMs; see Table 4 in the paper. 

---

## 12. Classifier-Free Guidance (CFG) (optional performance boost)

LLaDA supports **CFG** (Eq. (16)):

```math
\tilde p_\theta(r_0\mid p_0,r_t) \propto
\frac{p_\theta(r_0\mid p_0,r_t)^{1+w}}{p_\theta(r_0\mid m,r_t)^w},
```

where:

* (m) = masked prompt (same length as (p_0)),
* (w) = CFG weight.

In log-space:

```python
def cfg_logits(logits_cond, logits_uncond, w):
    logp_c = F.log_softmax(logits_cond, dim=-1)
    logp_u = F.log_softmax(logits_uncond, dim=-1)
    return (1 + w) * logp_c - w * logp_u
```

* In practice, you compute logits twice (with prompt vs masked prompt) and combine.
* The paper finds CFG consistently improves performance in ablations (Table 6). 

---

## 13. End-to-end script (ready to run)

I already included an end-to-end script in the previous message. If you want, you can literally copy that script as `llada_toy.py` and run it; it will:

1. Build tokenizer & toy datasets.
2. Train the mask predictor with LLaDA loss on pre-train data.
3. SFT on prompt–response pairs.
4. Run diffusion sampling.
5. Run likelihood-based multiple-choice evaluation.

(If you’d like, I can re-paste it in a separate message; it’s long and we’ve walked through it piece by piece.)

---

## 14. What you should take away

1. **Generative modeling framework**

   * LLaDA is **not** heuristic; it’s grounded in Eq. (1) (MLE) and Eq. (4)/(13) (likelihood bound).

2. **Masked diffusion vs BERT**

   * Forward masking looks BERT-ish, but:

     * (t) is **continuous** in ([0,1]), not fixed (like 0.15).
     * Loss has a crucial **(1/t)** factor.
     * There’s a well-defined **reverse process** used for sampling and a clear bound on log-likelihood.

3. **Bidirectional thinking**

   * The model always sees **both left & right context** when predicting a masked token.
   * AO-ARM equivalence explains strong performance on **reversal** tasks.

4. **SFT is compatible & simple**

   * Just mask the **response** and leave the prompt intact.

5. **Sampling is fill→re-mask**

   * Low-confidence re-masking = “keep what you’re sure about; reconsider what you’re unsure of” across steps.

6. **Likelihood evaluation is possible**

   * Eq. (6)’s fixed-(l) estimator gives stable Monte-Carlo estimates for (\log p_\theta(r_0\mid p_0)), letting you do zero-/few-shot benchmarks the same way you do for ARMs.

7. **Empirical results**

   * LLaDA-8B is competitive with strong ARMs (LLaMA 3 8B) on many benchmarks, and clearly stronger on some reversal reasoning tasks. 

---

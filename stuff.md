````markdown
# Improving an LLM’s Agentic Capabilities with Reinforcement Learning

**Goal:**  
You have an LLM that can *act* (call tools, browse, run code, etc.). You want to use **reinforcement learning (RL)** to make its behavior more **effective, reliable, and aligned** when solving multi-step tasks.

This doc breaks things down into:

1. [Conceptual overview](#0-conceptual-overview-llm-as-an-rl-agent)  
2. [Data you need](#1-data-what-you-need-to-collect)  
3. [Infrastructure you need](#2-infrastructure-what-you-need-to-run-this)  
4. [Tooling you need](#3-tooling-libraries-and-platforms)  
5. [Code skeleton: small end-to-end example](#4-code-skeleton-a-tiny-end-to-end-example)  
6. [A concrete 3-phase implementation plan](#5-a-practical-3-phase-plan)  
7. [Key papers & websites to read next](#6-reading-list--websites)

The style below is meant so you can almost turn it into a design doc or internal wiki page.

---

## 0. Conceptual overview: LLM as an RL agent

Think of your agentic LLM as a standard RL agent:

- **State**: what the agent “sees” right now  
  - User request, chat history, tool outputs, web page DOM, environment metadata, etc. :contentReference[oaicite:0]{index=0}  

- **Action**: what the LLM *decides to do*  
  - “Call tool X with params P”, “click this button”, “run this code”, “answer user now”.

- **Reward**: numeric feedback
  - +1 if task successfully completed (e.g., environment check passes), 0 or negative if not.  
  - Extra shaping: penalty for too many steps, unsafe actions, etc. :contentReference[oaicite:1]{index=1}  

- **Policy**: the thing that chooses actions given states  
  - Your LLM (plus maybe a small extra head) that maps `state → token sequence` describing the action.

Two broad RL setups:

1. **RLHF-style (single-shot)**  
   - Whole answer = one action, reward from humans or a reward model (e.g. InstructGPT pipeline). :contentReference[oaicite:2]{index=2}  

2. **Agentic RL (multi-step)**  
   - Each *tool call / step* is an action; you optimize over full trajectories.  
   - Recent work like **RLTR (Reinforcement Learning with Tool-use Rewards)** explicitly targets *planning/tool use* quality for LLM agents. :contentReference[oaicite:3]{index=3}  

You’re focused on (2): **improving multi-step, tool-using behavior**.

---

## 1. Data: what you need to collect

RL lives and dies on the **data**: trajectories, rewards, and evaluation tasks.

### 1.1 Task & environment definitions

First, define **environments** and **tasks** where the agent can act and you can objectively check success.

Common choices:

- **Web agents**  
  - Use **WebArena**: a self-hostable environment with realistic websites (e-commerce, forums, Git, CMS) and an execution-based evaluation benchmark (“did the string on the page change in the right way?”). :contentReference[oaicite:4]{index=4}  
  - For multimodal agents, use **VisualWebArena**, which evaluates agents on visually-grounded web tasks (screenshots + DOM). :contentReference[oaicite:5]{index=5}  

- **Interactive text / embodied environments**  
  - **BabyAI-Text** and related environments are used in Lamorel’s work on grounding LLMs with online RL. :contentReference[oaicite:6]{index=6}  

- **Tool-using agents (APIs, databases, code)**  
  - Custom environment that exposes tools (search, DB query, code executor) and a checker that validates final state or outputs.

**Data structure per task:**

```jsonc
{
  "task_id": "buy_bluetooth_keyboard_cheapest",
  "natural_language_instruction": "Find the cheapest Bluetooth keyboard and add it to the cart.",
  "initial_state": {
    "url": "https://shop.webarena.dev",
    "dom": "<html>...</html>"
  },
  "success_checker": "python:function_that_checks_cart_for_correct_item"
}
````

You’ll reuse this task spec for:

* Evaluation (how good is a given agent?)
* RL training (defining reward).

---

### 1.2 Trajectories (behavior data)

You need **sequences of interactions**:

```jsonc
{
  "task_id": "buy_bluetooth_keyboard_cheapest",
  "steps": [
    {
      "obs": "...HTML/DOM/state summary...",
      "action": "click(id='search-box')",
      "reward": 0.0
    },
    {
      "obs": "...search results page...",
      "action": "type('bluetooth keyboard')",
      "reward": 0.0
    },
    ...
    {
      "obs": "...cart page...",
      "action": "ANSWER: I've added the cheapest Bluetooth keyboard to your cart.",
      "reward": 1.0,
      "done": true
    }
  ]
}
```

You’ll get trajectories from:

1. **Seed / behavior policy**

   * Humans controlling an agent UI.
   * A strong LLM with a hand-crafted agent prompt (ReAct, plan-and-execute, etc.). ([LangChain Docs][1])

2. **Your RL agent itself**

   * As training proceeds, you keep logging all episodes.

These trajectories feed:

* **Offline RL**: learn from a fixed dataset (no environment in the loop).
* **Online RL**: continually collect new experiences while training (Lamorel-style for LLMs). ([arXiv][2])

---

### 1.3 Reward / feedback data

You need a way to map trajectories → numbers.

#### 1.3.1 Environment-based rewards (automatic)

Best case: the environment itself can say whether you succeeded.

Examples:

* WebArena / VisualWebArena: a “functional correctness” checker tests whether the target text, button, DOM state, or database rows are correct. ([arXiv][3])
* Tool-based tasks: reward = 1 if final API call returns expected JSON, else 0.

You can also add **reward shaping**:

* `reward -= 0.01` per step to encourage shorter trajectories.
* Large negative reward if the agent triggers a safety violation (e.g., tries to run forbidden code).

#### 1.3.2 Human preference / reward model data (RLHF-style)

For “fuzzy” qualities like helpfulness, safety, politeness:

1. Collect **pairwise comparisons**

   * Show annotators: (Prompt, Output A, Output B) → “Which do you prefer?”
   * Tools like **Argilla** and **Label Studio** have templates for RLHF preference collection. ([Argilla][4])

2. Train a **reward model**

   * Input: (prompt, candidate output or trajectory).
   * Output: scalar reward approximating human preference (higher = better). ([arXiv][5])

3. Use that reward model in RL

   * This is the classic **RLHF** pattern used for models like ChatGPT. ([Codingscape][6])

This is critical if your agent must also be **aligned** in how it communicates and makes decisions.

#### 1.3.3 Planning / tool-use rewards

Recent work shows you can improve agentic behavior by **rewarding good plans and tool sequences**, not just final answers:

* **RLTR (Reinforcement Learning with Tool-use Rewards)** explicitly scores the *completeness & correctness of tool invocation sequences* instead of final text answers, improving planning quality by ~8–12% on their benchmarks. ([arXiv][7])

For your agent, you can:

* Reward calling *relevant* tools and penalize pointless loops.
* Reward achieving intermediate subgoals (e.g., “located correct product page”) even before final success.

---

### 1.4 Evaluation datasets & metrics

You need **separate** data purely for evaluation:

* Held-out tasks from WebArena/VisualWebArena or your custom environment. ([arXiv][3])
* Metrics:

  * **Task success rate** (primary).
  * **Path efficiency**: steps per success, tokens per episode, tool calls per episode.
  * **Safety metrics**: number of blocked actions, flagged prompts.
  * **Human eval**: Likert scores (1–5) on helpfulness, clarity, etc.

These give you a way to tell if RL is actually making your agent better instead of just optimizing weird exploits in the reward.

---

## 2. Infrastructure: what you need to run this

### 2.1 Environment runtime

At minimum, you need a service that looks conceptually like:

```python
obs = env.reset(task_id)
obs, reward, done, info = env.step(action)
```

Concrete setups:

* **Web agents**

  * Host **WebArena** or **VisualWebArena** on your infra (Docker + web servers). ([WebArena][8])
  * Drive a headless browser or environment API (Playwright / Selenium / BrowserGym adapter).

* **Tool / API agents**

  * A Python-based environment that exposes tools (search, DB, file system, etc.) and has a `step()` that:

    * Parses the LLM’s output into a tool call.
    * Executes tool.
    * Builds a new observation string or structured state.

* **Code agents**

  * A **sandboxed execution environment** (e.g. Docker/jail) for running Python code, with strict resource & security limits.

### 2.2 Rollout orchestration

You’ll need a **controller** that:

* Spins up many environments in parallel (e.g., via Ray, Kubernetes, or plain multiprocessing).
* Calls your LLM policy in batches.
* Collects trajectories and writes them to a store (S3, parquet, DB).

Projects focusing on **online RL with LLMs** (e.g. Lamorel) show how to do this efficiently: they batch environment interactions and model calls to keep GPUs busy. ([GitHub][9])

### 2.3 Training infrastructure

Depending on the scale:

1. **Small-scale: frozen LLM + small RL head**

   * Base LLM is an API or a fixed open model.
   * You train a **small network** on top that:

     * Chooses actions (which tool to use, when to stop, etc.).
     * Or chooses prompts / routes.

   Infra: a single GPU or even CPU is often enough.

2. **Mid-scale: fine-tune a 1–7B LLM**

   * Use open models with HuggingFace `transformers`.
   * Run PPO / GRPO / DPO with something like **TRL** or **RL4LMs**. ([Hugging Face][10])
   * Hardware: 1–4 decent GPUs (A100/4090 level).

3. **Large-scale: many tens or hundreds of GPUs**

   * Use systems like **DeepSpeed-Chat** to implement SFT → Reward Model → PPO pipeline at scale. ([arXiv][5])

Core stack:

* **PyTorch** or JAX for training.
* **Transformers** for LLMs.
* RL library (TRL, RL4LMs, Lamorel) for algorithms and rollout integration.

### 2.4 Monitoring, logging, and safety

You’ll want:

* **Experiment tracking** (Weights & Biases, MLflow, etc.) for:

  * Reward curves
  * Success rates
  * KL-divergence to base model, etc. ([rl4lms.readthedocs.io][11])

* **Trajectory logging**:

  * Every episode: states, actions, rewards, tool calls, and environment info.
  * Use it later for offline RL or reward model training.

* **Safety and guardrails**:

  * Hard filters for high-risk actions (e.g., file system writes, shell commands).
  * Moderation / policy checks on prompts and actions.
  * Budget limits: max steps, max tokens, max wall-clock time per task.

---

## 3. Tooling: libraries and platforms

### 3.1 RL + LLM training libraries

These give you ready-made PPO/GRPO/DPO loops around LLMs.

#### 3.1.1 TRL (Transformer Reinforcement Learning, HuggingFace)

* Full-stack post-training: SFT, GRPO, PPO, DPO, reward modeling. ([Hugging Face][10])
* Plugs into HuggingFace `transformers`.
* Widely used for RLHF/RLAIF and preference-based fine-tuning.

Docs: HuggingFace TRL documentation. ([Hugging Face][10])

#### 3.1.2 RL4LMs (AllenAI)

* Modular RL library tailored to language models. ([arXiv][5])
* Supports multiple algorithms (PPO, A2C, TRPO, NLPO).
* Comes with **GRUE benchmark** and >2000 experiments across 7 NLP tasks to study RL for text. ([arXiv][5])

Good when you want more algorithmic flexibility.

#### 3.1.3 Lamorel

* A library built explicitly for **online RL with LLMs** in interactive environments (e.g., BabyAI-Text). ([GitHub][9])
* Lets you:

  * Attach value heads.
  * Use PPO/LoRA fine-tuning. ([GitLab][12])

This is especially helpful if you want **true environment-level RL** (not just text-only RLHF).

### 3.2 Agent frameworks & environments

#### 3.2.1 LangChain + LangGraph + Deep Agents

* **LangChain agents**: built-in support for tool-calling agents (LLM + tools + loop). ([LangChain Docs][1])
* **LangGraph**: low-level orchestration framework for building *stateful*, long-running agents as graphs of nodes/edges. Great for multi-step workflows and multi-agent setups. ([LangChain][13])
* **Deep Agents**: library built on LangGraph for complex, planning-capable agents with subagents and file systems. ([LangChain Docs][14])

These frameworks give you a **production-quality agent shell** which you can then improve with RL.

#### 3.2.2 WebArena & VisualWebArena

* **WebArena**: realistic web env with ICML/ICLR-grade benchmark and GitHub repo. ([WebArena][8])
* **VisualWebArena**: benchmark for multimodal web agents, with visual + textual context, and ports like `browsergym.visualwebarena`. ([GitHub][15])

These are almost drop-in RL environments: they already define tasks and success metrics.

### 3.3 Data & feedback collection tools (for RLHF/RLAIF)

#### 3.3.1 Argilla

* Open-source platform specialized for **LLM data**: demos, preferences, evaluation.
* Provides guides and templates for **RLHF workflows**, including comparison data for reward modeling and alternatives like DPO/CoH. ([Argilla][4])

#### 3.3.2 Label Studio

* Open-source data labeling platform, popular for multimodal and RLHF data. ([Label Studio][16])
* Has templates and docs specifically for **human preference collection for RLHF**. ([Label Studio][17])

### 3.4 Experiment tracking & analysis

* **Weights & Biases / MLflow** – for logging runs, metrics, and artifacts.
* Many RLHF tutorials and RL4LMs/TRL docs show how to integrate these tools. ([rl4lms.readthedocs.io][11])

---

## 4. Code skeleton: a tiny end-to-end example

We’ll keep this *very* simple, so you can focus on the structure:

* Environment = simple QA task.
* Agent = “model” function.
* We collect a trajectory of (state, action, reward).

Then you can swap:

* Environment → WebArena / tool environment.
* Policy → real LLM + TRL/RL4LMs PPO trainer.

### 4.1 Transition record

First define a data structure to store a single step:

```python
from dataclasses import dataclass

@dataclass
class Transition:
    obs: str        # Observation (state)
    action: str     # Model's output (e.g. a tool call or final answer)
    reward: float   # Scalar reward
    done: bool      # Episode finished?
    next_obs: str   # Next observation
```

### 4.2 Simple environment

Here the environment just checks if the model says “Paris” when asked about France’s capital.

```python
class SimpleQAEnv:
    def __init__(self, question: str, answer_keyword: str):
        self.question = question
        self.answer_keyword = answer_keyword
        self.done = False

    def reset(self) -> str:
        """Start a new episode and return the initial observation."""
        self.done = False
        # In a real agent, obs might include DOM, tool outputs, etc.
        return f"User: {self.question}"

    def step(self, action_text: str):
        """
        action_text: model's reply as a string.
        Returns: (next_obs, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode already finished")

        # Reward: 1 if answer contains keyword, else 0
        reward = 1.0 if self.answer_keyword.lower() in action_text.lower() else 0.0
        self.done = True

        next_obs = "Episode finished."
        info = {"correct": bool(reward)}
        return next_obs, reward, self.done, info
```

In your real system:

* `step()` would parse **tool calls**, run them, and return tool outputs as part of `next_obs`.
* `reward` would come from environment checks, reward models, and safety filters.

### 4.3 Running one episode

Now we define how to run a full episode and collect transitions:

```python
def generate_reply(model, prompt: str) -> str:
    """
    Placeholder: call your LLM here.
    - For a local HuggingFace model, you'd use a text-generation pipeline.
    - For an API, you'd send an HTTP request.
    """
    return model(prompt)  # Here `model` is a callable for simplicity


def run_episode(model, env: SimpleQAEnv):
    obs = env.reset()
    transitions = []
    done = False

    while not done:
        # Build the prompt from the observation
        prompt = obs + "\nAssistant:"

        # Ask the LLM to act
        action = generate_reply(model, prompt)

        # Apply the action in the environment
        next_obs, reward, done, info = env.step(action)

        # Save the transition
        transitions.append(
            Transition(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                next_obs=next_obs,
            )
        )

        # Move to next state
        obs = next_obs

    return transitions
```

### 4.4 Final toy script

Putting it all together:

```python
from dataclasses import dataclass

@dataclass
class Transition:
    obs: str
    action: str
    reward: float
    done: bool
    next_obs: str


class SimpleQAEnv:
    def __init__(self, question: str, answer_keyword: str):
        self.question = question
        self.answer_keyword = answer_keyword
        self.done = False

    def reset(self) -> str:
        """Return initial observation."""
        self.done = False
        return f"User: {self.question}"

    def step(self, action_text: str):
        """Apply action_text and return next state + reward."""
        if self.done:
            raise RuntimeError("Episode already finished")

        reward = 1.0 if self.answer_keyword.lower() in action_text.lower() else 0.0
        self.done = True

        next_obs = "Episode finished."
        info = {"correct": bool(reward)}
        return next_obs, reward, self.done, info


def generate_reply(model, prompt: str) -> str:
    """Stub: call your LLM here."""
    return model(prompt)


def run_episode(model, env: SimpleQAEnv):
    obs = env.reset()
    transitions = []
    done = False

    while not done:
        prompt = obs + "\nAssistant:"
        action = generate_reply(model, prompt)

        next_obs, reward, done, info = env.step(action)

        transitions.append(
            Transition(obs=obs, action=action, reward=reward,
                       done=done, next_obs=next_obs)
        )

        obs = next_obs

    return transitions


if __name__ == "__main__":
    # Dummy model: always answers correctly
    def echo_model(prompt: str) -> str:
        return "The capital of France is Paris."

    env = SimpleQAEnv("What is the capital of France?", "Paris")
    traj = run_episode(echo_model, env)

    for t in traj:
        print(t)
```

To turn this into a *real* RL setup:

* Replace `SimpleQAEnv` with WebArena / VisualWebArena or your agent environment.
* Replace `echo_model` with a real LLM.
* Feed `traj` into PPO or GRPO from TRL / RL4LMs.

---

## 5. A practical 3-phase plan

Here’s a concrete roadmap for improving an LLM agent with RL.

### Phase 1 – Stand up the environment & agent (no RL yet)

1. **Pick 1–2 environments**:

   * WebAgent: WebArena (text-only) or VisualWebArena (multimodal). ([arXiv][3])

2. **Wrap them in a simple Python API**:

   ```python
   obs = env.reset(task_id)
   obs, reward, done, info = env.step(action)
   ```

3. **Build a baseline agent**:

   * Use LangChain + LangGraph or a simple loop: “observe → LLM → parse tool call → step”. ([LangChain][13])

4. **Log everything**:

   * States, actions, rewards, tool calls, token counts.

5. **Evaluate baseline**:

   * Flat success rate over ~100–1000 tasks.

### Phase 2 – RL for better plans & actions

1. **Define reward functions**:

   * `+1` on success, small step penalty, big penalty for violations.
   * Add planning/tool-use rewards (à la RLTR): reward good tool sequences even if the final answer isn’t fully verifiable. ([arXiv][7])

2. **Collect seed trajectories**:

   * Use your baseline agent to generate thousands of episodes.
   * Optionally, add expert demonstrations (hand-crafted trajectories).

3. **Hook up RL library**:

   * For mid-scale models: TRL or RL4LMs with PPO/GRPO:

     * Policy model: your LLM
     * Value head: added on top of final hidden state
     * Reward: environment + optional reward model. ([Hugging Face][10])

4. **Train & monitor**:

   * Track:

     * Average episode reward
     * Success rate on a development set
     * KL divergence to base policy (avoid drifting too far).

5. **Iterate on reward & environment**:

   * If you see reward hacking (e.g., short weird trajectories), tighten rewards and add constraints.

### Phase 3 – RLHF/RLAIF for alignment & communication quality

1. **Define RLHF tasks**:

   * Which outputs should be “more helpful / safe / honest”? ([Codingscape][6])

2. **Collect pairs or rankings**:

   * Use Argilla or Label Studio to collect human preferences over agent outputs. ([Argilla][4])

3. **Train a reward model**:

   * Input: `(prompt, environment context, output)`
   * Output: scalar “preference score”.

4. **Run RLHF or DPO**:

   * Either full RL (PPO/GRPO) or offline preference-based methods like DPO for safer, easier tuning. ([arXiv][5])

5. **Combine with environment-level RL**:

   * Reward = `α * environment_reward + β * alignment_reward`.
   * Adjust α, β to trade off task success vs. aligned behavior.

---

## 6. Reading list & websites

Here are highly relevant resources, grouped:

### Agentic RL & planning

* **Encouraging Good Processes Without the Need for Good Answers: Reinforcement Learning for LLM Agent Planning (RLTR)** – Li et al., 2025.

  * RL that rewards *planning/tool use* quality rather than final text, improves planning by ~8–12%. ([arXiv][7])

* **Grounding Large Language Models with Online Reinforcement Learning** & the Lamorel GitHub.

  * Shows how to use online RL with LLMs in BabyAI-Text. ([GitHub][18])

### RL for language models (general)

* **Is Reinforcement Learning (Not) for Natural Language Processing?** – RL4LMs paper + library.

  * Benchmarks, baselines, and building blocks; shows RL often beats supervised methods for aligning LMs. ([arXiv][5])

* **TRL (Transformer Reinforcement Learning)** – HuggingFace docs & GitHub.

  * Full stack for SFT, reward modeling, PPO/GRPO, DPO. ([Hugging Face][10])

### Environments & agent frameworks

* **WebArena** – environment + benchmark for web agents. Website + paper + GitHub. ([WebArena][8])
* **VisualWebArena** – multimodal benchmark for web agents. ([GitHub][15])
* **LangChain agents & LangGraph** – docs and GitHub for building complex, long-running agents. ([LangChain Docs][1])

### Human feedback data & RLHF

* **Argilla RLHF docs & blog** – practical guides to collecting RLHF data and training reward models; also covers alternatives like DPO and CoH. ([Argilla][4])
* **Label Studio** – RLHF human preference templates and LLM integration. ([Label Studio][16])

---

If you tell me *which* agent you’re building (web, code, RAG, etc.), I can turn this into a concrete design: “here’s your env, here’s your reward, here’s the exact TRL config, here’s how to log and evaluate.”

```
::contentReference[oaicite:51]{index=51}
```

[1]: https://docs.langchain.com/oss/python/langchain/agents?utm_source=chatgpt.com "Agents - Docs by LangChain"
[2]: https://arxiv.org/html/2302.02662v4?utm_source=chatgpt.com "Grounding Large Language Models in Interactive ..."
[3]: https://arxiv.org/abs/2307.13854?utm_source=chatgpt.com "WebArena: A Realistic Web Environment for Building Autonomous Agents"
[4]: https://docs.v1.argilla.io/en/v1.21.0/conceptual_guides/llm/rlhf.html?utm_source=chatgpt.com "Collecting RLHF data - Argilla 1.21 documentation"
[5]: https://arxiv.org/abs/2210.01241?utm_source=chatgpt.com "Is Reinforcement Learning (Not) for Natural Language Processing: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization"
[6]: https://codingscape.com/blog/what-is-rlhf-reinforcement-learning-from-human-feedback?utm_source=chatgpt.com "What is RLHF? (Reinforcement Learning from Human ..."
[7]: https://arxiv.org/abs/2508.19598?utm_source=chatgpt.com "Encouraging Good Processes Without the Need for Good Answers: Reinforcement Learning for LLM Agent Planning"
[8]: https://webarena.dev/?utm_source=chatgpt.com "WebArena: A Realistic Web Environment for Building ..."
[9]: https://github.com/flowersteam/lamorel?utm_source=chatgpt.com "Language Models for Reinforcement Learning - Lamorel"
[10]: https://huggingface.co/docs/trl/en/index?utm_source=chatgpt.com "TRL - Transformer Reinforcement Learning"
[11]: https://rl4lms.readthedocs.io/en/latest/index.html?utm_source=chatgpt.com "Welcome to RL4LMs's documentation! — RL4LMs 0.0.1 ..."
[12]: https://gitlab.mn.tu-dresden.de/lest161c/lamorel/-/tree/fbb8f63bda3728c031a66cf211c5b3216f354553/examples/PPO_LoRA_finetuning?utm_source=chatgpt.com "examples/PPO_LoRA_finetuning - lamorel - GitLab"
[13]: https://www.langchain.com/langgraph?utm_source=chatgpt.com "LangGraph"
[14]: https://docs.langchain.com/oss/python/deepagents/overview?utm_source=chatgpt.com "Deep Agents overview - Docs by LangChain"
[15]: https://github.com/web-arena-x/visualwebarena?utm_source=chatgpt.com "VisualWebArena is a benchmark for multimodal agents."
[16]: https://labelstud.io/?utm_source=chatgpt.com "Label Studio: Open Source Data Labeling"
[17]: https://labelstud.io/blog/data-labeling-and-comparative-analysis-of-fine-tuning-methods/?utm_source=chatgpt.com "Data Labeling and Comparative Analysis of Fine-Tuning ..."
[18]: https://github.com/flowersteam/Grounding_LLMs_with_online_RL?utm_source=chatgpt.com "Grounding Large Language Models with Online ..."

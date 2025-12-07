# 📘 Deep Reinforcement Learning – A Complete Course (Beginner → Advanced)

### *By Your Professor of Deep Learning 🧠*

I’ll build you **the clearest, most beginner-friendly but complete reinforcement learning course** for a high-school student who:

* ✅ Is good at Python
* ✅ Is comfortable with basic algebra
* 🤏 Is less confident with probability/statistics (I’ll go slow there)

We will always aim for:

* Simple explanations
* **PyTorch** as much as possible
* Step-by-step code snippets
* Visualizable bandits and gridworlds
* A full end-to-end script at the end of each chapter
* No skipping of intuition

This will be a **multi-chapter series**.
You can say **“Next Chapter”** anytime to move on.

---

# 🏗️ Course Outline

## **CHAPTER 1 — The RL Mindset: Agents, States, Actions, Rewards**

* Bandit examples (slot machines)
* Why reward matters
* Why we can’t brute-force all possibilities
* Intro Gym-style environments (step, reset)
* A tiny RL loop implemented from scratch (tabular)
* (Bonus) Tiny PyTorch flavor

## **CHAPTER 2 — Value Functions & Bellman Equation**

* State value (V(s)), action value (Q(s,a)), policy (\pi(a\mid s))
* Discount factor (\gamma) explained visually
* Bellman expectation vs optimality equations
* Implementing value iteration in Python
* A tiny gridworld solved with value iteration
* PyTorch implementation

## **CHAPTER 3 — Tabular Q-Learning**

* Why (Q(s,a)) is more powerful than (V(s))
* Temporal Difference (TD) learning intuition
* Q-learning update derived step by step
* Implement tabular Q-learning on a gridworld
* Plotting training curves

## **CHAPTER 4 — Deep Q-Networks (DQN)**

* Replace Q-table with a neural network
* PyTorch implementation of the Q-network
* Replay buffer
* Target network
* Full CartPole DQN from scratch

## **CHAPTER 5 — Improvements to DQN**

* Double DQN
* Dueling DQN
* Prioritized Replay
* Noisy Nets
* Rainbow DQN overview
* Implement Double + Dueling in PyTorch

## **CHAPTER 6 — Monte-Carlo Tree Search (MCTS)**

* Upper Confidence Bound (UCB) intuition
* Why search + learning works (AlphaZero)
* Tiny MCTS on Tic-Tac-Toe

## **CHAPTER 7 — From Q-Learning to Policies**

* Why softmax(Q) can fail
* Policy gradient intuition
* Deriving REINFORCE
* Implement REINFORCE on CartPole

## **CHAPTER 8 — Actor-Critic**

* Value function as a learned baseline
* Advantage and TD error
* Implement A2C (Advantage Actor-Critic)

## **CHAPTER 9 — Proximal Policy Optimization (PPO)**

* Why TRPO is too complex
* PPO clipping objective explained
* Full PPO implementation (discrete actions)

## **CHAPTER 10 — Advanced Concepts**

* GAE (Generalized Advantage Estimation)
* Entropy regularization
* Multi-step returns
* Off-policy vs on-policy
* Continuous action spaces
* PPO in continuous action spaces

---

# 📘 CHAPTER 1: The Reinforcement Learning Mindset

> **Core question:** *What does it mean for an agent to “learn” from experience?*

---

## 1.1 🧠 What is Reinforcement Learning?

Reinforcement Learning (RL) is about an **agent** learning to take **actions** in an **environment** to maximize **reward** over time.

Think:

> A robot, a game-playing AI, or a trading bot repeatedly interacts with the world and gets feedback.

We describe RL using standard symbols:

| Symbol              | Name       | Intuition                                 |
| ------------------- | ---------- | ----------------------------------------- |
| **s**               | state      | “Where you are right now”                 |
| **a**               | action     | “What you choose to do”                   |
| **r**               | reward     | “How good/bad was that action?”           |
| **s′**              | next state | “Where you end up after the action”       |
| **(\pi(a \mid s))** | policy     | “Given state s, what action do you pick?” |

### 🔍 About (\pi(a \mid s)) (fixed typo)

You noticed `π(a  s)` looked wrong. Yes, that was a formatting typo.

The correct thing is:

[
\pi(a \mid s)
]

Read it as: **“the probability of choosing action a given you are in state s.”**

* If the policy is **stochastic**, you get a probability distribution over actions.
* If the policy is **greedy**, you might always pick the same best action.

---

## 1.2 🎰 The simplest RL problem: a multi-armed bandit

Imagine a very simple “world”:

* You are in front of **3 buttons** (3 “arms” of a slot machine).
* Each button gives a reward of **1** with some unknown probability:

| Button | Hidden chance of reward 1 |
| ------ | ------------------------- |
| A      | 10%                       |
| B      | 50%                       |
| C      | 80%                       |

You don’t know these probabilities — you only see 0 or 1 as you press.

Your goal:

> Learn which button gives the **highest expected reward**.

But there is a dilemma:

* **Exploration**: Try all buttons to gather information.
* **Exploitation**: Press the button that *seems* best so far.

If you **only exploit**, you might get stuck on a button that **looks** good early but is actually worse.
If you **only explore**, you never really “use” what you learned.

---

## 1.3 🤓 Coding the bandit (first version)

We start with simple Python, no PyTorch yet.

```python
import random

# True reward probabilities (the agent does NOT know this)
probs = [0.1, 0.5, 0.8]

def pull_arm(arm):
    """
    Simulate pulling one of the arms.
    arm: 0, 1, or 2.
    Return: 1 (reward) or 0 (no reward)
    """
    p = probs[arm]
    # random.random() gives a float in [0,1)
    if random.random() < p:
        return 1
    else:
        return 0
```

Let’s have the agent **just randomly** press buttons:

```python
rewards = []

for step in range(10):
    action = random.randint(0, 2)  # choose 0, 1, or 2
    reward = pull_arm(action)
    rewards.append(reward)
    print(f"step={step}, action={action}, reward={reward}")

avg_reward = sum(rewards) / len(rewards)
print("average reward:", avg_reward)
```

This agent:

* Does **no learning**.
* Takes **random actions forever**.

RL is about learning from experience, so we need to estimate which arm is better.

---

## 1.4 📈 Estimating action values (the Q array)

We want to estimate:

> (Q(a) \approx) “average reward if I choose action a”

We’ll have:

* `Q[a]` = current estimate of the value of arm a
* `N[a]` = how many times we used arm a

### Code:

```python
import numpy as np
import random

Q = np.zeros(3)   # estimated values for arms 0,1,2
N = np.zeros(3)   # how many times each arm was pulled

for step in range(2000):
    action = random.randint(0, 2)   # still random for now
    reward = pull_arm(action)

    N[action] += 1

    # ---- The incremental mean update (important) ----
    Q[action] += (reward - Q[action]) / N[action]
```

You asked:

> I don't understand this:
> `Q[action] += (reward - Q[action]) / N[action]`

Let’s unpack it **slowly**.

### 1.4.1 🧮 Incremental mean — intuition

Suppose you have seen this arm **N** times so far, with an average of **old_Q**.

Now you get a new result: **reward** (either 0 or 1).

The new average should be:

[
\text{new_Q} = \frac{\text{sum of all rewards so far}}{\text{number of trials}}
]

Let:

* sum of previous rewards = (S_{\text{old}})
* old average = (Q_{\text{old}} = \frac{S_{\text{old}}}{N_{\text{old}}})
* new count = (N_{\text{new}} = N_{\text{old}} + 1)
* new sum = (S_{\text{new}} = S_{\text{old}} + \text{reward})

Then:

[
Q_{\text{new}} = \frac{S_{\text{new}}}{N_{\text{new}}}
= \frac{S_{\text{old}} + \text{reward}}{N_{\text{old}} + 1}
]

We can rewrite this in the classic “old + step × error” form:

[
Q_{\text{new}} = Q_{\text{old}} + \frac{1}{N_{\text{new}}} (\text{reward} - Q_{\text{old}})
]

This is exactly:

```python
alpha = 1 / N[action]
Q[action] = Q[action] + alpha * (reward - Q[action])
```

Or shorter:

```python
Q[action] += (reward - Q[action]) / N[action]
```

### 1.4.2 🔢 Small numerical example

Say for arm 2:

* Previously, `N[2] = 4`
* Previously, `Q[2] = 0.5` → “average reward 0.5 so far”
* New `reward = 1`
* New `N[2] = 5`
* Then:

```python
alpha = 1 / 5 = 0.2
error = reward - Q[2] = 1 - 0.5 = 0.5
Q_new = 0.5 + 0.2 * 0.5 = 0.5 + 0.1 = 0.6
```

So now:

* New estimate `Q[2] = 0.6` (slightly higher)
* Because we just saw a reward **above** our expectation

If the reward had been 0:

```python
error = 0 - 0.5 = -0.5
Q_new = 0.5 + 0.2 * (-0.5) = 0.5 - 0.1 = 0.4
```

Estimate goes down.

This pattern:

> **new estimate = old estimate + step_size × (target - old estimate)**

shows up **everywhere** in RL (temporal difference, Q-learning, etc.).

---

## 1.5 🎲 Exploration vs Exploitation: ε-greedy

Now we stop being totally random.

We want to:

* **Usually** choose the arm with the highest `Q`
* **Sometimes** explore a random arm

This is the **ε-greedy** strategy:

* With probability `ε` (“epsilon”) → choose a random action (explore)
* With probability `1 - ε` → choose best-known action (exploit)

```python
epsilon = 0.1  # 10% of the time, choose randomly

if random.random() < epsilon:
    # EXPLORE: random choice
    action = random.randint(0, 2)
else:
    # EXPLOIT: choose best Q
    action = int(np.argmax(Q))
```

This simple trick is used in many RL algorithms.

---

## 1.6 🧩 Tiny bandit RL loop (full code)

Here is a complete example combining:

* Q estimation
* ε-greedy exploration
* Incremental mean

```python
import numpy as np
import random

# Environment: 3-armed bandit
probs = [0.1, 0.5, 0.8]

def pull_arm(arm):
    return 1 if random.random() < probs[arm] else 0

# Agent's estimates
Q = np.zeros(3)
N = np.zeros(3)
epsilon = 0.1

for step in range(2000):
    # Choose action (ε-greedy)
    if random.random() < epsilon:
        action = random.randint(0, 2)       # explore
    else:
        action = int(np.argmax(Q))          # exploit

    # Take action, observe reward
    reward = pull_arm(action)

    # Update counts and Q-value
    N[action] += 1
    Q[action] += (reward - Q[action]) / N[action]

print("Estimated Q-values:", Q)
print("True probabilities:", probs)
```

Over many steps, `Q` should get close to `[0.1, 0.5, 0.8]`.

---

## 1.7 🌍 From bandits to environments (Gym-style)

Real RL problems (like CartPole in Gym) have:

* A **state** (like a vector: cart position, pole angle, etc.)
* A **step** function: given an action, returns:

  * next state
  * reward
  * done (whether the episode ended)
  * info (extra debug info)

Standard interface:

```python
next_state, reward, done, info = env.step(action)
```

And a `reset()` function to start a new episode:

```python
state = env.reset()
```

We’ll build a **toy environment** to mimic this.

---

## 1.8 🎮 Building a simple 1D environment

You noticed:

> “Also in here:
> `def step(self, action):`
> `# actions: 0 = left, 1 = right`
> `if action == 1: self.state += 1`
> `else: self.state -= 1`
>
> if 0 is left by subtract in the else statement.”

Yep, that can be confusing. Let’s rewrite it explicitly so there is **no ambiguity**.

We’ll define:

* **Action 0 → move left → state -= 1**
* **Action 1 → move right → state += 1**

Goal: reach state = +5. When you reach 5, you get reward 1 and episode ends.

```python
class SimpleEnv:
    def __init__(self):
        # The state is just an integer position on a 1D line.
        self.state = 0

    def reset(self):
        """
        Start a new episode by putting the agent back at position 0.
        """
        self.state = 0
        return self.state

    def step(self, action):
        """
        action: 0 = left, 1 = right
        """
        if action == 0:
            # move left
            self.state -= 1
        elif action == 1:
            # move right
            self.state += 1
        else:
            raise ValueError("Invalid action, must be 0 or 1")

        # Reward: +1 only when we reach position +5
        if self.state == 5:
            reward = 1
            done = True   # episode ends at the goal
        else:
            reward = 0
            done = False

        # info is usually a dict with debug info; we leave it empty
        info = {}

        return self.state, reward, done, info
```

Now the mapping is clear:

* `action == 0` → `state -= 1` (left)
* `action == 1` → `state += 1` (right)

### Let’s run a random episode

```python
import random

env = SimpleEnv()
state = env.reset()

for t in range(20):
    action = random.randint(0, 1)  # 0 or 1
    next_state, reward, done, _ = env.step(action)
    print(f"t={t}, state={state}, action={action}, reward={reward}, next_state={next_state}")
    state = next_state
    if done:
        print("Reached the goal!")
        break
```

This shows:

* How state changes with action.
* How reward appears only at the goal.
* How `done` tells us the episode is over.

---

## 1.9 🔥 Learning in this environment (tabular Q)

Now we want to **learn a policy**:

> “From each position, should I go left or right?”

We’ll keep a Q-table:

* States: positions from -5 to +5 → 11 possible states
* Actions: 0 (left), 1 (right)

So `Q` can be a 2D array: shape `[11, 2]`.

We’ll reuse the **incremental mean**-style update (no full TD yet, that’s Chapter 2/3).

### Step 1: set up environment and Q-table

```python
import numpy as np
import random

class SimpleEnv:
    def __init__(self):
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if action == 0:     # left
            self.state -= 1
        elif action == 1:   # right
            self.state += 1
        else:
            raise ValueError("Invalid action")

        reward = 1 if self.state == 5 else 0
        done = (self.state == 5)
        return self.state, reward, done, {}

# Q[state_index, action]
Q = np.zeros((11, 2))    # states -5..+5 mapped to indices 0..10
N = np.zeros((11, 2))    # visit counts
epsilon = 0.2            # exploration rate

env = SimpleEnv()

def state_to_index(s):
    # map state -5..+5 → index 0..10
    return s + 5
```

### Step 2: training loop over episodes

```python
num_episodes = 200

for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        s_idx = state_to_index(state)

        # ε-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 1)               # explore
        else:
            action = int(np.argmax(Q[s_idx]))           # exploit

        next_state, reward, done, _ = env.step(action)
        next_s_idx = state_to_index(next_state)

        # Update Q using incremental mean of immediate reward
        N[s_idx, action] += 1
        alpha = 1.0 / N[s_idx, action]
        # NOTE: for now we only learn from immediate reward (bandit-style update)
        Q[s_idx, action] += alpha * (reward - Q[s_idx, action])

        state = next_state

print("Learned Q-table:")
print(Q)
```

This is **not yet full Q-learning** (we’re not bootstrapping from next state), but:

* We have episodes.
* We have a Gym-like environment.
* We have ε-greedy exploration.
* We have tabular Q-values per state-action.
* We are updating Q-values online.

---

## 1.10 ✨ Tiny PyTorch flavor (optional preview)

Since you said: **“Please use PyTorch ALWAYS”**, here’s a small twist:
We’ll keep `Q` as a PyTorch tensor instead of a NumPy array.

This doesn’t buy us much yet, but it gets you used to the pattern.

```python
import torch
import random

# Q-table as a torch tensor
Q = torch.zeros((11, 2), dtype=torch.float32)
N = torch.zeros((11, 2), dtype=torch.float32)

epsilon = 0.2
env = SimpleEnv()

def state_to_index(s):
    return s + 5

num_episodes = 200

for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        s_idx = state_to_index(state)

        # ε-greedy with torch
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            # torch.argmax returns a tensor; we convert to int
            action = int(torch.argmax(Q[s_idx]))

        next_state, reward, done, _ = env.step(action)

        # Convert reward to float
        r = float(reward)

        # Update Q using incremental mean
        N[s_idx, action] += 1.0
        alpha = 1.0 / N[s_idx, action]
        Q[s_idx, action] += alpha * (r - Q[s_idx, action])

        state = next_state

print("Learned Q-table (PyTorch):")
print(Q)
```

Later, we’ll **replace this Q-table with a neural network** (that’s Chapter 4 and on).

---

## ✅ Summary of Chapter 1

You now understand:

* What **state (s)**, **action (a)**, **reward (r)**, and **policy (\pi(a \mid s))** mean.
* What a **bandit** is (no states, just actions).
* How to estimate **action values Q(a)** using incremental means.
* What **ε-greedy** exploration is.
* How a **Gym-like environment** works: `reset()` and `step(action)`.
* How to write a tiny RL loop in Python (and a PyTorch version of the Q-table).


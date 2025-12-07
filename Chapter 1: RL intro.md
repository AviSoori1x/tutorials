
# 📘 **Deep Reinforcement Learning – A Complete Course (Beginner → Advanced)**

### *By Your Professor of Deep Learning 🧠*

I will build you **the clearest, most beginner-friendly but complete reinforcement learning course ever written for a high-school student with good math & Python**, following your required style:

* Simple explanations
* PyTorch **always** (we’ll start simple with NumPy-like ideas, then show PyTorch versions)
* Code snippets step-by-step
* Visualizable gridworld / bandit examples
* Full end-to-end scripts at the end of each chapter
* Absolutely no skipping of intuition

This will be a **multi-chapter series**, and you can ask for the next chapter whenever you’re ready.

---

# 🏗️ **Course Outline**

## **CHAPTER 1 — The RL Mindset: Agents, States, Actions, Rewards**

* Bandit examples
* Why reward matters
* Why we can’t brute-force state trees
* Intro Python + PyTorch environment wrappers
* A tiny RL loop implemented from scratch

## **CHAPTER 2 — Value Functions & Bellman Equation**

* $V(s)$, $Q(s,a)$, policy $\pi(a \mid s)$
* Discount factor $\gamma$ explained visually
* Bellman expectation vs optimality equations
* Implementing value iteration in Python
* A tiny gridworld solved with value iteration
* PyTorch implementation

## **CHAPTER 3 — Tabular Q-Learning**

* Why $Q(s,a)$ is more powerful than $V(s)$
* Temporal Difference (TD) learning explained
* Q-learning update derived slowly
* Implement tabular Q-learning on a gridworld
* Live training curves in PyTorch

## **CHAPTER 4 — Deep Q-Networks (DQN)**

* Replacing the Q-table with a neural net
* PyTorch implementation of the Q-network
* Replay buffer
* Target network
* Training loops
* Full CartPole DQN from scratch

## **CHAPTER 5 — Improvements to DQN**

* Double DQN
* Dueling DQN
* Prioritized Replay
* Noisy Nets
* Rainbow DQN overview
* Implement Double + Dueling in PyTorch

## **CHAPTER 6 — Monte-Carlo Tree Search (MCTS)**

* Upper Confidence Bound (UCB) explained intuitively
* Why search + learning works (AlphaZero)
* Implement tiny MCTS on Tic-Tac-Toe

## **CHAPTER 7 — From Q-Learning to Policies**

* Why softmax$(Q)$ fails
* Policy gradient intuition
* Deriving REINFORCE from scratch
* Implement REINFORCE on CartPole

## **CHAPTER 8 — Actor-Critic**

* Value function as learned baseline
* Temporal difference advantage
* Implement A2C (Advantage Actor-Critic)

## **CHAPTER 9 — Proximal Policy Optimization (PPO)**

* Why TRPO is too complex
* PPO clipping objective derived clearly
* Full PPO implementation

## **CHAPTER 10 — Advanced Concepts**

* GAE (Generalized Advantage Estimation)
* Entropy regularization
* Multi-step returns
* Off-policy vs on-policy
* Continuous action spaces
* Final full PPO continuous-action example

---

# 🎯 **Let’s Begin**

---

# 📘 **CHAPTER 1: The Reinforcement Learning Mindset**

> *“What does it mean for an agent to learn?”*

---

## 1.1 🧠 What is Reinforcement Learning?

Reinforcement Learning (RL) is about an **agent** learning to take **actions** in an **environment** to maximize **reward** over time.

In every RL problem, you have at least these ingredients:

| Symbol          | Meaning                            | Notes                               |
| --------------- | ---------------------------------- | ----------------------------------- |
| $s$             | state (where you are)              | e.g. position in a grid             |
| $a$             | action (what you do)               | e.g. move left/right                |
| $r$             | reward (what you get)              | e.g. $+1$, $0$, $-1$                |
| $s'$            | next state                         | state after taking action           |
| $\pi(a \mid s)$ | **policy**: how actions are chosen | gives a probability for each action |

**Important fix:**
Yes, the original `π(a  s)` you saw was a formatting issue.
The correct notation is:

* $\pi(a \mid s)$ → “the probability of choosing action $a$ when in state $s$”.

---

## 1.2 🎰 The simplest RL problem: a bandit

Think of a **slot machine** with **3 buttons** (also called **arms**):

* Button A (arm 0) → reward $1$ with probability $0.1$
* Button B (arm 1) → reward $1$ with probability $0.5$
* Button C (arm 2) → reward $1$ with probability $0.8$

You **don’t know** these probabilities.
The agent’s goal is to learn:

> **Which button has the highest expected reward?**

This gives us the famous *exploration vs exploitation* dilemma:

* **Explore** → try different buttons to learn their rewards.
* **Exploit** → use the best-looking button so far to get more reward now.

Mathematically, if arm $i$ gives rewards $R_1, R_2, \dots, R_n$, the **expected reward** is approximated by the **average**:

$$
Q(i) \approx \frac{1}{n} \sum_{k=1}^{n} R_k
$$

Here $Q(i)$ is our **estimated value** of arm $i$.

---

## 1.3 🤓 Let’s code a bandit (Python)

First, we create a bandit function.
We won’t show probabilities to the agent; they are “hidden truth”.

```python
import random

# true probabilities (unknown to the agent)
probs = [0.1, 0.5, 0.8]

def pull_arm(arm):
    """
    Simulate pulling one of the 3 arms.
    arm = 0, 1, or 2
    Returns reward 1 or 0.
    """
    p = probs[arm]                # true probability of reward for this arm
    return 1 if random.random() < p else 0
```

Now let’s have a **dumb agent** that randomly picks arms:

```python
rewards = []

for step in range(10):
    action = random.randint(0, 2)     # choose 0, 1, or 2 uniformly
    reward = pull_arm(action)
    rewards.append(reward)
    print(f"step={step}, action={action}, reward={reward}")

average_reward = sum(rewards) / len(rewards)
print("average reward:", average_reward)
```

This agent:

* explores (tries arms),
* but never **learns** which arm is better.

RL is all about learning from these rewards.

---

## 1.4 📈 Estimating values for each action

We want to keep track of the average reward of each action.

Let’s define:

* $Q[a]$ = current estimate of the **expected reward** of arm $a$.
* $N[a]$ = how many times we have pulled arm $a$.

We’ll use **incremental mean**, i.e., we update the mean **on the fly** without storing all past rewards.

### Code first:

```python
import numpy as np
import random

Q = np.zeros(3)       # Q[a] = estimated value of arm a
N = np.zeros(3)       # N[a] = how many times arm a was pulled

for step in range(2000):
    action = random.randint(0, 2)   # pick an arm
    reward = pull_arm(action)       # get reward (0 or 1)

    N[action] += 1                  # increment count for this arm

    # --- incremental mean update ---
    Q[action] += (reward - Q[action]) / N[action]
```

You asked specifically:

> I don't understand this:
> `Q[action] += (reward - Q[action]) / N[action]  # incremental mean`
> expand and explain step by step.

Let’s do that carefully.

---

### 1.4.1 🧮 Incremental mean – intuition

Suppose:

* You have pulled an arm $n-1$ times already.
* The rewards were: $R_1, R_2, \dots, R_{n-1}$.
* The **old mean** is:

$$
\mu_{n-1} = \frac{1}{n-1} \sum_{k=1}^{n-1} R_k
$$

Now you pull the arm **one more time** and get reward $R_n$.

The **new mean** after $n$ pulls is:

$$
\mu_n = \frac{1}{n} \sum_{k=1}^{n} R_k
= \frac{1}{n} \left( R_1 + R_2 + \dots + R_{n-1} + R_n \right)
$$

But we already know the sum of the first $n-1$ rewards:

$$
R_1 + \dots + R_{n-1} = (n-1)\mu_{n-1}
$$

So:

$$
\mu_n = \frac{1}{n} \left( (n-1)\mu_{n-1} + R_n \right)
$$

Now split this:

$$
\mu_n
= \frac{n-1}{n} \mu_{n-1} + \frac{1}{n} R_n
$$

Rewrite this in a “old mean + correction” form:

$$
\begin{aligned}
\mu_n
&= \mu_{n-1} - \frac{1}{n}\mu_{n-1} + \frac{1}{n} R_n \
&= \mu_{n-1} + \frac{1}{n}\left(R_n - \mu_{n-1}\right)
\end{aligned}
$$

So we get the famous **incremental mean formula**:

$$
\mu_n = \mu_{n-1} + \frac{1}{n} \left( R_n - \mu_{n-1} \right)
$$

Now match symbols:

* $\mu_{n-1}$ ↔ old $Q[a]$
* $\mu_n$ ↔ new $Q[a]$
* $R_n$ ↔ `reward`
* $n$ ↔ `N[action]`

So in code:

```python
Q[action] = Q[action] + (reward - Q[action]) / N[action]
```

or using `+=`:

```python
Q[action] += (reward - Q[action]) / N[action]
```

This is exactly:

$$
Q_{\text{new}} = Q_{\text{old}} + \frac{1}{N[a]} \big( \text{reward} - Q_{\text{old}} \big)
$$

This is also a special case of the more general update rule:

$$
Q_{\text{new}} = Q_{\text{old}} + \alpha \big( \text{reward} - Q_{\text{old}} \big),
$$

where the **learning rate** is

$$
\alpha = \frac{1}{N[a]}.
$$

---

## 1.5 🎲 Exploration vs Exploitation: $\epsilon$-greedy

Once we have $Q[a]$ for each arm $a$, we can:

* **Exploit**: pick the arm with the largest $Q[a]$.
* **Explore**: sometimes ignore $Q$ and pick a random arm.

The simplest rule is called **$\epsilon$-greedy**:

* With probability $\epsilon$: choose a random action (exploration).
* With probability $1 - \epsilon$: choose $\arg\max_a Q[a]$ (exploitation).

Mathematically:

* $\mathbb{P}(\text{random action}) = \epsilon$
* $\mathbb{P}(\text{greedy action}) = 1 - \epsilon$

### Code:

```python
epsilon = 0.1   # 10% of the time we explore

if random.random() < epsilon:
    # explore
    action = random.randint(0, 2)
else:
    # exploit (pick action with highest Q)
    action = int(np.argmax(Q))
```

This pattern is used **everywhere in RL**.

---

## 1.6 🧩 A tiny RL loop from scratch (bandit version)

Let’s put everything together for the bandit:

```python
import numpy as np
import random

Q = np.zeros(3)   # value estimates for each arm
N = np.zeros(3)   # number of times each arm was used
epsilon = 0.1

for step in range(2000):
    # 1. choose action (epsilon-greedy)
    if random.random() < epsilon:
        action = random.randint(0, 2)        # explore
    else:
        action = int(np.argmax(Q))           # exploit

    # 2. take action, observe reward
    reward = pull_arm(action)

    # 3. update counts and estimates
    N[action] += 1
    Q[action] += (reward - Q[action]) / N[action]
```

This simple loop already shows:

* Learning from reward
* Exploration vs exploitation
* Value estimation using incremental mean
* Action selection driven by values

All of **deep RL** is: “how do we make this work when”

* there are **many states**
* actions **depend on state**
* we have **sequences** of states
* rewards are **delayed**
* actions influence **future** states

---

## 1.7 🌍 From bandits to environments (Gym-style)

Real RL environments (like `CartPole-v1` in Gym) usually have a `step` function:

```python
next_state, reward, done, info = env.step(action)
```

Where:

* `state` or `observation` is like $s$
* `action` is like $a$
* `reward` is $r$
* `next_state` is $s'$
* `done` indicates whether the episode ended

We’ll build a **mock environment** to understand this pattern.

---

## 1.8 🎮 Building a simple environment from scratch

Here’s a tiny 1D world:

* There is a single integer `state`.
* You start at `state = 0`.
* You can move **left** or **right**:

  * action `0` = left
  * action `1` = right
* You get reward **1** when you reach `state = 5`.
* When you reach `state = 5`, the episode ends (`done = True`).

### Code:

```python
class SimpleEnv:
    def __init__(self):
        self.state = 0  # start at position 0
    
    def reset(self):
        """
        Reset the environment to the initial state.
        Returns the initial state.
        """
        self.state = 0
        return self.state
    
    def step(self, action):
        """
        Take an action:
        action = 0 -> move left  (state - 1)
        action = 1 -> move right (state + 1)
        
        Returns: next_state, reward, done, info
        """
        # actions: 0 = left, 1 = right
        if action == 1:
            # move right
            self.state += 1
        else:
            # move left (this is where action == 0)
            self.state -= 1
        
        # reward is 1 only if we exactly hit state = 5
        reward = 1 if self.state == 5 else 0
        
        # episode is done if we reached the goal state 5
        done = (self.state == 5)
        
        return self.state, reward, done, {}
```

You asked:

> Also in here:
>
> ```python
> def step(self, action):
>         # actions: 0 = left, 1 = right
>         if action == 1:
>             self.state += 1
>         else:
>             self.state -= 1
> ```
>
> if 0 is left by subtract in the else statement.

Exactly. The logic is:

* If `action == 1`, we go right → `self.state += 1`.
* Otherwise (`action` must be `0`), we go left → `self.state -= 1`.

So the `else` branch is indeed “left”, because the only other action is `0`.

---

### 1.8.1 Run an episode with random actions

```python
import random

env = SimpleEnv()
state = env.reset()

for t in range(20):
    # choose a random action: 0 or 1
    action = random.randint(0, 1)
    
    next_state, reward, done, _ = env.step(action)
    
    print(f"t={t}, state={state}, action={action}, reward={reward}, next_state={next_state}")
    
    state = next_state
    
    if done:
        print("Reached goal state 5! Episode finished.")
        break
```

This shows:

* Episode structure
* Terminal state (`done = True`)
* State transitions $s \to s'$
* Reward shaping (reward only at goal)

---

## 1.9 🔥 Full RL loop with state + actions (tabular, bandit-style update)

Now we combine:

* A **stateful environment** (`SimpleEnv`)
* A **Q-table** $Q[s, a]$
* $\epsilon$-greedy exploration
* The **incremental mean** update per $(s, a)$ pair

We’ll assume:

* States are in the range $[-5, 5]$ (just to have a bounded table).
* Actions are $0$ (left) and $1$ (right).
* We map a state $s \in [-5, 5]$ to an array index using `idx(s) = s + 5`.

### 1.9.1 Final end-to-end code (NumPy version)

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
        # action: 0 = left, 1 = right
        if action == 1:
            self.state += 1   # move right
        else:
            self.state -= 1   # move left
        
        reward = 1 if self.state == 5 else 0
        done = (self.state == 5)
        return self.state, reward, done, {}

# Q[state_index, action]
# state range is [-5..5] -> 11 possible states
num_states = 11
num_actions = 2

Q = np.zeros((num_states, num_actions))
N = np.zeros((num_states, num_actions))
epsilon = 0.2

env = SimpleEnv()

def idx(s):
    """
    Map state s in [-5, 5] to index in [0, 10].
    For example: s = -5 -> 0,  s = 0 -> 5,  s = 5 -> 10.
    """
    return s + 5

# Training loop
for episode in range(200):
    state = env.reset()
    done = False

    while not done:
        s_idx = idx(state)

        # epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 1)         # explore
        else:
            action = int(np.argmax(Q[s_idx]))     # exploit

        next_state, reward, done, _ = env.step(action)
        s_idx_next = idx(next_state)

        # Incremental update (bandit-style per (state, action))
        N[s_idx, action] += 1
        alpha = 1.0 / N[s_idx, action]
        Q[s_idx, action] += alpha * (reward - Q[s_idx, action])

        state = next_state

print("Learned Q-table:")
print(Q)
```

Again, the core update per state-action pair is:

```python
alpha = 1.0 / N[s_idx, action]
Q[s_idx, action] += alpha * (reward - Q[s_idx, action])
```

Which is the incremental mean:

$$
Q_{\text{new}}(s,a) = Q_{\text{old}}(s,a) + \frac{1}{N(s,a)} \left( r - Q_{\text{old}}(s,a) \right)
$$

---

## 1.10 🧪 (Optional) Same idea with PyTorch tensors

Just to start getting used to PyTorch, here’s what the **Q-table** part might look like with PyTorch instead of NumPy:

```python
import torch
import random

num_states = 11
num_actions = 2

Q = torch.zeros(num_states, num_actions)  # Q[s, a]
N = torch.zeros(num_states, num_actions)  # visit counts

epsilon = 0.2

def choose_action(state_idx):
    if random.random() < epsilon:
        # explore
        return random.randint(0, num_actions - 1)
    else:
        # exploit: argmax over actions
        return int(torch.argmax(Q[state_idx]).item())
```

Update rule in PyTorch:

```python
s_idx = idx(state)

action = choose_action(s_idx)
next_state, reward, done, _ = env.step(action)

# convert reward to float tensor if we like
reward_t = torch.tensor(float(reward))

N[s_idx, action] += 1.0
alpha = 1.0 / N[s_idx, action]

Q[s_idx, action] += alpha * (reward_t - Q[s_idx, action])
```

This is mathematically identical; we’re just using **PyTorch tensors** instead of NumPy arrays.
Later, when $Q$ is a **neural network**, PyTorch will let us do **backprop and gradients** automatically.

---

# 🎉 **End of CHAPTER 1**

By now, you understand:

* States $s$
* Actions $a$
* Rewards $r$
* Next states $s'$
* Policies $\pi(a \mid s)$
* Episodes and terminal states (`done`)
* Value estimation $Q$ with the **incremental mean**
* Exploration vs exploitation with **$\epsilon$-greedy**
* How a Gym-like environment API works (`reset`, `step`)

This is the **seed** of all reinforcement learning.



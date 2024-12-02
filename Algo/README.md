# PPO (Proximal Policy Optimization)

## Overview
The `PPO` folder contains the implementation of the Proximal Policy Optimization (PPO) algorithm, a widely-used reinforcement learning (RL) technique. The PPO algorithm is designed to balance learning efficiency and stability by using a clipped objective function.

---

## File Descriptions

### **PPO.py**
- **Purpose**: Implements the PPO algorithm, including the policy and value networks, the update process, and memory for rollouts.
- **Key Classes and Functions**:
  1. **PolicyNet (Actor)**:
      - Defines the policy network responsible for action selection.
      - Utilizes a flexible architecture defined by `net_generator`.

  2. **ValueNet (Critic)**:
      - Defines the value network to estimate state values for advantage computation.
      - Shares a similar architecture with the policy network but outputs a single scalar value.

  3. **PPO**:
      - Implements the PPO algorithm, including:
        - Clipped surrogate loss for policy updates.
        - Value function loss for value updates.
        - Advantage computation using Generalized Advantage Estimation (GAE).
      - Key hyperparameters:
        - `GAMMA`: Discount factor for rewards.
        - `LAMBDA`: GAE lambda for smoothing advantages.
        - `CLIP_EPSILON`: Clipping parameter for the surrogate loss.
        - `ENTROPY_COEFF`: Coefficient for entropy regularization.
        - `VALUE_COEFF`: Coefficient for value loss.
        - `LEARNING_RATE`: Learning rate for optimization.
        - `K_EPOCHS`: Number of optimization epochs per update.
        - `BATCH_SIZE`: Batch size for mini-batch optimization.

  4. **Memory**:
      - A helper class for storing rollouts (state, action, reward, log-probability, and done flag).
      - Includes methods to store and clear rollout data.

---

## Usage Example

The `PPO.py` file is designed to work with OpenAI Gym environments. Below is an example of how to use the PPO agent:

```python
import gym
from PPO import PPO, Memory

# Create environment
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize agent and memory
agent = PPO(state_dim, action_dim, hidden_layers_actor=[256, 128], hidden_layers_critic=[128, 128], activation_fn="LeakyReLU")
memory = Memory()

for episode in range(1000):
    state = env.reset()
    total_reward = 0

    while True:
        action, log_prob, _ = agent.action(state)
        next_state, reward, done, _ = env.step(action)

        memory.store(state, action, reward, log_prob, done)

        state = next_state
        total_reward += reward

        if done:
            break

    # Update PPO
    agent.update(memory)
    memory.clear()

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    if total_reward >= 500:
        print("Solved!")
        break
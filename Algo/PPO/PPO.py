import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

import net_generator

# Policy Network (Actor)
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=[256, 128], activation_fn="LeakyReLU"):
        super(PolicyNet, self).__init__()
        activation_function = net_generator.get_activation_fn(activation_fn)
        self.net = net_generator.create_network(input_dim=state_dim,
                                                output_dim=action_dim,
                                                hidden_layers=hidden_layers,
                                                activation_fn=activation_function)
        
    def forward(self, state):
        return self.net(state)

# Value Network (Critic)
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_layers=[128, 128], activation_fn="LeakyReLU"):
        super(ValueNet, self).__init__()
        activation_function = net_generator.get_activation_fn(activation_fn)
        self.net = net_generator.create_network(input_dim=state_dim,
                                                output_dim=1,
                                                hidden_layers=hidden_layers,
                                                activation_fn=activation_function)
    
    def forward(self, state):
        return self.net(state)

# PPO Agent BATCH_SIZE should be checked
class PPO:
    def __init__(self, state_dim, action_dim, hidden_layers_actor, hidden_layers_critic, activation_fn,
                 GAMMA=0.99, LAMBDA=0.95, CLIP_EPSILON=0.2, ENTROPY_COEFF=0.01, VALUE_COEFF=0.5, 
                 LEARNING_RATE=1e-4, K_EPOCHS=4, BATCH_SIZE=4):
        # generate actor, critic, actor_old networks
        self.policy = PolicyNet(state_dim, action_dim, hidden_layers_actor, activation_fn)
        self.value = ValueNet(state_dim, hidden_layers_critic, activation_fn)
        self.policy_old = PolicyNet(state_dim, action_dim, hidden_layers_actor, activation_fn)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=LEARNING_RATE)
        
        # set hyperparameters for PPO
        self.GAMMA = GAMMA                      # Discount Factor
        self.LAMBDA = LAMBDA                    # GAE Lambda
        self.CLIP_EPSILON = CLIP_EPSILON        # PPO Clipping Parameter
        self.ENTROPY_COEFF = ENTROPY_COEFF      # Entropy Coefficient
        self.VALUE_COEFF = VALUE_COEFF          # Value Loss Coefficient
        self.LEARNING_RATE = LEARNING_RATE      # Learning Rate
        self.K_EPOCHS = K_EPOCHS                # Number of optimization epochs
        self.BATCH_SIZE = BATCH_SIZE            # Mini-batch size

    def update(self, memory):
        states = torch.FloatTensor(np.array(memory["states"]))
        actions = torch.LongTensor(np.array(memory["actions"]))
        old_log_probs = torch.cat(memory["log_probs"])
        rewards = torch.FloatTensor(memory["rewards"])
        dones = torch.FloatTensor(memory["dones"])
        values = self.value(states).squeeze()
        next_values = torch.cat([values[1:], torch.zeros(1)])
        
        # Compute advantages
        advantages = self.advantage(rewards, values, next_values, dones)
        returns = advantages + values  # Target for value function

        # Update policy and value networks
        for _ in range(self.K_EPOCHS):
            for i in range(0, len(states), self.BATCH_SIZE):
                # Mini-batch
                idx = slice(i, i + self.BATCH_SIZE)
                state_batch = states[idx]
                action_batch = actions[idx]
                old_log_prob_batch = old_log_probs[idx]
                advantage_batch = advantages[idx]
                return_batch = returns[idx]

                # Policy Update
                probs = self.policy(state_batch)
                dist = Categorical(probs)
                log_probs = dist.log_prob(action_batch)
                entropy = dist.entropy().mean()

                # Compute ratios
                ratios = torch.exp(log_probs - old_log_prob_batch)

                # Clipped surrogate objective
                surr1 = ratios * advantage_batch
                surr2 = torch.clamp(ratios, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEFF * entropy

                # Value function loss
                value_loss = nn.MSELoss()(self.value(state_batch).squeeze(), return_batch)

                # Combined loss
                loss = policy_loss + VALUE_COEFF * value_loss

                # Gradient descent
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()
                self.value_optimizer.step()

        # Update old policy weights
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_old(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()
    
    def advantage(self, rewards, values, next_values, dones):
        deltas = rewards + GAMMA * next_values * (1 - dones) - values
        advantages = []
        advantage = 0
        for delta in reversed(deltas):
            advantage = delta + GAMMA * LAMBDA * advantage
            advantages.insert(0, advantage)
        return torch.FloatTensor(advantages)

# Memory to store rollouts
class Memory:
    def __init__(self):
        self.clear()

    def store(self, state, action, reward, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []

# Example Usage (Assume Gym environment)
if __name__ == "__main__":
    import gym
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim)
    memory = Memory()

    for episode in range(1000):
        state = env.reset()
        total_reward = 0

        for t in range(ROLL_OUT_LENGTH):
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

        if total_reward >= 500:  # Stop early if solved
            print("Solved!")
            break

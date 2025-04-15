import copy
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPG:
    def __init__(self, actor, critic, device="cuda", batch_size=256, gamma=0.99, tau=0.005, 
                 actor_lr=0.001, critic_lr=0.001, hard_update_interval=None):
        self.batch_size = batch_size  # Batch size
        self.GAMMA = gamma  # Discount factor
        self.TAU = tau  # Target network soft update factor
        self.device = device
        self.hard_update_interval = hard_update_interval  # If set, uses hard updates every N steps
        self.update_counter = 0  # Track number of updates for hard updates

        # Initialize actor & critic
        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)

        # Move models to device
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Loss function
        self.mse_loss = nn.MSELoss().to(self.device)

    def choose_action(self, s, deterministic=False):
        """
        Select an action using the actor network.
        - Uses Gumbel Softmax during training.
        - Switches to deterministic argmax selection during inference.
        """
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        a, a_logit = self.actor(s)

        if deterministic:
            a = torch.argmax(a_logit, dim=-1)  # Ensure deterministic selection during inference

        return a.data.cpu().numpy().flatten(), a_logit.data.cpu().numpy()

    def learn(self, replay_buffer, wandb_run=None):
        """
        Perform a training step using replay buffer data.
        """
        if replay_buffer.size < self.batch_size:
            return  # Avoid training if buffer is too small

        # Sample from replay buffer (including predicted rewards)
        batch_s, batch_a, batch_a_logit, batch_r, batch_pred_r, batch_s_, batch_dw = replay_buffer.sample(
            self.batch_size
        )
        batch_s, batch_a, batch_a_logit, batch_r, batch_pred_r, batch_s_, batch_dw = (
            batch_s.to(self.device),
            batch_a.to(self.device),
            batch_a_logit.to(self.device),
            batch_r.to(self.device),
            batch_pred_r.to(self.device),  # ✅ Predicted reward integration
            batch_s_.to(self.device),
            batch_dw.to(self.device),
        )

        # Normalize predicted rewards (ensure it's within a reasonable range)
        batch_pred_r = torch.clamp(batch_pred_r, -1.0, 1.0)

        # Compute target Q-value
        with torch.no_grad():
            Q_ = self.critic_target(batch_s_, self.actor_target(batch_s_)[1])
            target_Q = batch_r + batch_pred_r + self.GAMMA * (1 - batch_dw) * Q_  # ✅ Include predicted reward

        # Compute critic loss
        current_Q = self.critic(batch_s, batch_a_logit)
        critic_loss = self.mse_loss(target_Q, current_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(batch_s, self.actor(batch_s)[1]).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Target network updates (soft or hard)
        if self.hard_update_interval is not None and self.update_counter % self.hard_update_interval == 0:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.actor_target.load_state_dict(self.actor.state_dict())
        else:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        self.update_counter += 1  # Increment update step counter

        # Log losses to WandB
        if wandb_run:
            wandb_run.log({"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()})


def evaluate_policy(env, agent):
    """
    Evaluate the policy by running the agent in the environment.
    """
    times = 3  # Perform three evaluations and calculate the average
    evaluate_reward = 0

    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0

        while not done:
            a, _ = agent.choose_action(s, deterministic=True)  # ✅ Ensure deterministic evaluation
            s_, r, done, _ = env.step(a)
            episode_reward += r
            s = s_

        evaluate_reward += episode_reward

    return int(evaluate_reward / times)


def reward_adapter(r, env_index):
    """
    Normalize rewards based on environment index.
    """
    if env_index == 0:  # Pendulum-v1
        r = (r + 8) / 8
    elif env_index == 1:  # BipedalWalker-v3
        if r <= -100:
            r = -1
    return r


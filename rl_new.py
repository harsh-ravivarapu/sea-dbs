import os
import gym
import matlab.engine
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import torch.quantization

from ddpg import DDPG
from utils import seed_everything, init_weights_xavier
from predictive_model import PredictiveModel  # EfficientZero-inspired predictive model

os.environ["WANDB_SILENT"] = "true"

# Select Training Mode: "baseline", "qat", "ptq_int8", "ptq_fp16", "combined"
TRAINING_MODE = "baseline"

# ===========================
#  Brain Class (Environment)
# ===========================
class Brain(gym.Env):
    def __init__(self, freq, l, b, time_step, stride, window_size, n_obs, action_dim):
        self.freq = float(freq)
        self.len = float(l)
        self.b = float(b)
        self.isdone = False
        self.IT = None
        self.kk = None
        self.time_step = float(time_step)
        self.stride = float(stride)
        self.window_size = float(window_size)
        self.eng = matlab.engine.start_matlab()
        self.obs = None
        self.n_obs = n_obs
        self.action_dim = action_dim
        self.reward = None
        
    def reset(self):
        # run the reset function
        obs, IT = self.eng.reset_function_SMC_step(self.freq, self.len, self.time_step, self.stride, self.window_size,
                                                   nargout=2)
        self.IT = float(IT)
        obs = np.array(obs)
        self.state = obs
        self.reward = 0
        return obs
    
    def step(self, action):
        # run the step function
        action = np.array(action, dtype='float64')
        # action = np.ones_like(action)  # TODO: update this.
        action = matlab.double(action)
        obs, reward, isdone, IT = self.eng.step_function_SMC_step(
            action, self.IT, self.freq, self.len, self.b, self.time_step, self.stride, self.window_size, nargout=4)
        obs = np.array(obs)
        self.IT = float(IT)
        reward = float(reward)
        isdone = float(isdone)
        info = {}  # placeholder
        self.state = obs
        self.reward += reward
        
        return obs, reward, isdone, info
    
    def random_action(self):
        return np.random.randint(2, size=self.n_obs)
    
    def uniform_action(self):
        action = np.zeros(self.n_obs)
        action[:: (self.n_obs // self.action_dim)] = 1
        return action
    
    def end(self):
        self.eng.quit()
        
# ===========================
#  Actor Class
# ===========================
class Actor(nn.Module):
    def __init__(self, state_dim=2, state_len=100, action_dim=2, shrink_dim=4):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.state_len = state_len
        self.action_dim = action_dim
        self.shrink_dim = shrink_dim
        
        # === Convolutional Layers ===
        self.conv1 = nn.Conv1d(state_dim, 32, 3, padding=1)
        self.avg_pool1 = nn.AvgPool1d(shrink_dim)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.avg_pool2 = nn.AvgPool1d(shrink_dim)
        
        # === Fully Connected Layers ===
        self.linear1 = nn.Linear(self.state_len // shrink_dim // shrink_dim * 64, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, action_dim * state_len)
        
        # === Quantization Placeholder ===
        if TRAINING_MODE in ["qat", "ptq_int8", "ptq_fp16", "combined"]:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
            
    def forward(self, state, device="cuda"):
        batch_size = state.size(0)
        
        # === Apply Quantization (if enabled) ===
        if TRAINING_MODE in ["qat", "ptq_int8", "ptq_fp16", "combined"]:
            state = self.quant(state)
            
        output = self.avg_pool1(F.relu(self.conv1(state)))
        output = self.avg_pool2(F.relu(self.conv2(output)))
        
        output = output.view(batch_size, -1)
        output = F.relu(self.linear1(output))
        output = F.relu(self.linear2(output))
        logits = self.linear3(output)
        
        # === Apply Dequantization (if enabled) ===
        if TRAINING_MODE in ["qat", "ptq_int8", "ptq_fp16", "combined"]:
            logits = self.dequant(logits)
            
        output2 = logits.view(-1, self.action_dim, self.state_len)
        output2 = F.softmax(output2, dim=1)
        output2 = torch.argmax(output2, dim=-1)
        
        actions = torch.zeros(batch_size, self.state_len).to(device)
        actions[torch.arange(batch_size).unsqueeze(1), output2] = 1
        
        return actions, logits
    
    def quantize_model(self):
        """ Apply Post-Training Quantization (PTQ) """
        if TRAINING_MODE == "ptq_int8":
            self.fuse_model()
            self.qconfig = torch.quantization.get_default_qconfig("fbgemm")
            torch.quantization.prepare(self, inplace=True)
            torch.quantization.convert(self, inplace=True)
        elif TRAINING_MODE == "ptq_fp16":
            self.half()
            
    def fuse_model(self):
        """ Fuse Conv + ReLU layers for quantization efficiency """
        torch.quantization.fuse_modules(self, [['conv1', 'relu'], ['conv2', 'relu']], inplace=True)
        
# ===========================
#  Critic Class
# ===========================
class Critic(nn.Module):
    def __init__(self, state_dim=2, state_len=100, action_dim=2, shrink_dim=4):
        super(Critic, self).__init__()
        self.state_len = state_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.shrink_dim = shrink_dim
        
        # === Convolutional Layers ===
        self.conv1 = nn.Conv1d(state_dim, 32, 3, padding=1)
        self.avg_pool1 = nn.AvgPool1d(shrink_dim)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.avg_pool2 = nn.AvgPool1d(shrink_dim)
        
        # === Fully Connected Layers ===
        self.linear1 = nn.Linear(self.state_len // shrink_dim // shrink_dim * 64, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, action_dim * state_len)
        self.linear4 = nn.Linear(action_dim * state_len, 256)
        self.linear5 = nn.Linear(256, 1)
        
        # === Quantization Placeholder ===
        if TRAINING_MODE in ["qat", "ptq_int8", "ptq_fp16", "combined"]:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
            
    def forward(self, state, action_logits):
        # === Apply Quantization (if enabled) ===
        if TRAINING_MODE in ["qat", "ptq_int8", "ptq_fp16", "combined"]:
            state = self.quant(state)
            action_logits = self.quant(action_logits)
            
        output = self.avg_pool1(F.relu(self.conv1(state)))
        output = self.avg_pool2(F.relu(self.conv2(output)))
        
        output = output.view(output.size(0), -1)
        output = F.relu(self.linear1(output))
        output = F.relu(self.linear2(output))
        
        action_output = F.relu(self.linear4(action_logits))
        all_output = output + action_output
        value = self.linear5(all_output)
        
        # === Apply Dequantization (if enabled) ===
        if TRAINING_MODE in ["qat", "ptq_int8", "ptq_fp16", "combined"]:
            value = self.dequant(value)
            
        return value
    
    def quantize_model(self):
        """ Apply Post-Training Quantization (PTQ) """
        if TRAINING_MODE == "ptq_int8":
            self.fuse_model()
            self.qconfig = torch.quantization.get_default_qconfig("fbgemm")
            torch.quantization.prepare(self, inplace=True)
            torch.quantization.convert(self, inplace=True)
        elif TRAINING_MODE == "ptq_fp16":
            self.half()
            
    def fuse_model(self):
        """ Fuse Conv + ReLU layers for quantization efficiency """
        torch.quantization.fuse_modules(self, [['conv1', 'relu'], ['conv2', 'relu']], inplace=True)

# ===========================
#  Replay Buffer
# ===========================
class ReplayBuffer:
    def __init__(self, state_dim, state_len, action_dim, max_size):
        self.max_size = max_size
        self.count = 0
        self.size = 0
        
        # === Storage Initialization ===
        self.s = np.zeros((self.max_size, state_dim, state_len), dtype=np.float32)
        self.a = np.zeros((self.max_size, state_len), dtype=np.float32)
        self.a_logit = np.zeros((self.max_size, action_dim * state_len), dtype=np.float32)
        self.r = np.zeros((self.max_size, 1), dtype=np.float32)
        self.pred_r = np.zeros((self.max_size, 1), dtype=np.float32)  #  Store predicted reward
        self.s_ = np.zeros((self.max_size, state_dim, state_len), dtype=np.float32)
        self.dw = np.zeros((self.max_size, 1), dtype=np.float32)
        
    def store(self, s, a, a_logit, r, pred_r, s_, dw):
        """ Store transition in the buffer """
        idx = self.count % self.max_size
        self.s[idx] = s
        self.a[idx] = a
        self.a_logit[idx] = a_logit
        self.r[idx] = r
        self.pred_r[idx] = pred_r  #  Store predicted reward
        self.s_[idx] = s_
        self.dw[idx] = dw
        
        self.count += 1
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        """ Sample a batch from the buffer """
        index = np.random.choice(self.size, size=batch_size, replace=False)
        
        batch_s = torch.tensor(self.s[index], dtype=torch.float32)
        batch_a = torch.tensor(self.a[index], dtype=torch.float32)
        batch_a_logit = torch.tensor(self.a_logit[index], dtype=torch.float32)
        batch_r = torch.tensor(self.r[index], dtype=torch.float32)
        batch_pred_r = torch.tensor(self.pred_r[index], dtype=torch.float32)  #  Include predicted reward
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float32)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float32)
        
        return batch_s, batch_a, batch_a_logit, batch_r, batch_pred_r, batch_s_, batch_dw
# ===========================
#  Gumbel Softmax for Exploration (with Annealing)
# ===========================

# Annealing parameters
tau_0 = 1.0  # Initial tau
tau_min = 0.1  # Minimum tau (to prevent non-differentiability)
lambda_tau = 0.001  # Decay rate

def get_tau(t):
    """
    Compute the annealed temperature parameter tau.
    
    Args:
        t (int): Current training step.
    
    Returns:
        float: The annealed tau value.
    """
    return max(tau_min, tau_0 * np.exp(-lambda_tau * t))

def gumbel_softmax(logits, tau):
    """
    Perform Gumbel-Softmax trick to sample discrete actions from logits with annealing tau.

    Args:
        logits (torch.Tensor): The logits (unnormalized scores) from the policy network.
        tau (float): Temperature parameter that controls randomness (annealed).

    Returns:
        torch.Tensor: A one-hot encoded action.
    """
    noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10))  # Gumbel noise
    gumbel_logits = (logits + noise) / tau  # Apply temperature scaling
    return F.softmax(gumbel_logits, dim=-1)  # Compute probabilities

# ===========================
#  Main Function
# ===========================
def main():
    seed = 0
    seed_everything(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === ENVIRONMENT PARAMETERS ===
    freq = 50  
    t_enviro = 1000  
    max_episodes = 150  
    steps = 30  
    critic_lr = 5e-4    
    actor_lr = 1e-4    
    gamma = 0.99  
    tau = 0.3  
    buffer_size = 8126  
    batch_size = 32  
    enviro_stride = 1  
    enviro_window_size = 1000  
    state_dim = 1  
    step_size = 0.02
    action_dim = freq * t_enviro // 1000
    n_obs = t_enviro // enviro_stride
    update_freq = 4
    start_steps = 10

    run = wandb.init(
        project="rl_brain", mode="offline",
        config={
            "critic_lr": critic_lr,
            "actor_lr": actor_lr,
            "seed": seed,
            "enviro_stride": enviro_stride,
            "enviro_window_size": enviro_window_size,
            "t_enviro": t_enviro,
            "freq": freq,
            "max_episodes": max_episodes,
            "steps": steps,
            "step_size": step_size,
            "buffer_size": buffer_size,
            "state_dim": state_dim,
            "batch_size": batch_size,
            "update_freq": update_freq
        })

    print("Total length:", t_enviro, "step_size:", step_size)
    print('Length of observation vector:', n_obs, 'Action Dim:', action_dim)

    # === INITIALIZE MODELS ===
    actor = Actor(state_dim=state_dim, state_len=n_obs, action_dim=action_dim, shrink_dim=2).to(device)
    critic = Critic(state_dim=state_dim, state_len=n_obs, action_dim=action_dim, shrink_dim=2).to(device)
    predictive_model = PredictiveModel(state_dim=state_dim, action_dim=action_dim).to(device)

    # === Apply Quantization (if enabled) ===
    if TRAINING_MODE in ["qat", "ptq_int8", "ptq_fp16", "combined"]:
        actor.quantize_model()
        critic.quantize_model()

    actor.apply(init_weights_xavier)
    critic.apply(init_weights_xavier)

    env = Brain(freq, t_enviro, action_dim, step_size, enviro_stride, enviro_window_size, n_obs, action_dim)
    agent = DDPG(actor, critic, batch_size=batch_size, gamma=gamma, tau=tau, actor_lr=actor_lr, critic_lr=critic_lr,
                 device=device)
    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, state_len=n_obs, max_size=buffer_size)

    total_steps = 0

    for episode in range(max_episodes):
        print(f"Episode {episode}")
        s = env.reset()
        s = np.expand_dims(s, axis=0)  #  Ensure correct shape (1, state_dim, state_len)
        done = False
        r_list = []
        beta_list = []
        ei_list = []
        
        for step in range(steps):
            # === Convert state to PyTorch tensor ===
            s_tensor = torch.tensor(np.array(s), dtype=torch.float32).to(device).squeeze(0)
            
            #  Compute annealed tau before action selection
            tau_t = get_tau(total_steps)
            
            # === Select Action ===
            logits_resized = np.zeros((action_dim * n_obs))
            if total_steps < start_steps:
                a = env.uniform_action()
                if a.shape[0] != action_dim:
                    a = np.random.randint(2, size=action_dim)
                a = np.expand_dims(a, axis=0)
                logits = np.zeros((action_dim * n_obs))
            else:
                #  Get action and logits separately
                a, logits = agent.choose_action(s_tensor)
                
                #  Ensure logits is a NumPy array
                if isinstance(logits, torch.Tensor):
                    logits = logits.cpu().detach().numpy()
                    
                #  Fix logits shape before storing
                expected_logits_shape = (action_dim * n_obs,)  # (50 * 1000 = 50000)
                logits_resized[:] = logits[:expected_logits_shape[0]]
                
                if logits.shape == expected_logits_shape:
                    logits_resized[:] = logits
                elif logits.size > expected_logits_shape[0]:
                    logits_resized[:] = logits[:expected_logits_shape[0]]  # Trim if too large
                elif logits.size < expected_logits_shape[0]:
                    logits_resized[:logits.shape[0]] = logits  # Fill available values
                    
                #  Convert logits to tensor for Gumbel-Softmax
                logits_tensor = torch.tensor(logits_resized, dtype=torch.float32, device=device)
                
                #  Sample action using annealed tau
                a = gumbel_softmax(logits_tensor, tau_t).cpu().numpy()
                
            #  Ensure action has correct shape
            if a.shape[0] != action_dim:
                a = np.random.randint(2, size=action_dim)
            a = np.expand_dims(a, axis=0)
            
            # === Predict Reward (EfficientZero) ===
            with torch.no_grad():
                predicted_r = predictive_model.predict(s, a)
                
            # === Step in Environment ===
            s_, r, done, _ = env.step(a)
            s_ = np.expand_dims(s_, axis=0)
            
            #  Extract `beta` from the state
            beta = np.mean(s_[0, :])
            ei = 0
            
            #  Fix action shape before storing in replay buffer
            a_resized = np.zeros(n_obs)
            a_resized[:action_dim] = a
            
            #  Store in Replay Buffer with corrected logits shape
            replay_buffer.store(s, a_resized, logits_resized, r, predicted_r, s_, done)
            
            s = s_  # Update state
            r_list.append(r)
            beta_list.append(beta)
            ei_list.append(ei)
            
            #  Update total steps for annealing
            total_steps += 1
            
            # === Train the Model ===
            for _ in range(update_freq):
                agent.learn(replay_buffer, wandb_run=run)
                
            print(f"Step {step}, reward: {r:.3f}, predicted reward: {predicted_r:.3f}, beta: {beta:.3f}, ei: {ei:.3f}")
            
            #  Log metrics to WandB
            wandb.log({
                "reward": r,
                "predicted_reward": predicted_r,
                "beta": beta,
                "ei": ei,
                "tau": tau_t,  #  Log tau for debugging
                "step": step,
                "episode": episode
            })
            
            if done:
                break
            
        print(f"Episode {episode}, "
                f"reward: {np.sum(r_list):.3f}, beta: {np.mean(beta_list):.3f}, ei: {np.mean(ei_list):.3f}")
        
        wandb.log({
            "reward_epoch": np.sum(r_list),
            "beta_epoch": np.mean(beta_list),
            "ei_epoch": np.mean(ei_list),
            "episode": episode
        })
        
        # ===== Save Model Checkpoints =====
        torch.save(actor.state_dict(), "actor_model.pth")
        torch.save(critic.state_dict(), "critic_model.pth")
        print("Actor and Critic models saved successfully.")
        
    env.end()
   
if __name__ == "__main__":
    main()


import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictiveModel(nn.Module):
    def __init__(self, state_dim, action_dim, state_len=1000, hidden_dim=128):
        """
        EfficientZero-inspired predictive model.
        Predicts future rewards based on state-action pairs.
        """
        super(PredictiveModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_len = state_len
        
        # Correct input size for fc1 (flattened state + action)
        self.input_size = (state_dim + action_dim) * state_len  #look into inpust size
        
        self.fc1 = nn.Linear(self.input_size, hidden_dim)  # Ensure correct input size
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Predicts scalar reward
        
    def forward(self, state, action):
        """
        Forward pass for predicting future reward.
        Args:
            state (torch.Tensor): (batch_size, state_dim, state_len)
            action (torch.Tensor): (batch_size, action_dim)
        Returns:
            torch.Tensor: Predicted reward
        """
        device = next(self.parameters()).device  
        state = state.to(device)  
        action = action.to(device)  
        
        batch_size, state_dim, state_len = state.shape
        batch_size, action_dim = action.shape  # Extract correct action shape
        
       # print(f" DEBUG: state shape = {state.shape}")  # (batch_size, state_dim, state_len)
       # print(f" DEBUG: action shape (before expand) = {action.shape}")  # (batch_size, action_dim)
        
        #  Fix Expansion - Ensure action matches expected dimensions
        action = action.unsqueeze(2).expand(-1, -1, state_len)  # (batch_size, action_dim, state_len)
        
        #print(f" DEBUG: action shape (after expand) = {action.shape}")  # Expected: (batch_size, action_dim, state_len)
        
        # Concatenate state and action correctly
        x = torch.cat([state, action], dim=1)  # (batch_size, state_dim + action_dim, state_len)
        
       # print(f" DEBUG: concatenated x shape = {x.shape}")  # Expected: (batch_size, state_dim + action_dim, state_len)
        
        #  Ensure Flattening is Correct
        x = x.view(batch_size, -1)  # Flatten (batch_size, (state_dim + action_dim) * state_len)
        
        #print(f" DEBUG: Flattened x shape = {x.shape}")  # Expected: (batch_size, 51000)
        
        assert x.shape[1] == self.input_size, f"Expected {self.input_size}, but got {x.shape[1]}"
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Output predicted reward
    
    
    
    def predict(self, state, action):
        """Predict reward given state and action."""
        state_tensor = torch.tensor(state, dtype=torch.float)
        
        #  Ensure state is 3D (batch_size, state_dim, state_len)
        if state_tensor.dim() == 2:  # If missing batch dimension
            state_tensor = state_tensor.unsqueeze(0)  # Convert to (1, state_dim, state_len)
        
        action_tensor = torch.tensor(action, dtype=torch.float).view(1, -1)  # Ensure (1, action_dim)
        
        # Debugging logs to verify
       # print(f" DEBUG: state_tensor shape: {state_tensor.shape}")   # Should be (1, state_dim, state_len)
       # print(f" DEBUG: action_tensor shape: {action_tensor.shape}") # Should be (1, action_dim)
        
        with torch.no_grad():
            return self.forward(state_tensor, action_tensor).item()
        
        
        
    def train_model(self, replay_buffer, optimizer, batch_size=32):
        """
        Train the predictive model using stored experiences from the replay buffer.
        Args:
            replay_buffer (ReplayBuffer): Experience replay buffer
            optimizer (torch.optim): Optimizer for updating model parameters
            batch_size (int): Number of samples per training step
        Returns:
            float: Training loss
        """
        if replay_buffer.size < batch_size:
            return None  # Not enough samples to train

        # Sample from replay buffer
        batch_s, batch_a, _, batch_r, _, _, _ = replay_buffer.sample(batch_size)

        # Convert to tensors
        batch_s = batch_s.to(torch.float32)
        batch_a = batch_a.to(torch.float32)
        batch_r = batch_r.to(torch.float32)

        # Compute predicted rewards
        predicted_rewards = self.forward(batch_s, batch_a)

        # Compute loss
        loss = F.mse_loss(predicted_rewards.squeeze(), batch_r.squeeze())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def save_model(self, path="predictive_model.pth"):
        """Save the model's state dictionary."""
        torch.save(self.state_dict(), path)

    def load_model(self, path="predictive_model.pth", device="cpu"):
        """Load model from a saved checkpoint."""
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()  # Set model to evaluation mode
                                                  


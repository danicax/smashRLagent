import torch.nn as nn
import torch

class TransformerQNet(nn.Module):
    def __init__(self, obs_dim, n_actions, seq_len, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        # e.g. a simple TransformerEncoder that takes tokens of size obs_dim
        self.input_proj = nn.Linear(obs_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # after encoding the entire sequence, project the final hidden state to Q-values:
        self.head = nn.Linear(d_model, n_actions)

    def forward(self, x_seq):
        """
        x_seq: Tensor of shape [B, SEQ_LEN, obs_dim]
        returns: Tensor [B, n_actions] (Q-values for the current time step)
        """
        z = self.input_proj(x_seq)  # [B, SEQ_LEN, d_model]
        z_enc = self.transformer(z) # [B, SEQ_LEN, d_model]
        last_hidden = z_enc[:, -1]  # [B, d_model]
        return self.head(last_hidden)
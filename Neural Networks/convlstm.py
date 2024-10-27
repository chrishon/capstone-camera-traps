# Source: https://github.com/ndrplz/ConvLSTM_pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, hidden_state):
        h_cur, c_cur = hidden_state
        combined = torch.cat([x, h_cur], dim=1)  
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, height, width):
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.conv_lstm_cells = nn.ModuleList([
            ConvLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size, padding=1)
            for i in range(num_layers)
        ])
        self.conv = nn.Conv2d(hidden_dim, 3, kernel_size=3, padding=1)  # Output layer to predict next frame

    def forward(self, x):
        batch_size, seq_len, _, height, width = x.size()
        hidden_state = [cell.init_hidden(batch_size, height, width) for cell in self.conv_lstm_cells]

        for t in range(seq_len):
            x_t = x[:, t, :, :, :]
            for i, cell in enumerate(self.conv_lstm_cells):
                h, c = hidden_state[i]
                h, c = cell(x_t, (h, c))
                hidden_state[i] = (h, c)
                x_t = h

        # Apply final Conv layer to get the next frame
        output = self.conv(x_t)
        return output

import torch
import pdb

from utils import distances


class LSTMAE(torch.nn.Module):
    def __init__(self, nin, nh, nl, nout, nlayers, do):
        super(LSTMAE, self).__init__()
        self.nh = nh
        self.nl = nl
        if nlayers >= 2:
            self.enc = torch.nn.LSTM(input_size=nin, hidden_size=nh, num_layers=nlayers, dropout=do, batch_first=True)
            self.dec = torch.nn.LSTM(input_size=nl + 1, hidden_size=nh, num_layers=nlayers, dropout=do, batch_first=True)
        else:
            self.enc = torch.nn.LSTM(input_size=nin, hidden_size=nh, num_layers=nlayers, batch_first=True)
            self.dec = torch.nn.LSTM(input_size=nl + 1, hidden_size=nh, num_layers=nlayers, batch_first=True)
        self.fcd = torch.nn.Linear(nh, nout)
        self.fce = torch.nn.Linear(nh, nl)
        self.do = torch.nn.Dropout(p=do)

    def forward(self, x, seq_len):
        n, _, _ = x.shape
        z = self.encode(x)
        z = z[torch.arange(n), (seq_len - 1).type(dtype=torch.long)]
        pred = self.decode(x[:, :, 0], z)  # index: 0-time, 1-flux, 2-flux_err
        return pred, z

    def encode(self, x):
        # input (batch, seq_len, input_size)
        # output  (batch, seq_len, num_directions * hidden_size)
        x, (_, _) = self.enc(x)
        x = self.fce(x)
        return x

    def decode(self, dt, z):
        n, l = dt.shape
        z = self.do(z)
        x_lat = torch.zeros((n, l, self.nl + 1)).to(dt.device)
        new_z = z.view(-1, self.nl, 1).expand(-1, -1, l).transpose(1, 2)
        x_lat[:, :, :-1] = new_z
        x_lat[:, :, -1] = dt
        output, (_, _) = self.dec(x_lat)  # input shape (seq_len, batch, features)
        output = self.fcd(output).squeeze()
        return output.squeeze()


class GRUGMM(torch.nn.Module):
    def __init__(self, nin, nh, nl, ne, ngmm, nout, nlayers, do):
        super(GRUGMM, self).__init__()
        self.nh = nh
        self.nl = nl
        if nlayers >= 2:
            self.enc = torch.nn.GRU(input_size=nin, hidden_size=nh, num_layers=nlayers, dropout=do, batch_first=True)
            self.dec = torch.nn.GRU(input_size=nl + 1, hidden_size=nh, num_layers=nlayers, dropout=do, batch_first=True)
        else:
            self.enc = torch.nn.GRU(input_size=nin, hidden_size=nh, num_layers=nlayers, batch_first=True)
            self.dec = torch.nn.GRU(input_size=nl + 1, hidden_size=nh, num_layers=nlayers, batch_first=True)

        self.estimation_network = torch.nn.Sequential(
            torch.nn.Linear(nl + 4, ne),  # + 5
            torch.nn.Tanh(),
            torch.nn.Dropout(p=do),
            torch.nn.Linear(ne, ngmm)
            )

        self.fcd = torch.nn.Linear(nh, nout)
        self.fce = torch.nn.Linear(nh, nl)
        self.do = torch.nn.Dropout(p=do)

    def forward(self, x, m, s, seq_len, p=None):
        n, _, _ = x.shape
        z = self.encode(x)
        z = z[torch.arange(n), (seq_len - 1).type(dtype=torch.long)]
        pred = self.decode(x[:, :, 0], z)  # index: 0-time, 1-flux, 2-flux_err
        euc, cos = distances(x, pred)
        if p is None:
            zc = torch.cat((z, euc, cos, m.unsqueeze(-1), s.unsqueeze(-1)), dim=1)
        else:
            zc = torch.cat((z, euc, cos, m.unsqueeze(-1), s.unsqueeze(-1), p.unsqueeze(-1)), dim=1) 
        logits = self.estimation_network(zc)
        return pred, z, logits

    def encode(self, x):
        # input (batch, seq_len, input_size)
        # output  (batch, seq_len, num_directions * hidden_size)
        x, _ = self.enc(x)
        x = self.fce(x)
        return x

    def decode(self, dt, z):
        n, l = dt.shape
        z = self.do(z)
        x_lat = torch.zeros((n, l, self.nl + 1)).to(dt.device)
        new_z = z.view(-1, self.nl, 1).expand(-1, -1, l).transpose(1, 2)
        x_lat[:, :, :-1] = new_z
        x_lat[:, :, -1] = dt
        output, _ = self.dec(x_lat)  # input shape (seq_len, batch, features)
        output = self.fcd(output).squeeze()
        return output.squeeze()
        
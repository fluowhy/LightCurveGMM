import torch


def compute_params(z, gamma):
    # source https://github.com/danieltan07/dagmm/blob/master/model.py
    N = gamma.size(0)
    # K
    sum_gamma = torch.sum(gamma, dim=0)
    

    # K
    phi = sum_gamma / N

    # K x D
    mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
    # z = N x D
    # mu = K x D
    # gamma N x K

    # z_mu = N x K x D
    z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

    # z_mu_outer = N x K x D x D
    z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

    # K x D x D
    cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)

    return phi, mu, cov


def distances(x, x_pred):
    mask = (x[:, :, 0] != 0) * 1.
    x = x[:, :, 1]
    x_norm = (x.pow(2) * mask).sum(1, keepdim=True)
    x_pred_norm = (x_pred.pow(2) * mask).sum(1, keepdim=True)
    euclidean_distance = ((x_pred - x).pow(2) * mask).sum(dim=1, keepdim=True) / x_norm
    cosine_distance = (x * x_pred * mask).sum(1, keepdim=True) / x_norm / x_pred_norm
    return euclidean_distance, cosine_distance


class MLP(torch.nn.Module):
    def __init__(self, nin, nh, nout, do=0.5):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(nin, nh)
        self.fc2 = torch.nn.Linear(nh, nout)
        self.tanh = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(do)

    def forward(self, x):
        return self.fc2(self.dropout(self.tanh(self.fc1(x))))


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
    def __init__(self, nin, nh, nl, ne, ngmm, nout, nlayers, do, fold):
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
            torch.nn.Linear(nl + 5, ne) if fold else torch.nn.Linear(nl + 2, ne),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=do),
            torch.nn.Linear(ne, ngmm)
            )

        self.fcd = torch.nn.Linear(nh, nout)
        self.fce = torch.nn.Linear(nh, nl)
        self.do = torch.nn.Dropout(p=do)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, seq_len, p=None):
        n, _, _ = x.shape
        z = self.encode(x)
        z = z[torch.arange(n), (seq_len - 1).type(dtype=torch.long)]
        pred = self.decode(x[:, :, 0], z)  # index: 0-time, 1-flux, 2-flux_err
        euc, cos = distances(x, pred)
        if p is None:
            zc = torch.cat((z, euc, cos), dim=1)
        else:
            zc = torch.cat((z, euc, cos, m, s, p.unsqueeze(-1)), dim=1) 
        logits = self.estimation_network(zc)
        gamma = self.softmax(logits)
        phi, mu, cov = compute_params(zc, gamma)
        return pred, zc, logits, phi, mu, cov

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
        
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math


class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """

    def __init__(self, in_dims, out_dims, emb_size, lamda1, lamda2, time_type="cat", norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.lamda1 = lamda1
        self.lamda2 = lamda2

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        self.in_layers_ori = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.in_layers_sim = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        # self.out_layers_ori = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        self.out_layers_sim = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        # [Linear(in_features=1000, out_features=entity_n+10, bias=True)]
        self.decoder = nn.Linear(out_dims_temp[-1], out_dims_temp[-1])

        self.drop = nn.Dropout(dropout)
        self.drop_ori = nn.Dropout(dropout)
        self.drop_sim = nn.Dropout(dropout)
        self.init_weights()

    @staticmethod
    def init_p(layers):
        for layer in layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

    def init_weights(self):
        self.init_p(self.in_layers)
        self.init_p(self.out_layers)
        self.init_p(self.in_layers_ori)
        self.init_p(self.in_layers_sim)
        self.init_p(self.out_layers_sim)

        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, x_start, timesteps):
        sim_infos = torch.empty(x_start.shape).to(x.device)
        cor = torch.matmul(x_start, x_start.T)
        u_degree = torch.sum(x_start, dim=-1)
        x_shape = len(u_degree)
        u_d_e = u_degree.expand(x_shape, x_shape)
        weight = 1 / (u_d_e * u_d_e.T)
        res = cor*weight
        mask = torch.eye(res.shape[0], dtype=torch.bool).to(x.device)
        res_ = res.masked_fill_(mask, 0)
        value, index = torch.topk(res_, 1)
        index = index.squeeze()
        sim_infos[:, :] = x_start[index, :]
        sim_infos = torch.clamp((sim_infos - x_start), 0, 1).detach()

        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
            x_start = F.normalize(x_start)
            sim_infos = F.normalize(sim_infos)
        x = self.drop(x)
        x_0 = self.drop_ori(x_start)
        x_s = self.drop_sim(sim_infos)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        h_0 = torch.cat([x_0, emb], dim=-1)
        for i, layer in enumerate(self.in_layers_ori):
            h_0 = layer(h_0)
            h_0 = torch.tanh(h_0)
        h_s = torch.cat([x_s, emb], dim=-1)
        for i, layer in enumerate(self.in_layers_sim):
            h_s = layer(h_s)
            h_s = torch.tanh(h_s)
        h_in1 = h + self.lamda1 * h_0
        h_in2 = h + self.lamda2 * h_s

        for i, layer in enumerate(self.out_layers):
            h_in1 = layer(h_in1)
            if i != len(self.out_layers) - 1:
                h_in1 = torch.tanh(h_in1)
        for i, layer in enumerate(self.out_layers_sim):
            h_in2 = layer(h_in2)
            if i != len(self.out_layers_sim) - 1:
                h_in2 = torch.tanh(h_in2)
        h_res = h_in1 + h_in2
        return h_res


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# mepognn_encoder.py
# Minimal, self-contained encoder-ready version of your MepoGNN.
# Changes vs your original:
#   - stcell.forward returns (param_b, param_g, feat) with feat = ReLU(skip)
#   - mepognn.forward caches self.last_feat
#   - mepognn.encode(...) returns feat only (B, C_ctx, V, T_p)
#   - MepoGNNEncoder wraps a backbone and exposes a clean encoder interface

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- building blocks (same as your code) ----------
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()
    def forward(self,x, A):
        if len(A.shape) == 2:
            x = torch.einsum('vw, ncwl->ncvl', A, x)
        else:
            x = torch.einsum('nvw, ncwl->ncvl', A, x)
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=1):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order
    def forward(self, x, support):
        out = []
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

# ---------- stcell: returns feat (ReLU(skip)) ----------
class stcell(nn.Module):
    def __init__(self, num_nodes, dropout, in_dim, out_len, residual_channels, dilation_channels, skip_channels,
                 end_channels, kernel_size, blocks, layers):
        super(stcell, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.ln = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        receptive_field = 1
        self.supports_len = 2

        for b in range(blocks):
            additional_scope = 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                self.ln.append(nn.LayerNorm([residual_channels, num_nodes, (2 ** layers - 1) * blocks + 2 - receptive_field]))
                self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        self.end_conv_b1 = nn.Conv2d(in_channels=skip_channels * blocks * layers,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)
        self.end_conv_b2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_len,
                                    kernel_size=(1,1),
                                    bias=True)
        self.end_conv_g1 = nn.Conv2d(in_channels=skip_channels* blocks * layers,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)
        self.end_conv_g2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_len,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.receptive_field = receptive_field

    def forward(self, input, adp_g):
        """
        Args:
          input: (B, C_in, V, T_h)
          adp_g: list of 2 supports [A_forward, A_backward] shaped
                 - Adaptive: (V, V)
                 - Dynamic:  (B, V, V)

        Returns:
          param_b: (B, out_len, V, T_eff)  with T_eff aligned to prediction window
          param_g: (B, out_len, V, T_eff)
          feat   : (B, skip_channels*blocks*layers, V, T_eff)  <-- encoder embedding
        """
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        for i in range(self.blocks * self.layers):
            res = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = torch.cat((s, skip[:, :, :,  -s.size(3):]), dim=1)
            except Exception:
                skip = s

            x = self.gconv[i](x, adp_g)

            try:
                dense = dense[:, :, :, -x.size(3):]
            except Exception:
                dense = 0
            dense = res[:, :, :, -x.size(3):] + dense

            gate = torch.sigmoid(x)
            x = x * gate + dense * (1 - gate)
            x = self.ln[i](x)

        feat = F.relu(skip)  # <-- encoder feature for each node & step

        # original heads (kept intact)
        param_b = F.relu(feat)
        param_b = F.relu(self.end_conv_b1(param_b))
        param_b = torch.sigmoid(self.end_conv_b2(param_b))

        param_g = F.relu(feat)
        param_g = F.relu(self.end_conv_g1(param_g))
        param_g = torch.sigmoid(self.end_conv_g2(param_g))

        return param_b, param_g, feat

# ---------- optional SIRcell (unchanged) ----------
class SIRcell(nn.Module):
    def __init__(self):
        super(SIRcell, self).__init__()
    def forward(self, param_b: torch.Tensor, param_g: torch.Tensor, mob: torch.Tensor, SIR: torch.Tensor):
        if len(mob.shape) == 2:
            batch_size = SIR.shape[0]
            mob = mob.unsqueeze(0).expand(batch_size, -1, -1)
        num_node = SIR.shape[-2]
        S = SIR[..., [0]]
        I = SIR[..., [1]]
        R = SIR[..., [2]]
        pop = (S + I + R).expand(-1, num_node, num_node)
        propagtion = (mob/pop * I.expand(-1, num_node, num_node)).sum(1) +\
                     (mob/pop * I.expand(-1, num_node, num_node).transpose(1, 2)).sum(2)
        propagtion = propagtion.unsqueeze(2)
        I_new = param_b * propagtion
        R_t = I * param_g + R
        I_t = I + I_new - I * param_g
        S_t = S - I_new
        Ht_SIR = torch.cat((I_new, S_t, I_t, R_t), dim=-1)
        return Ht_SIR

# ---------- mepognn with encoder output ----------
class mepognn(nn.Module):
    def __init__(self, num_nodes, adpinit, glm_type, dropout=0.5, in_dim=4, in_len=14, out_len=14,
                 residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=2, layers=3):
        super(mepognn, self).__init__()
        self.stcell = stcell(num_nodes, dropout, in_dim, out_len, residual_channels, dilation_channels,
                             skip_channels, end_channels, kernel_size, blocks, layers)
        self.SIRcell = SIRcell()
        self.out_dim = out_len
        self.glm_type = glm_type
        self.in_len = in_len
        self.out_len = out_len

        if self.glm_type == 'Adaptive':
            log_g = torch.log(adpinit+1.0)
            self.max_log = log_g.max()
            self.g_rescaled = nn.Parameter(log_g/self.max_log, requires_grad=True)
        elif self.glm_type == 'Dynamic':
            self.inc_init = nn.Parameter(torch.empty(out_len, in_len), requires_grad=True)
            nn.init.normal_(self.inc_init, 1, 0.01)
            self.od_scale_factor = 3
        else:
            raise NotImplementedError('Invalid graph type.')
        self.last_feat = None  # for convenience

    def forward(self, x_node, SIR, od, max_od):
        """
        Original forward: returns mechanistic outputs (unchanged),
        but now also caches self.last_feat for external access.
        """
        if self.glm_type == 'Adaptive':
            mob = torch.exp(torch.relu(self.g_rescaled*self.max_log))
            g_adp = [mob / mob.sum(1, True), mob.T / mob.T.sum(1, True)]
            param_b, param_g, feat = self.stcell(x_node, g_adp)
            self.last_feat = feat  # (B, C_ctx, V, T_p)
            outputs_SIR = []
            SIR = SIR[:, -1, ...]
            for i in range(self.out_dim):
                NSIR = self.SIRcell(param_b[:, i, ...], param_g[:, i, ...], mob, SIR)
                SIR = NSIR[..., 1:]
                outputs_SIR.append(NSIR[..., [0]])

        elif self.glm_type == 'Dynamic':
            incidence = torch.softmax(self.inc_init, dim=1)     # (T_p, T_h)
            # od: (B, T_h, V, V, 1) or already (B, T_p, V, V, 1)
            if od.dim()==5 and od.shape[1] != self.out_len:
                # project history to future steps
                od = torch.einsum('kt, btnvw -> bknvw', incidence, od.squeeze(-1)).unsqueeze(-1)
            mob = od.squeeze(-1)                                # (B, T_p, V, V)
            g = mob.mean(1)                                     # (B, V, V)
            g_t = g.permute(0, 2, 1)
            g_dyn = [g / g.sum(2, True), g_t / g_t.sum(2, True)]
            param_b, param_g, feat = self.stcell(x_node, g_dyn)
            self.last_feat = feat
            outputs_SIR = []
            SIR = SIR[:, -1, ...]
            for i in range(self.out_dim):
                NSIR = self.SIRcell(param_b[:,i,...], param_g[:,i,...], mob[:,i,...]*max_od*self.od_scale_factor, SIR)
                SIR = NSIR[...,1:]
                outputs_SIR.append(NSIR[...,[0]])
        else:
            raise NotImplementedError

        outputs = torch.stack(outputs_SIR, dim=1)  # (B, T_p, V, 1)
        return outputs

    @torch.no_grad()
    def encode(self, x_node: torch.Tensor, SIR: torch.Tensor,
               od: Optional[torch.Tensor]=None, max_od: Optional[float]=None) -> torch.Tensor:
        """
        Encode a window into node-time embeddings.

        Args:
            x_node: (B, C_in, V, T_h)
            SIR   : (B, T_h, V, 3)   # kept for signature compatibility
            od    : For 'Dynamic':
                        (B, T_h, V, V, 1) or (B, T_p, V, V, 1).
                    For 'Adaptive': can be None.
            max_od: optional scalar for dynamic scaling (not used in encoder)
        Returns:
            feat: (B, C_ctx, V, T_p)
        """
        if self.glm_type == 'Adaptive':
            mob = torch.exp(torch.relu(self.g_rescaled*self.max_log))
            g_list = [mob / mob.sum(1, True), mob.T / mob.T.sum(1, True)]
        elif self.glm_type == 'Dynamic':
            assert od is not None, "encode(): 'Dynamic' requires od (history or projected)."
            incidence = torch.softmax(self.inc_init, dim=1)   # (T_p, T_h)
            if od.dim()==5 and od.shape[1] != self.out_len:
                od = torch.einsum('kt, btnvw -> bknvw', incidence, od.squeeze(-1)).unsqueeze(-1)
            mob = od.squeeze(-1)                              # (B, T_p, V, V)
            g = mob.mean(1)
            g_t = g.permute(0, 2, 1)
            g_list = [g / g.sum(2, True), g_t / g_t.sum(2, True)]
        else:
            raise NotImplementedError
        _, _, feat = self.stcell(x_node, g_list)              # (B, C_ctx, V, T_p)
        return feat

# ---------- tiny wrapper (optional) ----------
class MepoGNNEncoder(nn.Module):
    """
    Wrap a backbone mepognn and expose only encoder outputs.
    Usage:
        enc = MepoGNNEncoder(backbone)
        feat = enc(x_node, x_SIR, od, max_od)  # (B, C_ctx, V, T_p)
    """
    def __init__(self, backbone: mepognn):
        super().__init__()
        self.backbone = backbone
    @torch.no_grad()
    def forward(self, x_node, SIR, od=None, max_od=None):
        return self.backbone.encode(x_node, SIR, od=od, max_od=max_od)
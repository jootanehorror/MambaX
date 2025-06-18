import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
import math

def segsum(x):

    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool,
                      device=x.device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(
        T, T, dtype=torch.bool, device=x.device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x, A, B, C, chunk_size, initial_states=None, device = None):

    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)


    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    else:
        initial_states = initial_states.unsqueeze(1)
    
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(
        segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp",
                         C, states, state_decay_out)


    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state



class Mamba2(nn.Module):
    def __init__(
        self,
        dim,
        dim_inner,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        activation="swish",
        bias=False,
        conv_bias=True,

        chunk_size=256,

        use_cuda=False,
        use_mem_eff_path=False,

    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dim_inner = dim_inner
        self.headdim = headdim
        self.ngroups = ngroups
        self.nheads = self.dim_inner // self.headdim
        self.dt_limit = dt_limit

        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * dim_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(dim, d_in_proj, bias=bias)

        conv_dim = dim_inner + 2 * self.ngroups * d_state
        self.conv_dim = conv_dim
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=0,
        )

        self.act = nn.SiLU()

        dt = torch.exp(
            torch.rand(self.nheads,) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        A = torch.empty(self.nheads, dtype=torch.float32,).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.nheads,))
        self.D._no_weight_decay = True

        if use_cuda:
            if use_mem_eff_path:
                try:
                    from mamba_ssm.ops.triton.ssd_combined import  mamba_split_conv1d_scan_combined
                    self.mamba_split_conv1d_scan_combined =  mamba_split_conv1d_scan_combined
                    self.use_cuda = True
                    self.use_mem_eff_path = True
                except ImportError:
                    self.mamba_split_conv1d_scan_combined =  None
                    self.use_cuda = False
                    self.use_mem_eff_path = False


        
        try:
            from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
            self.norm = RMSNormGated(dim_inner, eps=1e-5, norm_before_gate=False)
        except ImportError:
            self.norm = RMSNormGated(dim_inner)


        self.out_proj = nn.Linear(dim_inner, dim, bias=bias)

    def forward(self, x, conv_cache=None, ssm_cache=None):

        A = -torch.exp(self.A_log)  # (nheads,)
        zxbcdt = self.in_proj(x)  # (batch, seqlen, d_in_proj)

        if self.use_cuda:
            return self._forward_cuda(zxbcdt, A)
        else:
            return self._forward_ssd(zxbcdt, A, conv_cache, ssm_cache)
    
    def _forward_cuda(self, zxbcdt, A):

        if self.use_mem_eff_path:
            out = self.mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=None,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=None,
            )

        return out
    

    def _forward_ssd(self, zxbcdt, A, conv_cache=None, ssm_cache=None):

        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.dim_inner,
                self.dim_inner + 2 * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        dt = F.softplus(dt + self.dt_bias)
        xBC = xBC.permute(0, 2, 1)

        if conv_cache is None:
            xBC = F.pad(xBC, (self.d_conv-1, 0, 0, 0))
        else:
            xBC = torch.cat([conv_cache, xBC], dim=-1)
        conv_cache = xBC[:, :, -(self.d_conv-1):]

        xBC = self.conv1d(xBC)
        xBC = self.act(xBC).permute(0, 2, 1)

        x, B, C = torch.split(xBC, [self.dim_inner, self.d_state, self.d_state], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)

        y, ssm_cache = ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.chunk_size,
            initial_states=ssm_cache,
        )

        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        out = self.out_proj(y)

        return out, conv_cache, ssm_cache

    def step(self, x, conv_cache, ssm_cache):

            zxbcdt = self.in_proj(x.squeeze(1))  # (B 2D)
            d_mlp = (zxbcdt.shape[-1] - 2 * self.dim_inner - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.dim_inner, self.dim_inner + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )

            xBC = torch.cat([conv_cache, xBC.unsqueeze(-1)], dim=-1)
            conv_cache = xBC[:, :, -(self.d_conv-1):]
            xBC = F.conv1d(xBC, weight=self.conv1d.weight, bias=self.conv1d.bias, groups=self.conv_dim)
            xBC = self.act(xBC).squeeze(-1)
    

            x, B, C = torch.split(xBC, [self.dim_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            A = -torch.exp(self.A_log.float()) 


            dt = F.softplus(dt + self.dt_bias)  
            dA = torch.exp(dt * A)  
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_cache.copy_(ssm_cache * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_cache, C)
            y = y + rearrange(self.D, "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            y = self.norm(y, z)

            out = self.out_proj(y)
            return out.unsqueeze(1), conv_cache, ssm_cache

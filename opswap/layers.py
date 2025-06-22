import torch
import torch.nn as nn

def create_piecewise_softmax_layers(dim=-1):
    class PiecewiseExp(nn.Module):
        def __init__(self):
            super().__init__()
            self.clamp_conv = nn.Conv1d(1, 1, kernel_size=1, bias=True)
            with torch.no_grad():
                self.clamp_conv.weight.fill_(1.0)
                self.clamp_conv.bias.fill_(0.0)
        
        def forward(self, x):
            original_shape = x.shape
            if x.dim() > 3:
                x = x.view(-1, 1, x.size(-1))
            elif x.dim() == 2:
                x = x.unsqueeze(1)
            
            x = torch.clamp(x, min=-4, max=4)
            
            exp_approx = torch.where(x < -2, torch.zeros_like(x), torch.where(x < 0, 0.5 + 0.25 * x, 1.0 + x))
 
            if len(original_shape) > 3:
                exp_approx = exp_approx.view(original_shape)
            elif len(original_shape) == 2:
                exp_approx = exp_approx.squeeze(1)
            
            return exp_approx
    
    return nn.Sequential(PiecewiseExp(),)

def create_conv1d_matmul(kernel_size):
    return nn.Conv1d(1, 1, kernel_size=kernel_size, bias=False)
    
def create_layernorm(features, weight, bias, eps=1e-5):
    class LayerNormLinearApprox(nn.Module):
        def __init__(self, n_features, gamma, beta, eps=1e-5):
            super().__init__()
            self.n_features = n_features
            self.eps = eps

            self.gamma = nn.Parameter(gamma.clone())
            self.beta = nn.Parameter(beta.clone()) if beta is not None else nn.Parameter(torch.zeros(n_features))

            self.mean_proj = nn.Linear(n_features, 1, bias=False)
            with torch.no_grad():
                self.mean_proj.weight.fill_(1.0 / n_features)
            self.mean_proj.weight.requires_grad = False

        def forward(self, x):
            orig_shape = x.shape
            x_flat = x.view(-1, self.n_features)

            mean = self.mean_proj(x_flat)
            x_centered = x_flat - mean

            sq = x_centered ** 2
            var = self.mean_proj(sq)
            inv_std = torch.rsqrt(var + self.eps)

            x_norm = x_centered * inv_std
            out = x_norm * self.gamma + self.beta

            return out.view(orig_shape)

    return LayerNormLinearApprox(features, weight, bias, eps)

def create_layernorm2d(channels, weight, bias, eps=1e-5):
    class LayerNorm2DApprox(nn.Module):
        def __init__(self, n_channels, gamma, beta, eps=1e-5):
            super().__init__()
            self.n_channels = n_channels
            self.eps = eps

            self.gamma = nn.Parameter(gamma.clone())
            self.beta = nn.Parameter(beta.clone()) if beta is not None else nn.Parameter(torch.zeros(n_channels))

            self.mean_conv = nn.Conv2d(n_channels, 1, kernel_size=1, bias=False)
            with torch.no_grad():
                self.mean_conv.weight.fill_(1.0 / n_channels)
            self.mean_conv.weight.requires_grad = False

            self.broadcast_conv = nn.Conv2d(1, n_channels, kernel_size=1, bias=False)
            with torch.no_grad():
                self.broadcast_conv.weight.fill_(1.0)
            self.broadcast_conv.weight.requires_grad = False

        def forward(self, x):
            mean = self.mean_conv(x)
            mean_broadcast = self.broadcast_conv(mean)

            x_centered = x - mean_broadcast
            var = self.mean_conv(x_centered ** 2) + self.eps
            std_broadcast = self.broadcast_conv(torch.sqrt(var))

            x_norm = x_centered / std_broadcast
            return x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)

    return LayerNorm2DApprox(channels, weight, bias, eps)

class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x) 

class Polynomial(nn.Module):
    def __init__(self, coeffs):
        super().__init__()
        self.coeffs = coeffs  # coeffs: list

    def forward(self, x):
        y = torch.zeros_like(x)
        power = torch.ones_like(x)
        for c in self.coeffs:
            y = y + c * power
            power = power * x
        return y

class StandardNormCDF(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = 0.2316419
        self.b = [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429]
        self.inv_sqrt_2pi = 0.3989422804
        self.poly_coeffs = self.b

        self.abs = Abs()

    def forward(self, x):
        a = self.abs(x)
        t = 1.0 / (1.0 + self.p * a)
        poly = self.poly_coeffs[0]*t
        for i in range(1, len(self.poly_coeffs)):
            poly = poly + self.poly_coeffs[i] * (t ** (i + 1))
        z = 0.5 * a * a
        exp_approx = 1 / (1 + z + 0.5 * z**2 + 0.16666666666666666 * z**3 + 0.041666666666666664 * z**4 + 0.008333333333333333 * z**5 + 0.001388888888888889 * z**6)
        cdf_approx = 1.0 - self.inv_sqrt_2pi * exp_approx * poly
        mask = (x >= 0).float()
        cdf_final = mask * cdf_approx + (1.0 - mask) * (1.0 - cdf_approx)
        return cdf_final

def create_gelu():
    class GELUApprox(nn.Module):
        def __init__(self):
            super().__init__()
            self.cdf = StandardNormCDF()
        def forward(self, x):
            return x * self.cdf(x)
    return GELUApprox()

def create_attention(q_proj, k_proj, v_proj, out_proj, num_heads, embed_dim):
    class AttentionApprox(nn.Module):
        def __init__(self, q_proj, k_proj, v_proj, out_proj, num_heads, embed_dim):
            super().__init__()
            self.q_proj = q_proj  # linear
            self.k_proj = k_proj  # linear
            self.v_proj = v_proj  # linear
            self.out_proj = out_proj  # linear
            self.num_heads = num_heads
            self.embed_dim = embed_dim
            self.head_dim = embed_dim // num_heads
            self.scale = self.head_dim ** -0.5
            
            self.score_conv = nn.Conv1d(self.head_dim, self.head_dim, kernel_size=1, bias=False)
            with torch.no_grad():
                self.score_conv.weight.copy_(torch.eye(self.head_dim).unsqueeze(-1))
            
            # softmax approx
            # self.softmax_layers = create_piecewise_softmax_layers(dim=-1)
            
        def forward(self, query, key=None, value=None, need_weights=False, attn_mask=None):
            if isinstance(query, tuple):
                query, key, value = query
            elif key is None:
                key = value = query
                
            if query.dim() == 3 and query.size(0) < query.size(1):
                seq_first = True
                query = query.transpose(0, 1)
                key = key.transpose(0, 1)
                value = value.transpose(0, 1)
            else:
                seq_first = False
                
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
            
            batch_size = q.size(0)
            seq_len = q.size(1)
            
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            scores = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, device=q.device)
            
            for i in range(seq_len):
                for j in range(seq_len):
                    qi = q[:, :, i, :]  # [batch, heads, head_dim]
                    kj = k[:, :, j, :]  # [batch, heads, head_dim]
                    scores[:, :, i, j] = (qi * kj).sum(dim=-1) * self.scale
            
            if attn_mask is not None:
                scores = scores + attn_mask
                
            attn_weights = torch.zeros_like(scores)
            for h in range(self.num_heads):
                for i in range(seq_len):
                    score_slice = scores[:, h, i, :]  # [batch, seq_len]
                    score_max = score_slice.max(dim=-1, keepdim=True)[0]
                    score_normalized = score_slice - score_max
                    exp_approx = torch.where(score_normalized < -2,
                                           torch.zeros_like(score_normalized),
                                           torch.where(score_normalized < 0,
                                                     0.5 + 0.25 * score_normalized,
                                                     1.0 + score_normalized))
                    sum_exp = exp_approx.sum(dim=-1, keepdim=True)
                    sum_exp = torch.clamp(sum_exp, min=1e-6)
                    attn_weights[:, h, i, :] = exp_approx / sum_exp
            
            attn_output = torch.zeros_like(v)
            for i in range(seq_len):
                for j in range(seq_len):
                    weight = attn_weights[:, :, i, j].unsqueeze(-1)  # [batch, heads, 1]
                    attn_output[:, :, i, :] += weight * v[:, :, j, :]
            
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
            
            output = self.out_proj(attn_output)
            
            if seq_first:
                output = output.transpose(0, 1)
                
            if need_weights:
                return output, attn_weights.mean(dim=1)
            else:
                return output, None
    
    return AttentionApprox(q_proj, k_proj, v_proj, out_proj, num_heads, embed_dim)
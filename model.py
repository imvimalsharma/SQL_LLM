import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

def precompute_theta_pos_frequencies(head_dim, seq_len, theta=10000.0):
    assert head_dim % 2 == 0
    exponent = torch.arange(0, head_dim, 2).float() / head_dim
    theta_freqs = 1.0 / (theta ** exponent)
    m = torch.arange(seq_len).float()
    freqs = torch.outer(m, theta_freqs)
    cos = torch.cos(freqs) # [seq_len, head_dim // 2]
    sin = torch.sin(freqs) # [seq_len, head_dim // 2]
    return cos, sin

def rotate_half(x):
    # Splits the last dimension in half and rotates
    d = x.shape[-1]
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    # x shape: [B, S, H, D]
    # cos, sin shape: [S, D // 2]
    B, S, H, D = x.shape
    
    # Repeat cos/sin to match full head_dim D, reshape to broadcast
    cos_d = torch.cat((cos, cos), dim=-1).view(1, S, 1, D)
    sin_d = torch.cat((sin, sin), dim=-1).view(1, S, 1, D)
    
    return (x * cos_d) + (rotate_half(x) * sin_d)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, max_seq_len):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )
        
    def forward(self, x, cos, sin):
        B, S, C = x.shape
        
        q = self.q_proj(x).view(B, S, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_head, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_head, self.head_dim)
        
        # Apply RoPE using precomputed cached sin/cos
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        q = q.transpose(1, 2) # [B, n_head, S, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Efficient PyTorch scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = att.masked_fill(self.mask[:, :, :S, :S] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v # [B, n_head, S, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, S, C)
        return self.out_proj(y)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, max_seq_len):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_seq_len)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        
    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class LLM(nn.Module):
    def __init__(self, vocab_size=4096, d_model=256, n_layer=6, n_head=8, d_ff=1024, max_seq_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, max_seq_len)
            for _ in range(n_layer)
        ])
        
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.token_emb.weight = self.lm_head.weight
        
        # Precompute and register RoPE frequencies to avoid recomputing on GPU
        cos, sin = precompute_theta_pos_frequencies(
            head_dim=d_model // n_head,
            seq_len=max_seq_len
        )
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        
    def forward(self, idx, targets=None):
        B, S = idx.shape
        
        x = self.token_emb(idx)
        
        # Slice registered RoPE buffer to current sequence length
        cos = self.cos[:S]
        sin = self.sin[:S]
        
        for layer in self.layers:
            x = layer(x, cos, sin)
            
        x = self.norm(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :]) # [B, 1, vocab_size]
            return logits, None

if __name__ == "__main__":
    print("Testing LLM model...")
    model = LLM(vocab_size=256, d_model=64, n_layer=2, n_head=4, d_ff=128, max_seq_len=64)
    x = torch.randint(0, 256, (2, 32))
    logits, loss = model(x, targets=x)
    print("Logits shape:", logits.shape)
    print("Loss value:", loss.item())
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {params / 1e6:.3f} Million")
    print("Model test passed successfully!")

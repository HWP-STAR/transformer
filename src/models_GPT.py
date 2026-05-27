import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MultiHeadAttention,PositionalEncoding,PositionwiseFeedForward

#===GPT model==
class GPTDecoderLayer(nn.Module):
    def __init__(self,d_model,n_head,d_ff,dropout=0.1):
        super().__init__()
        self.self_attention=MultiHeadAttention(d_model,n_head,dropout)
        self.feed_forward=PositionwiseFeedForward(d_model,d_ff,dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,x,mask=None):
        attn_output=self.self_attention(x,x,x,mask)
        x=self.norm1(x+self.dropout(attn_output))
        #ffN
        ff_output=self.feed_forward(x)
        x=self.norm2(x+self.dropout(ff_output))

        return x
class GPT(nn.Module):
    def __init__(self,vocab_size,d_model=512,n_head=8,
                 num_layers=6,d_ff=2048,max_len=5000,
                 dropout=0.1,device='cuda:0'):
        super().__init__()
        self.device=device
        self.vocab_size=vocab_size
        self.d_model=d_model

        #embedding
        self.token_embedding=nn.Embedding(vocab_size,d_model)
        self.positional_encoding=PositionalEncoding(d_model,max_len,dropout)

        #GPT decoder
        self.decoder_layers=nn.ModuleList([
            GPTDecoderLayer(d_model,n_head,d_ff,dropout) for _ in range(num_layers)
            ])
        self.lm_head=nn.Linear(d_model,vocab_size,bias=False)
        # 可选：将 token_embedding 的权重与 lm_head 绑定
        self.lm_head.weight = self.token_embedding.weight
        #token_embedding 的工作是“把人类语言翻译成机器向量”，而 lm_head 的工作是“把机器向量翻译回人类语言”

        self.dropout=nn.Dropout(p=dropout)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def generate_causal_mask(self,seq_len):
        """生成因果掩码（下三角矩阵），形状 [seq_len, seq_len]"""
        mask = torch.tril(torch.ones((seq_len, seq_len), device=self.device)).bool()
        # 扩展维度为 [1, 1, seq_len, seq_len]，以便在多头注意力中使用
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self,x):
        """
        Args:
            x: 输入 token 序列 [batch_size, seq_len]
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        causal_mask=self.generate_causal_mask(x.size(1))

        x=self.token_embedding(x)*math.sqrt(self.d_model)
        x=self.positional_encoding(x)
        
        # 通过所有解码层
        for layer in self.decoder_layers:
            x=layer(x,mask=causal_mask)

        logits=self.lm_head(x)
        return logits

    def generate(self,start_tokens,max_new_tokens,temperature=1.0,top_k=None):
        """
        简单的自回归生成函数
        Args:
            start_tokens: 初始 token 序列 [batch_size, seq_len]
            max_new_tokens: 最多生成的 token 数
            temperature: 采样温度
            top_k: 可选，top-k 采样
        Returns:
            生成的完整序列 [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        generated=start_tokens
        for _ in range(max_new_tokens):
            logits=self(generated)
            next_token_logits=logits[:,-1,:] / temperature

            if top_k is not None:
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                probs = F.softmax(top_k_values, dim=-1)
                next_token = top_k_indices.gather(-1, torch.multinomial(probs, 1))
            else:
                probs=F.softmax(next_token_logits,dim=-1)
                next_token=torch.multinomial(probs,1)

            generated=torch.cat([generated,next_token],dim=1)
        return generated

# ---------- 测试代码 ----------
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'using device: {device}')

    # 超参数
    vocab_size = 1000
    d_model = 512
    n_head = 8
    num_layers = 6
    d_ff = 2048
    max_len = 100
    dropout = 0.1
    batch_size = 2
    seq_len = 10

    model = GPT(vocab_size, d_model, n_head, num_layers, d_ff,
                max_len, dropout, device).to(device)

    # 随机输入（注意 0 通常用于 padding，这里输入有效 token 从 1 开始）
    x = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)

    logits=model(x)
    print('='*50)
    print(f'input shape: {x.shape}')
    print(f'logits shape: {logits.shape}')

    # 计算损失（假设 target 是输入右移一位，即语言建模任务）
    target = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    loss = loss_fn(logits.view(-1, vocab_size), target.view(-1))
    print(f"语言建模损失: {loss.item():.4f}")

    # 生成示例
    start = torch.tensor([[1, 2, 3, 4]], device=device)  # 起始序列
    generated = model.generate(start, max_new_tokens=10, temperature=0.8, top_k=40)
    print(f'start:{start}')
    print(f"生成序列: {generated.tolist()}")


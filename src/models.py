import math
import torch.nn.functional as F
import torch
import torch.nn as nn
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=4, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # 初始化参数
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, idx):
        B, T = idx.shape
        # 位置索引 (0 到 T-1)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)
        # 词嵌入 + 位置嵌入
        x = self.token_embedding(idx) + self.position_embedding(pos)
        # 通过每个 Transformer Block
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffwd = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 准备因果掩码 (上三角矩阵，禁止看到未来)
        B, T, C = x.shape
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        # 自注意力 (需要 mask)
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)
        ff_out = self.ffwd(x)
        x = x + self.dropout(ff_out)
        x = self.ln2(x)
        return x
#========================================
#Transformer model
class MultiHeadAttention(nn.Module):
    """多头自注意力（Multi-Head Attention）
    将输入线性映射到多组 Q, K, V，分别计算注意力后拼接并线性变换。
    """
    def __init__(self, d_model, n_head, dropout=0.1):
        """
        Args:
            d_model: 输入/输出的特征维度
            n_head: 注意力头的数量（必须能整除 d_model）
            dropout: 注意力权重的 dropout
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"
        
        self.d_model = d_model
        self.n_head = n_head
        self.d_k=d_model//n_head

        
        # 定义线性变换层：用于生成 Q, K, V (论文中每个头都有自己的权重，这里合并为三个大矩阵)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)

    def scaled_dot_product_attention(self,Q,K,V,mask=None):
       # 计算 Q 和 K 的点积，除以 sqrt(d_k) 缩放
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [..., seq_len, seq_len]

        if mask is not None:
            scores=scores.masked_fill(mask==0,-1e9)

        attn_weights=F.softmax(scores,dim=-1)
        attn_weights=self.dropout(attn_weights)

        output=torch.matmul(attn_weights,V)

        return output,attn_weights
    
    def forward(self,query,key,value,mask=None):
        batch_size=query.size(0)

                # 1. 线性变换并拆分成多头
        Q = self.W_q(query).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # [batch, n_head, query_len, d_k]
        K = self.W_k(key).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)    # [batch, n_head, key_len, d_k]
        V = self.W_v(value).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # [batch, n_head, key_len, d_k]

        attn_output,_=self.scaled_dot_product_attention(Q,K,V,mask)

        #cat,mix
        attn_output=attn_output.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
        output=self.W_o(attn_output)
        return output

class PositionalEncoding(nn.Module):
    """位置编码（Positional Encoding）
    为序列中的每个位置添加位置信息，弥补自注意力机制无法感知顺序的缺点。
    论文使用正弦和余弦函数生成位置编码，不参与训练。
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: 词嵌入的维度（必须与模型内部维度一致）
            max_len: 最大序列长度（预先生成足够长的位置编码）
            dropout: Dropout 概率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建一个 [max_len, d_model] 的空白位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        
        # 位置索引 [0,1,2,... max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # shape: [max_len, 1]
        
        # 计算分母项：10000^(2i/d_model)
        # 其中 i 是维度索引的一半（因为一对正弦余弦对应两个维度）
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数位置用正弦，奇数位置用余弦
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维度
        
        # 增加 batch 维度，以便与输入相加 [max_len, d_model] -> [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer（不参与梯度更新，但会随着模型一起保存和移动到设备）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x + positional encoding: [batch_size, seq_len, d_model]
        """
        # 取前 seq_len 个位置编码，加到输入上
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    # ReLU
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.linear1=nn.Linear(d_model,d_ff)
        self.linear2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self,d_model,n_head,d_ff,dropout=0.1):# d_ff : hide lay
        super().__init__()
        self.self_attention=MultiHeadAttention(d_model,n_head,dropout)
        self.feed_forward=PositionwiseFeedForward(d_model,d_ff,dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,x,mask=None):
        attn_output=self.self_attention(x,x,x,mask)
        x=self.norm1(x+self.dropout(attn_output))

        ff_output=self.feed_forward(x)
        x=self.norm2(x+self.dropout(ff_output))

        return x

class DecoderLayer(nn.Module):
    def __init__(self,d_model,n_head,d_ff,dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x,encoder_output,src_mask=None,tgt_mask=None):
         # 1. 掩码自注意力（只能看到当前位置之前的输出）
         attn_self=self.self_attention(x,x,x,tgt_mask)
         x=self.norm1(x+self.dropout(attn_self))

         attn_cross=self.cross_attention(x,encoder_output,encoder_output,src_mask)
         x=self.norm2(x+self.dropout(attn_cross))

         #FNN
         ff_output=self.feed_forward(x)
         x=self.norm3(x+self.dropout(ff_output))
         return x

class Transformer(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,d_model=512,n_head=8,
                 num_encoder_layers=6,num_decoder_layers=6,d_ff=2048,
                 max_len=5000,dropout=0.1,device='cuda:0'):

        super().__init__()
        self.device=device
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.encoder_layers=nn.ModuleList([
            EncoderLayer(d_model,n_head,d_ff,dropout) for _ in range(num_encoder_layers)
            ])
        self.decoder_layers=nn.ModuleList([
            DecoderLayer(d_model,n_head,d_ff,dropout) for _ in range(num_decoder_layers)
            ])

        self.fc_out=nn.Linear(d_model,tgt_vocab_size)
        self.dropout=nn.Dropout(p=dropout)
        # 参数初始化（使用 Xavier 均匀分布）
        self._init_parameters()

    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_src_mask(self,src):
        src_mask=(src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def generate_tgt_mask(self,tgt):
        #1 padding mask
        tgt_pad_mask=(tgt != 0).unsqueeze(1).unsqueeze(2)

        # 2. 后续掩码（禁止看到未来的位置）
        tgt_len = tgt.size(1)
        subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device)).bool()
        # subsequent_mask 形状 [tgt_len, tgt_len]，下三角为 True

        tgt_mask=tgt_pad_mask & subsequent_mask.unsqueeze(0).unsqueeze(0)
        return tgt_mask

    def forward(self,src,tgt):
        """
        Args:
            src: 源序列 [batch_size, src_len]
            tgt: 目标序列（训练时通常输入 decoder 的是起始符号 + 前 tgt_len-1 个真实 token）
                 [batch_size, tgt_len]
        Returns:
            logits: [batch_size, tgt_len, tgt_vocab_size] (未经过 softmax)
        """
        src_mask=self.generate_src_mask(src)
        tgt_mask=self.generate_tgt_mask(tgt)

        # 1. 词嵌入 + 缩放 (论文中乘以 sqrt(d_model) 有助于训练稳定性)
        src_emb = self.encoder_embedding(src) * math.sqrt(self.encoder_embedding.embedding_dim)
        src_emb=self.positional_encoding(src_emb)

        enc_output=src_emb
        for layer in self.encoder_layers:
            enc_output=layer(enc_output,src_mask)

        # 解码器
        tgt_emb = self.decoder_embedding(tgt) * math.sqrt(self.decoder_embedding.embedding_dim)
        tgt_emb = self.positional_encoding(tgt_emb)

        dec_output=tgt_emb
        for layer in self.decoder_layers:
            dec_output=layer(dec_output,enc_output,src_mask,tgt_mask)

        logits=self.fc_out(dec_output)
        return logits

    def encode(self,src):
        src_mask=self.generate_src_mask(src)
        src_emb=self.encoder_embedding(src)*math.sqrt(self.encoder_embedding.embedding_dim)
        src_emb=self.positional_encoding(src_emb)
        enc_output=src_emb
        for layer in self.encoder_layers:
            enc_output=layer(enc_output,src_mask)
        return enc_output,src_mask

    def decode(self,tgt,encoder_output,src_mask):
        tgt_mask=self.generate_tgt_mask(tgt)
        tgt_emb = self.decoder_embedding(tgt) * math.sqrt(self.decoder_embedding.embedding_dim)
        tgt_emb = self.positional_encoding(tgt_emb)
        dec_output=tgt_emb
        for layer in self.decoder_layers:
            dec_output=layer(dec_output,encoder_output,src_mask,tgt_mask)
        logits=self.fc_out(dec_output)
        return logits

        
    


#=================================================
if __name__=="__main__":
    device=torch.device('cuda:0')
    print(f'using device:{device}')
    '''
    vocab_size=5000
    model=MiniTransformer(vocab_size=vocab_size).to(device)

    x=torch.randint(0,vocab_size,(2,64)).to(device) #(batch_size,seq_len)
    y=torch.randint(0,vocab_size,(2,64)).to(device)
    
    logits=model(x)
    loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    print(f"测试通过，损失: {loss.item():.4f}")
    '''
    src_vocab_size=1000
    tgt_vocab_size=1000
    d_model=512
    n_head=8
    num_encoder_layers=6
    num_decoder_layers=6
    d_ff=2048
    max_len=100
    dropout=0.1
    batch_size=2
    src_len=10
    tgt_len=12

    model=Transformer(src_vocab_size,tgt_vocab_size,d_model,n_head,
                      num_encoder_layers,num_decoder_layers,d_ff,
                      max_len,dropout,device).to(device)


    # 生成随机输入（模拟 batch）
    src = torch.randint(1, src_vocab_size, (batch_size, src_len)).to(device)   # 0 是 padding 索引
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len)).to(device)
    
    logits=model(src,tgt)
    print('='*50)
    print(f'src.shape:{src.shape}')
    print(f'tgt.shape:{tgt.shape}')
    print(f'logits.shape:{logits.shape}')

    cri=nn.CrossEntropyLoss(ignore_index=0)
    loss = cri(logits.view(-1, tgt_vocab_size), tgt.view(-1))
    print(f"损失值: {loss.item():.4f}")


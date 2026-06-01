import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import MultiHeadAttention, PositionwiseFeedForward

# ======================= MoE 核心组件（新增加的代码） =======================
class MoEExpert(nn.Module):
    """单个专家：就是一个普通的 FFN"""
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.ffn=PositionwiseFeedForward(d_model,d_ff,dropout)

    def forward(self,x):
        return self.ffn(x)

class TopKRouter(nn.Module):
    """
    路由（门控）网络：为每个 token 选择 top-k 个专家，并计算权重。
    原理：输入 x [batch, seq_len, d_model] -> 线性层输出 logits [batch, seq_len, num_experts]
         -> 取 top-k 索引和权重（softmax 归一化）。
    """
    def __init__(self,d_model,num_experts,top_k=2):
        super().__init__()
        self.num_experts=num_experts
        self.top_k=top_k
        self.gate=nn.Linear(d_model,num_experts,bias=False)

    def forward(self,x):
        # x: [batch, seq_len, d_model]
        logits=self.gate(x)
        topk_logits,topk_indices=torch.topk(logits,self.top_k,dim=-1)
        # 对 top-k 得分做 softmax（在专家维度上归一化）
        weights=F.softmax(topk_logits,dim=-1) #(batch,seq_len,top_k)

        return topk_indices,weights
class MoELayer(nn.Module):
    """
    混合专家层（Mixture of Experts Layer）
    包含 num_experts 个并行的 FFN 专家，以及一个路由网络。
    每个 token 独立地选择 top_k 个专家，并将它们的输出加权求和。
    同时计算辅助损失（负载均衡损失），用于鼓励专家使用均衡。
    """
    def __init__(self,d_model,d_ff,num_experts=8,top_k=2,dropout=0.1,aux_loss_weight=0.01):
        super().__init__()
        self.num_experts=num_experts
        self.top_k=top_k
        self.aux_loss_weight=aux_loss_weight
        # 创建多个专家（每个专家是一个独立的 FFN）
        self.experts=nn.ModuleList([
            MoEExpert(d_model,d_ff,dropout) for _ in range(num_experts)
            ])
        self.router=TopKRouter(d_model,num_experts,top_k)

    def forward(self,x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
            aux_loss: 标量，辅助损失（负载均衡损失）
        """
        batch_size,seq_len,d_model=x.shape
        # 1. 路由：得到每个 token 选择的专家索引和权重
        expert_indices,weights=self.router(x)
        # 2. 计算每个专家的输出（需要高效实现：可用 scatter 或循环）
        final_output=torch.zeros_like(x)
        total_tokens=batch_size*seq_len
        expert_count=torch.zeros(self.num_experts,device=x.device)

        # 对每个专家，收集被分配到的 token，批量计算输出
        for expert_idx in range(self.num_experts):
            # 找出哪些 token 选择了当前专家（一个 token 可能被多个专家选中，因为 top_k>1）
            # 方法：构造 mask，形状 [batch, seq_len, top_k]
            mask=(expert_indices==expert_idx)
            if mask.any():
                # 对于每个位置，可能多个 top_k 位置都有该专家，但我们只需收集一次。
                # 简化：将 mask 展平，得到所有选中该专家的 token 的线性索引
                # 但为了保留 batch 和 seq 信息，我们使用 where 获取坐标
                # 注意：同一个 token 可能因为 top_k>1 而多次选中同一专家？一般不会（top_k 是不同专家）
                # 所以我们直接对每个 token，只要该专家在它的 top_k 中，就处理一次。
                # 实际生产环境会使用更高效的实现，这里教学演示使用循环。
                indices = torch.nonzero(mask, as_tuple=True)  # (batch_idx, seq_idx, topk_idx) 三个张量
                # 提取对应位置的输入特征
                selected_x = x[indices[0], indices[1], :]          # [num_selected, d_model]
                # 计算该专家的输出
                expert_out = self.experts[expert_idx](selected_x)  # [num_selected, d_model]
                # 获取对应的权重（权重与专家一一对应，每个选择有独立权重）
                selected_weights = weights[indices[0], indices[1], indices[2]]  # [num_selected]
                # 累加到 final_output 的对应位置（每个 token 可能被多个专家加和）
                final_output[indices[0], indices[1], :] += expert_out * selected_weights.unsqueeze(-1)
                # 统计每个专家被选中的 token 数量（不计权重，只计次数）
                expert_count[expert_idx] += selected_x.size(0)

        # 3. 计算辅助损失（负载均衡损失）
        # 原理：希望每个专家被选中的概率大致相等。使用专家被选中的频率与均匀分布的 KL 散度。
        # 标准公式：aux_loss = num_experts * sum(f_i * P_i)，其中 f_i 是专家被选中的比例，
        # P_i 是路由概率的平均值。简化版本：aux_loss = num_experts * sum(f_i * g_i)
        # 这里我们采用 Switch Transformer 论文中的形式：
        # aux_loss = num_experts * sum( (count_i / total_tokens) * (mean(softmax(logits)_i) ) )
        # 其中 count_i 是专家 i 被选中的 token 数（每个 token 可能被多个专家选中，但 top_k 中每个选择算一次）
        # 更严谨地，应该是每个 token 对每个专家的路由概率（gate logits 的 softmax）的平均值，乘以该专家被选中的频率。
        # 为了简化，使用常用简化版：
        # 计算每个专家的平均路由概率（在所有 token 上对专家 i 的 softmax 概率的平均）
        router_logits = self.router.gate(x)  # [B, L, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)  # [B, L, num_experts]
        mean_router_probs = router_probs.mean(dim=(0,1))  # [num_experts]
        # 专家被选中的频率（fraction of tokens assigned to each expert）
        # 注意：一个 token 在 top_k 中可能选择多个专家，所以 total_tokens 应视为每个 token 的每个选择。
        # 实际论文计算频率时，每个 token 的每个选择都计入一次。
        total_selections = total_tokens * self.top_k
        fraction_per_expert = expert_count / total_selections  # [num_experts]
        # 辅助损失 = num_experts * sum(fraction_per_expert * mean_router_probs)
        aux_loss = self.num_experts * torch.sum(fraction_per_expert * mean_router_probs)
        # 乘以权重系数
        aux_loss = self.aux_loss_weight * aux_loss

        return final_output, aux_loss

# ======================= 改造后的 GPT-MoE Block =======================
class GPTMoEBlock(nn.Module):
    """
    GPT Block with MoE (替换原 FFN 为 MoE 层)
    保留因果自注意力部分，将原来的 FFN 替换为 MoELayer。
    仍使用 Pre-LN 结构。
    """
    def __init__(self,d_model,n_head,d_ff,num_experts=8,top_k=2,dropout=0.1,aux_loss_weight=0.01):
        super().__init__()
        self.ln1=nn.LayerNorm(d_model)
        self.attn=MultiHeadAttention(d_model,n_head,dropout)
        self.ln2=nn.LayerNorm(d_model)
        # PFF -> MoE
        self.moe=MoELayer(d_model,d_ff,num_experts,top_k,dropout,aux_loss_weight)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,mask=None):
        residual=x
        x=self.ln1(x)
        x=self.attn(x,x,x,mask)
        x=residual+self.dropout(x)

        #MoE part
        residual=x
        x=self.ln2(x)
        moe_out,aux_loss=self.moe(x)
        x=residual+self.dropout(moe_out)
        return x,aux_loss
# ======================= 完整的 GPT-MoE 模型 =======================
class GPTMoE(nn.Module):
    """
    基于 GPT 的 MoE 语言模型
    支持自回归生成，每一层的 FFN 被替换为 MoE 层，并累加所有层的辅助损失。
    """
    def __init__(self, vocab_size, d_model=512, n_head=8, num_layers=6,
                 d_ff=2048, max_seq_len=1024, dropout=0.1,
                 num_experts=8, top_k=2, aux_loss_weight=0.01, device='cuda'):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.aux_loss_weight = aux_loss_weight

        # 词嵌入与位置嵌入（与原 GPT 相同）
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.blocks=nn.ModuleList([
            GPTMoEBlock(d_model,n_head,d_ff,num_experts,top_k,dropout,aux_loss_weight)
            for _ in range(num_layers)
            ])
        self.ln_f=nn.LayerNorm(d_model)
        self.lm_head=nn.Linear(d_model,vocab_size,bias=False)

        # 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self,idx,return_aux_loss=False):
        """
        Args:
            idx: [batch_size, seq_len]
            return_aux_loss: 是否返回辅助损失（训练时返回，推理时可忽略）
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            aux_loss_total: 可选，所有 MoE 层的辅助损失之和
        """
        B, T = idx.shape
        positions = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)

        x=self.token_embedding(idx)*math.sqrt(self.d_model)
        x=x+self.position_embedding(positions)

        # 因果掩码（下三角为 True 表示允许 attend）
        causal_mask = torch.tril(torch.ones(T, T, device=idx.device)).bool()

        # 累计辅助损失
        total_aux_loss = 0.0
        
        for block in self.blocks:
            x,aux_loss=block(x,mask=causal_mask)
            total_aux_loss+=aux_loss

        x=self.ln_f(x)
        logits = self.lm_head(x)

        if return_aux_loss:
            return logits, total_aux_loss
        else:
            return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """自回归生成，与原始 GPT 相同（推理时无需辅助损失）"""
        self.eval()
        for _ in range(max_new_tokens):
            if idx.shape[1] > self.max_seq_len:
                idx = idx[:, -self.max_seq_len:]
            logits = self(idx, return_aux_loss=False)   # 推理时不需 aux loss
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device: {device}')

    vocab_size = 10000
    # 实例化 MoE-GPT 模型
    model = GPTMoE(
        vocab_size=vocab_size,
        d_model=256,
        n_head=4,
        num_layers=2,          # 为了快速测试，只堆2层
        d_ff=512,
        max_seq_len=128,
        dropout=0.1,
        num_experts=4,         # 4个专家
        top_k=2,               # 每个token选择2个专家
        aux_loss_weight=0.01,
        device=device
    ).to(device)

    # 模拟输入
    x = torch.randint(1, vocab_size, (2, 32)).to(device)   # (batch, seq_len)
    logits, aux_loss = model(x, return_aux_loss=True)

    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Auxiliary loss: {aux_loss.item():.6f}")

    # 计算语言建模损失（需加上辅助损失）
    y = torch.randint(1, vocab_size, (2, 32)).to(device)
    ce_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    total_loss = ce_loss + aux_loss
    total_loss.backward()

    print(f"CE loss: {ce_loss.item():.4f}, Total loss: {total_loss.item():.4f}")

    # 生成示例
    start = torch.tensor([[1, 2, 3]]).to(device)
    generated = model.generate(start, max_new_tokens=10, temperature=0.8, top_k=40)
    print(f"Generated shape: {generated.shape}")
    print(f'generated:{generated}\nstart:{start}')


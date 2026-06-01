import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.net=nn.Linear(input_dim,output_dim)

    def forward(self,x):
        # x: [batch_size, input_dim]
        return self.net(x)

# ===============================
# 2. 定义 MoE 层
# ===============================
class SimpleMoE(nn.Module):
    def __init__(self,num_experts,input_dim,output_dim,top_k=1):
        super().__init__()

        self.num_experts=num_experts
        self.top_k=top_k

        self.experts=nn.ModuleList([
            Expert(input_dim,output_dim) 
            for _ in range(num_experts)
            ])
        self.router=nn.Linear(input_dim,num_experts)

    def forward(self,x):
        router_logits=self.router(x)
        router_probs=F.softmax(router_logits,dim=-1)

        topk_probs,topk_indices=torch.topk(
            router_probs,self.top_k,dim=-1
                )
        # ---------- Step 3: 初始化最终输出 ----------
        final_output=torch.zeros(
            x.size(0),self.experts[0].net.out_features,
            device=x.device
                )
        for i in range(self.top_k):
            expert_idx=topk_indices[:,i]

            expert_weight=topk_probs[:,i].unsqueeze(-1)
            for b in range(x.size(0)):
                idx=expert_idx[b]
                weight=expert_weight[b]

                expert_out=self.experts[idx](x[b].unsqueeze(0))

                # 加权累加到最终输出
                final_output[b] += weight * expert_out.squeeze(0)
        return final_output

if __name__=="__main__":
    batch_size = 4
    input_dim = 8
    output_dim = 4
    num_experts = 2

    model=SimpleMoE(
        num_experts=num_experts,
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=1
            )
    x = torch.randn(batch_size, input_dim)
    y = model(x)

    print("输入 shape:", x.shape)
    print("输出 shape:", y.shape)
    print("输出值:\n", y)

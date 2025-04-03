# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MutualInformationLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super(MutualInformationLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, rep_time, rep_text):
#         # Ensure inputs are 2D tensors
#         if rep_time.dim() > 2:
#             rep_time = rep_time.mean(dim=1)  # Average over sequence length
#         if rep_text.dim() > 2:
#             rep_text = rep_text.mean(dim=1)  # Average over sequence length
            
#         # Normalize representations
#         rep_time_norm = F.normalize(rep_time, dim=-1)
#         rep_text_norm = F.normalize(rep_text, dim=-1)
        
#         # Compute similarity matrix
#         sim_matrix = torch.matmul(rep_time_norm, rep_text_norm.T) / self.temperature
        
#         # Create labels for positive pairs
#         labels = torch.eye(rep_time.size(0)).to(rep_time.device)
        
#         # Compute loss using InfoNCE
#         exp_sim = torch.exp(sim_matrix)
#         pos_sim = torch.sum(exp_sim * labels, dim=1)
#         neg_sim = torch.sum(exp_sim * (1 - labels), dim=1)
#         loss = -torch.mean(torch.log(pos_sim / (pos_sim + neg_sim)))
        
#         return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class MutualInformationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine = nn.CosineSimilarity(dim=1)

    def forward(self, x, y):
        # x: projected time series representation [B, D]
        # y: LLM representation [B, D]
        pos_sim = self.cosine(x, y)  # [B]
        # negative: all other pairs in batch
        sim_matrix = torch.mm(x, y.T)  # [B, B]
        neg_mask = 1 - torch.eye(sim_matrix.size(0), device=sim_matrix.device)
        neg_sim = sim_matrix * neg_mask
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(neg_sim).sum(dim=1) + 1e-8)).mean()
        return loss

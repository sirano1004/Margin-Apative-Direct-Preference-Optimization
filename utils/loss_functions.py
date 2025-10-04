import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaDPOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        #self.beta = beta

    def forward(self, 
                log_probs_y1_policy: torch.Tensor,
                log_probs_y2_policy: torch.Tensor,
                log_probs_y1_ref: torch.Tensor,
                log_probs_y2_ref: torch.Tensor,
                choices: torch.Tensor,
                beta: torch.Tensor):
        
        r1 = beta * (log_probs_y1_policy - log_probs_y1_ref)
        r2 = beta * (log_probs_y2_policy - log_probs_y2_ref)
        logits = torch.stack([r1, r2], dim=1) 

        loss = F.cross_entropy(logits, choices.long())
        
        return loss

class DPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, 
                log_probs_y1_policy: torch.Tensor,
                log_probs_y2_policy: torch.Tensor,
                log_probs_y1_ref: torch.Tensor,
                log_probs_y2_ref: torch.Tensor,
                choices: torch.Tensor):
        
        r1 = self.beta * (log_probs_y1_policy - log_probs_y1_ref)
        r2 = self.beta * (log_probs_y2_policy - log_probs_y2_ref)
        logits = torch.stack([r1, r2], dim=1) 

        loss = F.cross_entropy(logits, choices)
        
        return loss

class RewardLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
                r1: torch.Tensor,
                r2: torch.Tensor,
                choices: torch.Tensor):
        
        logits = torch.stack([r1, r2], dim=1) 

        loss = F.cross_entropy(logits, choices)
        
        return loss

class IPOLoss(nn.Module):
    def __init__(self, beta = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, 
                log_probs_y1_policy: torch.Tensor,
                log_probs_y2_policy: torch.Tensor,
                log_probs_y1_ref: torch.Tensor,
                log_probs_y2_ref: torch.Tensor,
                choices: torch.Tensor):
        
        r1 = (log_probs_y1_policy - log_probs_y1_ref)
        r2 = (log_probs_y2_policy - log_probs_y2_ref)
        # When y1 > y2
        loss_y1_y2 = ((r1- r2) - 0.5/self.beta) ** 2 
        loss_y2_y1 = ((r2- r1) - 0.5/self.beta) ** 2 

        loss = (1-choices) * loss_y1_y2 + choices * loss_y2_y1

        return loss.mean()
    

class MADPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1, c_max = 2, c_min = 0.5, lmbda = 1, tau = 2):
        super().__init__()
        self.beta = beta
        self.c_max = c_max
        self.c_min = c_min
        self.lmbda = lmbda
        self.tau = tau
    
    def coef(self,reward_margin):

        coef = (
            self.c_min + 
            (self.c_max - self.c_min) / 
            (1 + (self.c_max - 1)/(1 - self.c_min) * torch.exp(self.lmbda*(reward_margin - self.tau)))
        )

        return coef
    

    def forward(self, 
                log_probs_y1_policy: torch.Tensor,
                log_probs_y2_policy: torch.Tensor,
                log_probs_y1_ref: torch.Tensor,
                log_probs_y2_ref: torch.Tensor,
                reward1: torch.Tensor,
                reward2: torch.Tensor,
                choices: torch.Tensor):
        reward_margin = reward1 - reward2
        coef = self.coef(torch.abs(reward_margin))

        w1 = torch.sigmoid(coef * reward_margin) / torch.sigmoid(reward_margin) * (reward_margin > -self.tau) + 1 * (reward_margin <= -self.tau)
        w2 = torch.sigmoid(-coef * reward_margin) / torch.sigmoid(-reward_margin) * (-reward_margin > -self.tau) + 1 * (-reward_margin <= -self.tau)
        weights = torch.stack([w1, w2], dim=1) 

        r1 = self.beta * (log_probs_y1_policy - log_probs_y1_ref)
        r2 = self.beta * (log_probs_y2_policy - log_probs_y2_ref)
        logits = torch.stack([r1, r2], dim=1) 

        sample_weights = weights[torch.arange(len(choices)), choices]

        loss = F.cross_entropy(logits, choices, reduction='none')
        loss = (loss * sample_weights).mean()
        
        return loss
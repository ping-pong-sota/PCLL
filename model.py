import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F

class ProtPLL(nn.Module):

    def __init__(self, args, base_encoder):
        super().__init__()
        
        pretrained = args.dataset == 'cub200'
        self.encoder_q1 = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch, pretrained=pretrained)
        self.register_buffer("prototypes1", torch.zeros(args.num_class,args.low_dim))

    def reset_prototypes(self, prototypes1, prototypes2):
        self.prototypes1 = prototypes1

    def forward(self, img_q1, partial_Y=None, args=None, eval_only=False):
        output1, q1 = self.encoder_q1(img_q1)
        output = output1
        if eval_only:
            return output

        predicetd_scores1 = torch.softmax(output1, dim=1) * partial_Y
        max_scores1, pseudo_labels1 = torch.max(predicetd_scores1, dim=1)

        prototypes1 = self.prototypes1.clone().detach()
        logits_prot1 = torch.mm(q1, prototypes1.t())
        score_prot1 = torch.softmax(logits_prot1, dim=1)
        score_prot = score_prot1

        q1 = q1.detach()
        pseudo_labels1 = pseudo_labels1.detach()
        max_scores1 = max_scores1.detach()

        for feat, label, max_score in zip(q1, pseudo_labels1, max_scores1):
            self.prototypes1[label] = self.prototypes1[label]*args.proto_m + (1-args.proto_m)*feat
        # normalize prototypes    
        self.prototypes1 = F.normalize(self.prototypes1, p=2, dim=1)

        return output, score_prot, prototypes1,  q1, score_prot1


@torch.no_grad()
def get_entropy(logits):
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + 1e-7) # avoid NaN
    return -(probs*log_probs).sum(1)
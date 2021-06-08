from torch import nn
import torch.nn.functional as F


class SymmetricLoss(nn.Module):
    
    def __init__(self, metric):
      super().__init__()
      self.metric = metric

    def forward(self, z1, z2, p1, p2):
      loss = 0.5 * self.metric(p1, z2) + 0.5 * self.metric(p2, z1)
      return loss


def cosine_with_stopgrad(p,z):
    z = z.detach() # stop gradient
    p = F.normalize(p, p=2, dim=1)
    z = F.normalize(z, p=2, dim=1)
    return -(p*z).sum(dim=1).mean()

import torch
import torchvision.transforms as T
    
class SimSiamAug():
    
    def __init__(self):
        
        self.transform = T.Compose([
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply(torch.nn.ModuleList([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]), p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

    def __call__(self, x):
    
        aug1 = self.transform(x)
        aug2 = self.transform(x)
    
        return aug1, aug2


def baseline(train=True):

    if train == True:
        baseline_transforms = T.Compose([
                      T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                      T.RandomHorizontalFlip(),
                      T.ToTensor(),
                      T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

    else:
        baseline_transforms = T.Compose([
                      T.Resize(36),
                      T.CenterCrop(32),
                      T.ToTensor(),
                      T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

    return baseline_transforms

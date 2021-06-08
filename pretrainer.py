from config import load_args
from transforms import SimSiamAug, baseline
from models import Encoder, Predictor, SimSiam
from losses import SymmetricLoss, cosine_with_stopgrad
from schedulers import SimSiamScheduler
from utils import AverageMeter, load_from_saved, knn_monitor

import time
import os

import torch
from torch import nn, optim
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader


def unsupervised_train(train_set, device, args):

    if not os.path.isdir(args.checkpoint_dir):
        print('Directory %s not found. Creating local directory.' % args.checkpoint_dir) 
        os.mkdir(args.checkpoint_dir)
        
    train_loader = DataLoader(train_set, batch_size=args.pretrain_batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)

    # these loaders are just for the knn monitor
    memory_set = CIFAR10(root='./data', train=True, download=True, transform=baseline(train=True))
    test_set = CIFAR10(root='./data', train=False, download=True, transform=baseline(train=False))
    memory_loader = DataLoader(memory_set, batch_size=512, shuffle=False, drop_last=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, drop_last=True, pin_memory=True, num_workers=4)

    encoder = Encoder(hidden_dim=args.proj_hidden, output_dim=args.proj_out)
    predictor = Predictor(input_dim=args.proj_out, hidden_dim=args.pred_hidden, output_dim=args.pred_out)
    model = SimSiam(encoder, predictor)
        
    model.to(device)

    parameters = [{'name': 'encoder',
                   'params': [param for name, param in model.named_parameters() if name.startswith('encoder')],
                   'lr': args.pretrain_base_lr*args.pretrain_batch_size/256.},
                  {'name': 'predictor',
                   'params': [param for name, param in model.named_parameters() if name.startswith('predictor')],
                   'lr': args.pretrain_base_lr*args.pretrain_batch_size/256.}]

    optimizer = optim.SGD(parameters, 
                          lr=args.pretrain_base_lr*args.pretrain_batch_size/256., 
                          momentum=args.pretrain_momentum, 
                          weight_decay=args.pretrain_weight_decay)
    

    scheduler = SimSiamScheduler(optimizer, 
                                 warmup_epochs=args.pretrain_warmup_epochs, warmup_lr=args.pretrain_warmup_lr*args.pretrain_batch_size/256., 
                                 num_epochs=args.final_pretrain_epoch, base_lr=args.pretrain_base_lr*args.pretrain_batch_size/256., final_lr=0, iter_per_epoch=len(train_loader), 
                                 constant_predictor_lr=True)

    criterion = SymmetricLoss(cosine_with_stopgrad)
    
    loss_meter = AverageMeter('loss', time_unit='epoch', start_time=args.init_pretrain_epoch)
    knn_meter = AverageMeter('accuracy', time_unit='epoch', start_time=args.init_pretrain_epoch)

    if args.resume_from_checkpoint:
          model, optimizer, scheduler, loss_meter, knn_meter, args = load_from_saved(model, optimizer, scheduler, loss_meter, knn_meter, args)
    
    for epoch in range(args.init_pretrain_epoch, args.final_pretrain_epoch + 1, 1):

        print('===Beginning epoch %s===' % epoch)
        start = time.time()

        model.train()

        for batch_id, ((x1, x2), labels) in enumerate(train_loader):

            model.zero_grad()

            x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
            z1, z2, p1, p2 = model.forward(x1, x2)

            loss = criterion(z1, z2, p1, p2)
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item())
            
            if batch_id % 10 == 0:
                print('Batch %s of %s has loss=%.4f' % (batch_id, len(train_loader), loss.item()))

            scheduler.step()
        
        knn_acc = knn_monitor(model.encoder.backbone, device, memory_loader, test_loader)
        knn_meter.update(knn_acc)

        end = time.time()
        elapsed = (end - start)
        print('Epoch %s took %.2f seconds and has mean loss of %.4f and kNN accuracy %.2f %%.' % (epoch, elapsed, loss_meter.avg, knn_meter.avg * 100))
        
        loss_meter.reset()
        knn_meter.reset()

        if (epoch % 5 == 0) or (epoch == args.final_pretrain_epoch):
              torch.save({
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'scheduler_state_dict': scheduler.state_dict(),
              'current_epoch': epoch,
              'loss_meter': loss_meter,
              'knn_meter': knn_meter,
              'args': args,
              }, os.path.join(args.checkpoint_dir, args.checkpoint_name))
                
                
if __name__ == '__main__':

    args = load_args()

    train_set = CIFAR10(root='./data', train=True, download=True, transform=SimSiamAug())

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    unsupervised_train(train_set, device, args)

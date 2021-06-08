  
from config import load_args
from transforms import baseline
from models import Encoder, Predictor, SimSiam, LinearClassifier
from schedulers import SimpleCosineDecayLR
from utils import accuracy, AverageMeter

import time
import os

import torch
from torch import nn, optim
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader


def supervised_train(train_set, test_set, device, args):

    train_loader = DataLoader(train_set, batch_size=args.eval_batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=args.eval_batch_size, shuffle=False, num_workers=4)

    # load pre-trained model
    encoder = Encoder(hidden_dim=args.proj_hidden, output_dim=args.proj_out)
    predictor = Predictor(input_dim=args.proj_out, hidden_dim=args.pred_hidden, output_dim=args.pred_out)
    simsiam = SimSiam(encoder, predictor)
    state_dict = torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_name))
    simsiam.load_state_dict(state_dict['model_state_dict'])
    
    # remove everything after the backbone and freeze the representations
    model = simsiam.encoder.backbone
    for param in model.parameters():
        param.requires_grad = False
      
    # add a classifier, which we will train
    input_dim = model.output_dim
    model.classifier = LinearClassifier(input_dim)
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    model.to(device)
    
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=args.eval_base_lr*args.eval_batch_size/256., 
                          momentum=args.eval_momentum, 
                          weight_decay=args.eval_weight_decay)
    
    scheduler = SimpleCosineDecayLR(optimizer, start_epoch=args.init_eval_epoch, final_epoch=args.final_eval_epoch)

    criterion = nn.CrossEntropyLoss()

    loss_meter = AverageMeter('loss', time_unit='epoch', start_time=args.init_pretrain_epoch)
    train_acc_meter = AverageMeter('train accuracy', time_unit='epoch', start_time=args.init_pretrain_epoch)
    test_acc_meter = AverageMeter('test accuracy', time_unit='epoch', start_time=args.init_pretrain_epoch)

    for epoch in range(args.init_eval_epoch, args.final_eval_epoch + 1, 1):
        
        print('===Beginning epoch %s===' % epoch)
        start = time.time()

        for batch_id, (data, labels) in enumerate(train_loader):

            data, labels = data.to(device), labels.to(device)
            output = model(data)
            loss = criterion(output, labels) 
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            model.zero_grad()

            if batch_id % 10 == 0:
                print('Batch %s of %s has loss=%.4f' % (batch_id, len(train_loader), loss.item()))

        end = time.time()
        elapsed = (end - start)
        print('Epoch %s took %.2f seconds' % (epoch, elapsed))
          
        scheduler.step()

        train_acc = accuracy(model, train_loader, device)
        test_acc = accuracy(model, test_loader, device)        
        train_acc_meter.update(train_acc)
        test_acc_meter.update(test_acc)
        print('For epoch %s, accuracy on training is %.2f%%, accuracy on test is %.2f%%' % (epoch, train_acc * 100., test_acc * 100.))

        loss_meter.reset()
        train_acc_meter.reset()
        test_acc_meter.reset()

        if (epoch % 5 == 0) or (epoch == args.final_eval_epoch):
              torch.save({
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'current_epoch': epoch,
              'loss_meter': loss_meter,
              'train_acc_meter': train_acc_meter,
              'test_acc_meter': test_acc_meter,
              'args': args,
              }, os.path.join(args.checkpoint_dir, 'linear_' + args.checkpoint_name))
                
                
if __name__ == '__main__':

    args = load_args()

    train_set = CIFAR10(root='./data', train=True, download=True, transform=baseline(train=True))
    test_set  = CIFAR10(root='./data', train=False, download=True, transform=baseline(train=False))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    supervised_train(train_set, test_set, device, args)

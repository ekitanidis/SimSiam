import os
import torch
import torch.nn.functional as F


class AverageMeter():

    def __init__(self, param_name, time_unit='epoch', start_time=1):
        self.param_name = param_name
        self.time_unit = time_unit
        self.log = dict({time_unit:[], param_name:[]})
        self.time = start_time
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.log[self.time_unit].append(self.time)
        self.log[self.param_name].append(self.avg)
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.time += 1

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count        
                
        
def accuracy(model, dataloader, device):

    model.eval()
    num_correct = 0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            output = model(data).argmax(1)
            num_correct += (output == labels).sum().item()
    accuracy = num_correct / len(dataloader.dataset)

    return accuracy


def load_from_saved(model, optimizer, scheduler, loss_meter, knn_meter, args):

    saved = torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_name))

    model.load_state_dict(saved['model_state_dict'])
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    scheduler.load_state_dict(saved['scheduler_state_dict'])
    loss_meter = saved['loss_meter']
    knn_meter = saved['knn_meter']
    completed_epochs = saved['current_epoch']
    original_args = saved['args']

    print('Overriding hyper parameter arguments with values used to generate saved model, for consistency.')
    for key, value in vars(original_args).items():
        if key not in ['checkpoint_dir', 'checkpoint_name']:
            setattr(args, key, value)

    args.init_pretrain_epoch = 1 + completed_epochs
    assert (args.final_pretrain_epoch - args.init_pretrain_epoch) >= 0, "This training run is already complete."

    return model, optimizer, scheduler, loss_meter, knn_meter, args


# knn code modified from https://github.com/leftthomas/SimCLR/blob/master/main.py#L48
def knn_monitor(model, device, train_loader, test_loader, num_classes=10, k=200, softmax_temp=0.5):
    
    model.to(device)
    model.eval()

    with torch.no_grad():

        feat_bank = []
        for batch_id, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            feat = model(data).squeeze()
            feat = F.normalize(feat, dim=1)
            feat_bank.append(feat)
        feat_bank = torch.cat(feat_bank, dim=0).t().contiguous()
        feat_labels = torch.tensor(train_loader.dataset.targets, device=feat_bank.device)

        num_correct, total_num = 0, 0
        for batch_id, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)
            total_num += data.size(0)
            feat = model(data).squeeze()
            feat = F.normalize(feat, dim=1)
            sim_matrix = torch.mm(feat, feat_bank)
            sim_weights, sim_indices = sim_matrix.topk(k=k, dim=-1)
            sim_weights = (sim_weights / softmax_temp).exp()
            sim_labels = torch.gather(feat_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            one_hot_label = torch.zeros(data.size(0) * k, num_classes, device=sim_labels.device)
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            pred_scores = torch.sum(one_hot_label.view(data.size(0), k, num_classes) * sim_weights.unsqueeze(dim=-1), dim=1)
            pred_labels = pred_scores.argmax(dim=-1)
            num_correct += (pred_labels == labels).float().sum().item()
            
    return num_correct / total_num

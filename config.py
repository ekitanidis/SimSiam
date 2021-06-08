import argparse


def load_args():

    parser = argparse.ArgumentParser()

    # Contrastive pre-training
    
    parser.add_argument('--pretrain_batch_size', type=int, default=512)
    parser.add_argument('--pretrain_warmup_epochs', type=int, default=0)
    parser.add_argument('--pretrain_warmup_lr', type=float, default=0)
    parser.add_argument('--pretrain_base_lr', type=float, default=0.03)
    parser.add_argument('--pretrain_momentum', type=float, default=0.9)
    parser.add_argument('--pretrain_weight_decay', type=float, default=5e-4)
    parser.add_argument('--init_pretrain_epoch', type=int, default=1)
    parser.add_argument('--final_pretrain_epoch', type=int, default=800)
    
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default='simsiam_pretrained.pt')
    parser.add_argument('--resume_from_checkpoint', action='store_true')

    parser.add_argument('--proj_hidden', type=int, default=2048)
    parser.add_argument('--proj_out', type=int, default=2048)
    parser.add_argument('--pred_hidden', type=int, default=512)
    parser.add_argument('--pred_out', type=int, default=2048)

    # Downstream evaluation with linear classifier

    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--eval_base_lr', type=float, default=30.0)
    parser.add_argument('--eval_momentum', type=float, default=0.9)
    parser.add_argument('--eval_weight_decay', type=float, default=0)
    parser.add_argument('--init_eval_epoch', type=int, default=1)
    parser.add_argument('--final_eval_epoch', type=int, default=800)

    args = parser.parse_args()

    return args

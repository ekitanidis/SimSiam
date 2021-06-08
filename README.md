# SimSiam

**PyTorch implementation of SimSiam (https://arxiv.org/abs/2011.10566), verified on CIFAR-10.**

To pre-train the image representation model:

    python pretrainer.py --checkpoint_name='my_pretrained_model.pt'

You can also resume the pre-training from a checkpoint (if the original training run is incomplete, it will finish the remaining epochs):

    python pretrainer.py --resume_from_checkpoint --checkpoint_name='my_pretrained_model.pt'

To run the classifier on your pre-trained model:

    python classifier.py --checkpoint_name='my_pretrained_model.pt'
    
**Performance on CIFAR-10:**

Pretraining performance:

<img src="./figures/pretrain_results.jpg?raw=true)" alt="pretraining performance" width="700"/>

Final test accuracy of linear classifier: 

    91.6%

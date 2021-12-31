# [AAAI22] Cross-Domain Empirical Risk Minimization for Unbiased Long-tailed Classification

We point out the overlooked unbiasedness in long-tailed classification: models should perform well on both imbalanced and balanced test distributions.
To tackle this challenge, we propose a novel training paradigm called Cross-Domain Empirical Risk Minimization (xERM).
xERM consists of two ERM terms, which are the cross-entropy losses with different supervisions: 
one is the ground-truth label from the seen imbalanced domain, and the other is the prediction of a balanced model. 
The imbalanced domain empirical risk encourages the model to learn from the ground-truth annotations from the imbalanced distribution, 
which favors the head classes. The balanced domain empirical risk encourages the model to learn from the prediction of a balanced model, 
which imagines a balanced distribution and prefers the tail classes. These two risks are weighted to take advantage of both, 
i.e., protect the model training from being neither too “head biased” nor too “tail biased”, and can achieve the unbiasedness.

The codes are organized into three folders:

1. [xERM.PC.public](xERM.PC.public) uses Post-Compensated Softmax ([PC](https://arxiv.org/abs/2012.00321)) as balanced model to implement xERM.
2. [xERM.TDE.public](xERM.TDE.public) uses Total Direct Effect ([TDE](https://arxiv.org/abs/2009.12991)) as balanced model to implement xERM.
3. xERM.RIDE.public uses [RIDE](https://arxiv.org/abs/2010.01809) as balanced model to implement xERM.

## Citation

If you find our paper or this project helps your research, please kindly consider citing our paper in your publication.

```
@inproceedings{beierxERM,
  title={Cross-Domain Empirical Risk Minimization for Unbiased Long-tailed Classification},
  author={Zhu, Beier and Niu, Yulei and Hua, Xian-Sheng and Zhang, Hanwang},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2022}
}
```

## Update Logs
- 28 Dec 2021, Release train and eval code of xERM-PC on CIFAR100-LT and ImageNet-LT Dataset. 
- 30 Dec 2021, Release train and eval code of xERM-TDE on Places365-LT Dataset (ResNet50 Backbone). 
- 31 Dec 2021, Release train and eval code of xERM-TDE on Places365-LT Dataset (ResNet152 Backbone). 

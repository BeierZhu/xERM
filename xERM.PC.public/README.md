#Cross-Domain Empirical Risk Minimization for Unbiased Long-tailed Classification

- This codebase is built on [PC](https://github.com/hyperconnect/LADE).

We currently provide the training code of xERM-PC for CIFAR100-LT and ImageNet-LT. 

## Install

To obtain the same results, please make sure to set up the same environment. 
```
 conda create  --name xERM --file spec-list.txt
```


## Preliminaries
- Prepare dataset: CIFAR-100, ImageNet-LT
  - Please download those datasets following [Decoupling](https://github.com/facebookresearch/classifier-balancing#dataset).

## CIFAR100-LT

### Imbalance Ratio: 100
Step 1: Train PC model

```
python main.py --seed 1 --cfg config/CIFAR100_LT/ce100.yaml --gpu 0
```

Step 2: Train xERM-PC model:

```
python main.py --seed 1 --cfg config/CIFAR100_LT/ce100.yaml --gpu 0 --xERM
```
Step 3: Evaluate:

```
python main.py --seed 1 --cfg config/CIFAR100_LT/ce100.yaml --gpu 0 --xERM --test
```

### Imbalance Ratio: 50
Step 1: Train PC model:
```
python main.py --seed 1 --gpu 0 --cfg config/CIFAR100_LT/ce50.yaml
```

Step 2: Train xERM-PC model:
```
python main.py --seed 1 --gpu 0 --cfg config/CIFAR100_LT/ce50.yaml --xERM
```
Step 3: Evaluate:
```
python main.py --seed 1 --gpu 0 --cfg config/CIFAR100_LT/ce50.yaml --xERM --test
```

### Imbalace Ratio: 10
Step 1: Train PC model

```
python main.py --seed 1 --gpu 0 --cfg config/CIFAR100_LT/ce10.yaml
```

Step 2: Train xERM-PC model:

```
python main.py --seed 1 --gpu 0 --cfg config/CIFAR100_LT/ce10.yaml --xERM
```
Step 3: Evaluate:

```
python main.py --seed 1 --gpu 0 --cfg config/CIFAR100_LT/ce10.yaml --xERM --test
```

## ImageNet-LT

Step 1: Train PC model
```
python main.py --cfg  config/ImageNet_LT/ce.yaml --seed 1  --gpu 0,1,2,3
``` 

Step 2: Train xERM-PC model:

```
python main.py --cfg  config/ImageNet_LT/ce.yaml --seed 1  --gpu 0,1,2,3 --xERM
```
Step 3: Evaluate
```
python main.py --cfg  config/ImageNet_LT/ce.yaml --seed 1  --gpu 0,1,2,3 --xERM --test
```

*We modify the original config of PC. To run the original config of PC, please change the config/ImageNet_LT/ce.yaml to config/ImageNet_LT/ce_pc.yaml

## Citation
if you find our codes helpful, please cite our paper:

```
@inproceedings{beierxERM,
  title={Cross-Domain Empirical Risk Minimization for Unbiased Long-tailed Classification},
  author={Zhu, Beier and Niu, Yulei and Hua, Xian-Sheng and Zhang, Hanwang},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2022}
}
```

## License
The use of this software is released under BSD-3.


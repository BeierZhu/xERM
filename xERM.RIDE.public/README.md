# Cross-Domain Empirical Risk Minimization for Unbiased Long-tailed Classification

This project is built on top of [RIDE](https://arxiv.org/abs/2010.01809).

## Install 

Please refer to [RIDE](https://github.com/frank-xwang/RIDE-LongTailRecognition) to to set environment and dataset.


## CIFAR100-LT-IB-100

###  2 experts

#### Step 1: 
Train RIDE-PC-2 model:
```
python train.py -c configs/cifar100_ERM_2experts.json
```
Eval RIDE-PC-2 model:
```
python test.py -c configs/cifar100_ERM_2experts.json -r model_path
```

#### Step 2: 
Train xERM-RIDE-PC-2 model, please change the ```balanced_model_path``` to the  PC model path
```
python train.py -c "configs/config_imbalance_cifar100_ride_xERM_2experts.json"
```
Eval
```
python test.py  -c  configs/cifar100_xERM_2experts.json -r model_path
```

###  3 experts

#### Step 1: 
Train RIDE-PC-3 model:
```
python train.py -c configs/cifar100_ERM_3experts.json
```
Eval RIDE-PC-3 model:
```
python test.py -c configs/cifar100_ERM_3experts.json -r model_path
```

#### Step 2: 
Train xERM-RIDE-PC-3 model, please change the ```balanced_model_path``` to the  PC model path
```
python train.py -c "configs/config_imbalance_cifar100_ride_xERM_3experts.json"
```
Eval
```
python test.py  -c  configs/cifar100_xERM_3experts.json -r model_path
```

### Result

| Model          | Overall | Recall Many | Recall Med | Recall Few | Precision Many | Precision Med | Precision Few |
|----------------|---------|-------------|------------|------------|----------------|---------------|---------------|
| RIDE-PC-2      |  46.3    | 62.2   | 45.4  | 28.7    |     59.9   |  49.8      | 27.5              |
| xERM-RIDE-PC-2 |  47.8   |  65.5  |  50.5   |   24.0   |  53.9    |  53.5    |   45.0            |
| RIDE-PC-3      |  47.8   | 64.8    | 46.2    |  29.6   | 61.6     |  51.0      | 28.8         |
| xERM-RIDE-PC-3 | 50.5  | 66.7   | 52.2   | 29.6   |    56.5   |    53.0    |  35.6         |
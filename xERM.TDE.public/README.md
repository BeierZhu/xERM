# Cross-Domain Empirical Risk Minimization for Unbiased Long-tailed Classification

This project is built on top of [TDE](https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch)


## Install

Please refer to [TDE](https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch) to set environment and dataset.


## Places365-LT
Please change the ```resnet50 = load_model(resnet50, pretrain='/data1/pretrained/resnet50-0676ba61.pth')``` in ```models/ResNet50Feature.py``` to your resnet50 pretrained model path.

Please change the ResNet152 pretrained model path in ```models/ResNet152Feature.py```

## ResNet50

Step 1: Train TDE model

```
python main.py --cfg ./config/Places_LT/resnet50_TDE.yaml --gpu 0,1,2,3
``` 

Step 2: Train xERM-TDE model

```
python main.py --cfg ./config/Places_LT/resnet50_xERM.yaml --gpu 0,1,2,3 --xERM
```

Step 3: Evaluate
```
python main.py --cfg ./config/Places_LT/resnet50_xERM.yaml --gpu 0,1,2,3 --xERM --test
```

## ResNet152

Step 1: Train TDE model

```
python main.py --cfg ./config/Places_LT/resnet152_TDE.yaml --gpu 0,1,2,3
``` 

Step 2: Train xERM-TDE model

```
python main.py --cfg ./config/Places_LT/resnet152_xERM.yaml --gpu 0,1,2,3 --xERM
```

Step 3: Evaluate
```
python main.py --cfg ./config/Places_LT/resnet152_xERM.yaml --gpu 0,1,2,3 --xERM --test
```

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

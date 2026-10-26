# MSAFed

本仓库是对官方代码库 [Jchanel-1/MSAFed](https://github.com/Jchanel-1/MSAFed) 的**代码补充**，补全原仓库中缺失的模块：

- `model_zoo/`：模型定义
- `federated/Queue.py`：队列相关实现

其余训练流程、数据说明与官方仓库保持一致。使用时可对照官方仓库，将上述缺失文件合并进本地工程。

## Datasets
Fed-polyp: You can download Fed-polyp from [fed-polyp](https://drive.google.com/file/d/1w4kuvHaWChP9-OicA9Sxpd8t6v9bGW4G/view?usp=drive_link).

## Pretrain
Train a generalized global model across clients. The learned model will serve as the initialization for subsequent stages.

```
bash pretrain_inside.sh
```

## Inside Adaptation
Fine-tune the pretrained global model on each client using adaptive layer-wise learning rates. This stage outputs the adapted model parameters, multi-client prototypes and the adaptive learning rates.

```
bash ada_inside.sh
```

## Test Time Adaptation
Adapt the global model to outside clients using the prototypes and learning rates obtained from previous stages during test time.

```
bash tta_outside.sh
```

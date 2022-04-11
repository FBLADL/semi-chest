# S2MTS2
This repo contains the Pytorch implementation of our paper: 
> [**Self-supervised Mean Teacher for Semi-supervised Chest X-ray Classification**](https://arxiv.org/abs/2103.03629.pdf)
>
> Fengbei Liu*, [Yu Tian*](https://yutianyt.com/), Filipe R. Cordeiro, [Vasileios Belagiannis](https://www.uni-ulm.de/in/mrm/institut/mitarbeiter/gruppenleiter/vb/), [Ian Reid](https://cs.adelaide.edu.au/~ianr/), [Gustavo Carneiro](https://cs.adelaide.edu.au/~carneiro/).

- **Accepted at MICCAI MLMI2021 Workshop.**  

## Requirements
- Linux
- Python 3.8
- Pytorch 1.6
- Pretrain with 4 * V100 and finetune with 1 * V100

## Prepare Dataset
Download Chest Xray14 from https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345

## Pre-training and Fine-tuning

Prepare the dataset and then run the following command for pretrain: 
```shell
python pretrain.py --data <data_dir> --multiprocessing-distributed --world-size 1 --rank 0 --batch-size 256 --lr 0.03 --arch densenet121 --mlp --cos --task chestxray14 --dist-url tcp://localhost:10001 --jcl
```
For Fine-tuning, run
```shell
python finetune.py --pretrained <pretrain model dir>
```


## Citation

If you find this repo useful for your research, please consider citing our paper:

```bibtex
@article{liu2021self,
  title={Self-supervised Mean Teacher for Semi-supervised Chest X-ray Classification},
  author={Liu, Fengbei and Tian, Yu and Cordeiro, Filipe R and Belagiannis, Vasileios and Reid, Ian and Carneiro, Gustavo},
  journal={arXiv preprint arXiv:2103.03629},
  year={2021}
}
```
---

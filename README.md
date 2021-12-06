# S2MTS2
This repo contains the Pytorch implementation of our paper: 
> [**Self-supervised Mean Teacher for Semi-supervised Chest X-ray Classification**](https://arxiv.org/pdf/2103.03423.pdf)
>
> Fengbei Liu*, [Yu Tian*](https://yutianyt.com/), Filipe R. Cordeiro, [Vasileios Belagiannis](https://www.uni-ulm.de/in/mrm/institut/mitarbeiter/gruppenleiter/vb/), [Ian Reid](https://cs.adelaide.edu.au/~ianr/), [Gustavo Carneiro](https://cs.adelaide.edu.au/~carneiro/).

- **Accepted at MICCAI MLMI2021 Workshop.**  


## Training

Prepare the dataset and then simply run the following command for pretrain: 
```shell
python pretrain.py



```
For Fine-tuning, run
```shell
python finetune.py
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

# Data-Free Model Extraction

This repository complements the [Data-Free Model Extraction paper](https://arxiv.org/abs/2011.14779), that will be published at the 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition.

This project was conducted in collaboration between the [Cake Lab](https://cake.wpi.edu/) at Worcester Polytechnic Institute, and the [University of Toronto](https://www.utoronto.ca/) and the [Vector Institute](https://vectorinstitute.ai/).


## Updates
- Jan 14, 2022: Updated the default learning rate parameter for the generator to match the paper's experimental setup. 

## Citation
```
@InProceedings{Truong_2021_CVPR,
author = {Truong, Jean-Baptiste and Maini, Pratyush and Walls, Robert J. and Papernot, Nicolas},
title = {Data-Free Model Extraction},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2021}
}
```

## Dependencies
The code requires dependencies that can be installed using the `pip` environment file provided:
```
pip install -r requirements.txt
```

## Replicating DFME Results

### Load Victim Model Weights
First, download the pretrained victim model weights from [this dropbox](https://www.dropbox.com/sh/lt6w0nq3msp4do0/AADmJk2k3LQqFqWt9916W-nra?dl=0). The two file names are `cifar10-resnet34_8x.pt` and `svhn-resnet34_8x.pt`. The CIFAR10 weights were found on the [Data Free Adversarial Distillation](https://github.com/VainF/Data-Free-Adversarial-Distillation) dropbox, while we trained the SVHN model ourselves.

Then, store the pre-trained model weights at the following location

`dfme/checkpoint/teacher/{victim_dataset}-resnet34_8x.pt`


### Perform Model Extraction
```
bash run_cifar_dfme.sh
bash run_svhn_dfme.sh
```
Logs and saved models can be found at `save_results/{victim_dataset}/`  


## Surrogate Benchmarking
Standard model extraction attacks can be performed using the code in the folder `surrogate_benchmark`.

```
cd surrogate_benchmark
python train.py --surrogate {surrogate_dataset} --target {target_dataset} --temp {temperature_value} --lr_mode 1 --epochs 50
```
Typically, using `temperature_value` in {1,3,5} provides good extraction results. The number of epochs may be reduced to 30 in case the `target` dataset is `svhn`.



## Attribution

This repository was built on code from the paper [Data Free Adversarial Distillation](https://github.com/VainF/Data-Free-Adversarial-Distillation). The weights and model architectures for Resnet34-8x and Resnet18_8x were also found on the repository released with the Data Free Adversarial Distillation paper.

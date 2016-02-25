## Install

### Download the weights and bias

```
wget -P networks/ https://cloud.lip6.fr/index.php/s/7ZLIcvucyg4tyLN/download # overfeat
wget -P networks/ https://cloud.lip6.fr/index.php/s/xd1Ek8MjEz813ri/download # vgg16
wget -P networks/ https://cloud.lip6.fr/index.php/s/IpLqxLrKjeRMcyY/download # vgg19
```

## Setting the dataset(s)

- overfeat takes images formated as 3x221x221
- vgg16&vgg19 takes images formated as 3x224x224 

### Create data augmented *10 images

```
th main_augmentation.lua -h                   # display options
th main_augmentation.lua                      # run data augmentation
```

### Process the mean and std images

```
th main_meanstd.lua -h
th main_meanstd.lua                          # create mean&std.jpg in dataset dir
```

## Train a network

### Lunch overfeat for a given augmented dataset 

```
nvidia-smi				                           # display your GPU devices id
CUDA_VISIBLE_DEVICES=0 th main.lua -h        # display options
CUDA_VISIBLE_DEVICES=0 th main.lua           # create dataLoader in cache_221 and run
```

## Lunch an safe experience

When you have to wait for several hours or days, you want a safe way to lunch an experience.
With `nohup` the process will still run even if your ssh connexion is closed.

```
echo "CUDA_VISIBLE_DEVICES=0 th main.lua -netType vgg16 -pretrain yes -imageSize 224 -batchSize 26 -lr 0.1" > GPU0.sh
chmod 755 GPU0.sh                              # in order to be executed by the shell
nohup ./GPU0.sh > GPU0.log &                   # lunch without a terminal
tail -100 GPU0.log                             # keep track of the training process
```

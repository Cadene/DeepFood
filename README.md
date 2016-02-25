# Deep Food : An easy and fast way to fine tune deep networks with Torch7

## Download the weights and bias

```
wget -P networks/ https://cloud.lip6.fr/index.php/s/7ZLIcvucyg4tyLN/download # overfeat
wget -P networks/ https://cloud.lip6.fr/index.php/s/xd1Ek8MjEz813ri/download # vgg16
wget -P networks/ https://cloud.lip6.fr/index.php/s/IpLqxLrKjeRMcyY/download # vgg19
```

## Preprocess the dataset(s)

/!\ Don't forget, overfeat takes as input 3x221x221 RGB formated images. Also, vgg16 & vgg19 take 3x224x224.

This following code create data augmented images (*10) that way :

- resize the image to 256x256^ (^ means that you don't want to deform the proportions)
- do 5 crops by 221x221 (center, northweast, northeast, southwest, southeast)
- flop the crops and get 5 more images (flop means horizontal flip)

```
th main_augmentation.lua -h                   # display options
th main_augmentation.lua                      # run data augmentation
```

This code process the mean and std based on a trainset subset.
```
th main_meanstd.lua -h
th main_meanstd.lua                          # create mean&std.jpg in dataset dir
```

## Train the network

This code train and evaluate overfeat for a given augmented dataset.
```
nvidia-smi                                   # display your GPU devices id
CUDA_VISIBLE_DEVICES=0 th main.lua -h        # display options
CUDA_VISIBLE_DEVICES=0 th main.lua           # create GPU0 dir to log the exp
```

## Optional : Lunch a safe experience

When you have to wait for several hours or days, you want a safe way to lunch an experience.
With `nohup` the process will still run even if your ssh connexion is closed.

```
echo "CUDA_VISIBLE_DEVICES=0 th main.lua -netType vgg16 -pretrain yes -imageSize 224 -batchSize 26 -lr 0.1" > GPU0.sh
chmod 755 GPU0.sh                            # in order to be executed by the shell
nohup ./GPU0.sh > GPU0.log &                 # lunch without a terminal
tail -100 GPU0.log                           # keep track of the training process
```

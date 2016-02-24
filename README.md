## Download the weights and bias

```
wget -P networks/ http://webia.lip6.fr/~cadene/overfeat.t7
wget -P networks/ http://webia.lip6.fr/~cadene/vgg16.t7
wget -P networks/ http://webia.lip6.fr/~cadene/vgg19.t7
```

## Create the augmented dataset (I will simplify this step)

- overfeat takes images formated as 3x221x221
- vgg16&vgg19 takes images formated as 3x224x224 

```
./createdataset.sh                             # you have to edit those files
th compte_mean_std.lua                         # create mean&std.jpg in dataset dir
```

## Running overfeat for a given augmented dataset 

```
nvidia-smi				       # display your GPU devices id
CUDA_VISIBLE_DEVICES=0 th main.lua -h          # display options
CUDA_VISIBLE_DEVICES=0 th main.lua             # create dataLoader in cache_221 and run
```

## Lunching an experience

```
echo "CUDA_VISIBLE_DEVICES=0 th main.lua -netType vgg16 -pretrain yes -imageSize 224 -batchSize 26 -lr 0.1" > GPU0.sh
chmod 755 GPU0.sh                              # in order to be executed
nohup ./GPU0.sh > GPU0.log &                   # keep running with a closed terminal
tail -100 GPU0.log                             # keep track of the training process
```

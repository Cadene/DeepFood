--
--  Copyright (c) 2016, LIP6, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Author: remi.cadene@lip6.fr

require 'torch'
require 'nn'
require 'image'
require 'optim'
local threads = require 'threads'

local sig     = require 'posix.signal'
local unistd  = require 'posix.unistd'

--------------------------------------
-- Timers

local tm = torch.Timer()
local tmDataload = torch.Timer()
local tmEpoch = torch.Timer()
local tmBatch = torch.Timer()

-------------------------------------
-- Options

cmd = torch.CmdLine()
-- Training options
-- cmd:option('-pretrain', 'yes', 'Options: yes (finetuning) | no (from scratch)')
cmd:option('-threads', 1, 'Threads number (minimum 2)')
cmd:option('-imageSize', 221, 'w and h of an image to load')
cmd:option('-batchSize', 60, 'Size of a batch (60 for overfeat, 20 for vgg19)')
cmd:option('-netType', 'overfeat', 'Options: overfeat | vgg16')
-- Optimization options
cmd:option('-lr',  8e-1, 'Learning Rate')
cmd:option('-lrd', 3e-4, 'Learning Rate Decay')
cmd:option('-wd',  1e-3, 'Weight Decay')
cmd:option('-m',   0.6, 'Momentum')
cmd:option('-lrf_conv', 10, 'lr will be divided by lrf_conv only for convolution layers')
opt = cmd:parse(arg or {})

opt.idGPU = os.getenv('CUDA_VISIBLE_DEVICES')
opt.cuda = true
opt.nb_epoch = 60
opt.seed = 1337
opt.path2data = '/home/cadene/data/UPMC_Food101_'..opt.imageSize..'_augmented'
opt.save_model = false
opt.path2cache = '/home/cadene/doc/DeepFood/cache_'..opt.imageSize
opt.path2save = '/home/cadene/doc/DeepFood/GPU'..opt.idGPU
opt.path2networks = '/home/cadene/doc/DeepFood/networks'
opt.path2model = opt.path2networks..'/'..opt.netType..'.lua'

-------------------------------------
-- Info

print("Lunching using pid = "..unistd.getpid().." on CPU")
print("Lunching using GPU = "..opt.idGPU)
print("Options : ", opt)

print('Caching everything to: ' .. opt.path2cache)
os.execute('mkdir -p ' .. opt.path2cache)

print('Saving everything to: ' .. opt.path2save)
os.execute('mkdir -p ' .. opt.path2save)

os.execute('echo "'..unistd.getpid()..'" > '..opt.path2save..'/pid.log')
os.execute('echo "'..os.date():gsub(' ','')..'" > '..opt.path2save..'/lunchdate.log')

torch.manualSeed(opt.seed)
-- torch.setnumthreads(4) -- doesn't seem to affect anything...
torch.setdefaulttensortype('torch.FloatTensor')
if opt.cuda then
    print('Loading CUDA and CUDNN')
    require 'cutorch'
    cutorch.setDevice(1) -- use CUDA_VISIBLE_DEVICES=i th main.lua i=[0,7]
    cutorch.manualSeed(opt.seed)
    require 'cunn'
    require 'cudnn'
end

-------------------------------------
-- Lunching Threads and recovering Datasets

threads.serialization('threads.sharedserialize')
do 
    if opt.threads > 0 then
        local options = opt
        pool = threads.Threads(
            opt.threads,
            function(thread_id)
                print('Starting a new thread # ' .. thread_id)
                require 'torch'
            end,
            function(thread_id)
                opt = options
                paths.dofile('init_thread.lua')
                torch.manualSeed(opt.seed)
            end
        )
    else
        paths.dofile('init_thread.lua')
        pool = {}
        function pool:addjob(f1, f2) f2(f1()) end
        function pool:synchronize() end
    end
end

classes = nil
nClasses = 0
nTrain = 0
nTest = 0
pool:addjob(function() return trainLoader.classes end, function(c) classes = c end)
pool:addjob(function() return trainLoader:size() end, function(c) nTrain = c end)
pool:addjob(function() return testLoader:size() end, function(c) nTest = c end)
pool:synchronize()
nClasses = #classes
print('nClasses: ', nClasses)
print('nTrain: ', nTrain)
print('nTest: ', nTest)
opt.epochSize = nTrain / opt.batchSize

-------------------------------------
-- Model

print('Loading model : '..opt.netType)
paths.dofile(opt.path2model)
if opt.cuda then model:cuda() end

assert(opt.imageSize == model.imageSize)
assert(opt.netType == model.name)
assert(model.params_conv > 0)

print('Input = '..opt.batchSize..'x3x'..model.imageSize..'x'..model.imageSize)
print(model)

local inputs = torch.CudaTensor()
local targets = torch.CudaTensor()

print('Reshaping parameters and gradParameters')
local parameters, gradParameters = model:getParameters()

local criterion = nn.ClassNLLCriterion()
if opt.cuda then criterion:cuda() end

-- local confusion   = optim.ConfusionMatrix(nClasses)
local trainLogger = optim.Logger(paths.concat(opt.path2save, 'train.log'))
local testLogger  = optim.Logger(paths.concat(opt.path2save, 'test.log'))

-------------------------------------
-- Optimizer SGD

local config = {
    learningRate = opt.lr,
    weightDecay = opt.wd,
    momentum = opt.m,
    learningRateDecay = opt.lrd,
    learningRateConvFactor = opt.lrf_conv,
    paramsBeforeFullyConnected = model.params_conv
}
optim.sgd = require 'sgd'

-------------------------------------
-- Training 

function train()
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch_id)

    cutorch.synchronize()
    model:training()

    tmEpoch:reset()

    top1_epoch = 0
    top5_epoch = 0
    loss_epoch = 0

    local shuffle = torch.randperm(nTrain) -- if global not upvalue

    batch_id = 1
    for i = 1, opt.epochSize do
        local indexStart = (i-1) * opt.batchSize + 1
        local indexEnd = (indexStart + opt.batchSize - 1)
        pool:addjob(
            function()
                local inputs, targets = trainLoader:get(indexStart, indexEnd, shuffle)
                return inputs, targets
            end,
            trainBatch
        )
    end

    pool:synchronize()
    cutorch.synchronize()

    top1_epoch = top1_epoch * 100 / (opt.batchSize * opt.epochSize)
    top5_epoch = top5_epoch * 100 / (opt.batchSize * opt.epochSize)
    loss_epoch = loss_epoch / opt.epochSize

    trainLogger:add{
        ['top1 accuracy (%)'] = top1_epoch / opt.epochSize,
        ['top5 accuracy (%)'] = top5_epoch / opt.epochSize,
        ['avg loss'] = loss_epoch / opt.epochSize
    }
    print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 %.2f\t'
                          .. 'accuracy(%%):\t top-5 %.2f\t',
                       epoch_id, tmEpoch:time().real, loss_epoch, top1_epoch, top5_epoch))
    print()
    -- if opt.save_model then
    --     print('# ... saving model')
    --     torch.save(paths.concat(opt.path2save, 'model'..epoch_id..'.t7'), model)
    -- end
end

function trainBatch(inputsCPU, targetsCPU, threadid)
    --print('Training with CPU', 'Receving data from threadid='..threadid)
    cutorch.synchronize()
    collectgarbage()
    local dataLoadingTime = tmDataload:time().real 
    tmBatch:reset()

    inputs:resize(inputsCPU:size()):copy(inputsCPU) --transfer over to GPU
    targets:resize(targetsCPU:size()):copy(targetsCPU)

    local loss, outputs
    feval = function (x)
        model:zeroGradParameters()
        outputs = model:forward(inputs)
        loss    = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model:backward(inputs, df_do)
        return loss, gradParameters
    end
    optim.sgd(feval, parameters, config)

    cutorch.synchronize()
    loss_epoch = loss_epoch + loss
    local top1 = 0
    local top5 = 0
    do
      local _,prediction_sorted = outputs:float():sort(2, true) -- descending
      for i=1, opt.batchSize do
        if prediction_sorted[i][1] == targetsCPU[i] then
          top1_epoch = top1_epoch + 1;
          top1 = top1 + 1
        end
        for j=1,5 do
          if prediction_sorted[i][j] == targetsCPU[i] then
            top5_epoch = top5_epoch + 1
            top5 = top5 + 1
            break
          end
        end
      end
      top1 = top1 * 100 / opt.batchSize
      top5 = top5 * 100 / opt.batchSize
    end

    local current_lr = config.learningRate / (1 + config.evalCounter*config.learningRateDecay)

    print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Top1-%%: %.2f Top5-%%: %.2f LR %.2e DataLoadingTime %.3f'):format(
      epoch_id, batch_id, opt.epochSize, tmBatch:time().real,
      loss, top1, top5, current_lr, dataLoadingTime))

    batch_id = batch_id + 1
    tmDataload:reset()
end

-------------------------------------
-- Testing

function test()
    print('==> doing epoch on testing data:')
    print("==> online epoch # " .. epoch_id)

    cutorch.synchronize()
    model:evaluate()

    tmEpoch:reset()

    top1_epoch = 0
    top5_epoch = 0
    loss_epoch = 0

    batch_id = 1
    for i = 1, nTest/opt.batchSize do
        local indexStart = (i-1) * opt.batchSize + 1
        local indexEnd = (indexStart + opt.batchSize - 1)
        pool:addjob(
            function()
                local inputs, targets = testLoader:get(indexStart, indexEnd)
                return inputs, targets
            end,
            testBatch
        )
    end

    pool:synchronize()
    cutorch.synchronize()

    top1_epoch = top1_epoch * 100 / (opt.batchSize * opt.epochSize)
    top5_epoch = top5_epoch * 100 / (opt.batchSize * opt.epochSize)
    loss_epoch = loss_epoch / (nTest/opt.batchSize)

    testLogger:add{
        ['top1 accuracy (%)'] = top1_epoch / opt.epochSize,
        ['top5 accuracy (%)'] = top5_epoch / opt.epochSize,
        ['avg loss'] = loss_epoch / opt.epochSize
    }
    print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 %.2f\t'
                          .. 'accuracy(%%):\t top-5 %.2f\t',
                       epoch_id, tmEpoch:time().real, loss_epoch, top1_epoch, top5_epoch))
    print()
end

function testBatch(inputsCPU, targetsCPU, threadid)
    --print('Training with CPU', 'Receving data from threadid='..threadid)
    cutorch.synchronize()
    collectgarbage()
    local dataLoadingTime = tmDataload:time().real 
    tmBatch:reset()

    inputs:resize(inputsCPU:size()):copy(inputsCPU) --transfer over to GPU
    targets:resize(targetsCPU:size()):copy(targetsCPU)

    local outputs = model:forward(inputs)
    local loss = criterion:forward(outputs, targets)
    local pred = outputs:float()

    loss_epoch = loss_epoch + loss
    local top1 = 0
    local top5 = 0
    do
      local _,prediction_sorted = outputs:float():sort(2, true) -- descending
      for i=1, opt.batchSize do
        if prediction_sorted[i][1] == targetsCPU[i] then
          top1_epoch = top1_epoch + 1;
          top1 = top1 + 1
        end
        for j=1,5 do
          if prediction_sorted[i][j] == targetsCPU[i] then
            top5_epoch = top5_epoch + 1
            top5 = top5 + 1
            break
          end
        end
      end
      top1 = top1 * 100 / opt.batchSize
      top5 = top5 * 100 / opt.batchSize
    end

    print(('Epoch: Testing [%d][%d/%d]\tTime %.3f Err %.4f Top1-%%: %.2f Top5-%%: %.2f DataLoadingTime %.3f'):format(
      epoch_id, batch_id, nTest/opt.batchSize, tmBatch:time().real,
      loss, top1, top5, dataLoadingTime))

    batch_id = batch_id + 1
    tmDataload:reset()
end


-------------------------------------
-- Lunching training and testing

-- you can interrupt the training process sending a SIGUSR1 signal
sig.signal(sig.SIGUSR1, function()
  print('Interrupting training at the end of this epoch.')
  interrupt = true
end)

epoch_id = 1
while opt.nb_epoch >= epoch_id do
    train()
    test()
    if interrupt then
        break
    end
    epoch_id = epoch_id + 1
end

-------------------------------------
-- Saving network

local sanitize = require 'sanitize'

print('Saving model')
sanitize(model)
torch.save(opt.path2save..'/model'..epoch_id..'.t7', model)

print('End of training')
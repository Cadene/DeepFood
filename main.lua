--
--  Copyright (c) 2016, LIP6, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Authors:
--  Remi Cadene - remi.cadene@lip6.fr - github.com/Cadene


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

local cmd = torch.CmdLine()
-- Training options
cmd:option('-pretrain', 'yes', 'Options: yes (finetuning) | no (from scratch)')
cmd:option('-threads', 2, 'Threads number (minimum 2)')
cmd:option('-imageSize', 221, 'w and h of an image to load')
cmd:option('-batchSize', 60, 'Size of a batch (60 for overfeat, 20 for vgg19)')
cmd:option('-netType', 'overfeat', 'Options: overfeat | vgg16 | vgg19')
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

if opt.pretrain == 'no' then
    assert(opt.lrf_conv == 1,
        'The learning rate of each weights must be equal if you don\'t want to fine tune.')
end

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
    cutorch.setDevice(1) -- use `CUDA_VISIBLE_DEVICES=i th main.lua`
    cutorch.manualSeed(opt.seed)
    require 'cunn'
    require 'cudnn'
end


-------------------------------------
-- Creating dataLoaders if not cached

opt.trainCache = paths.concat(opt.path2cache, 'trainCache.t7')
opt.testCache = paths.concat(opt.path2cache, 'testCache.t7')
opt.path2mean = paths.concat(opt.path2data, 'mean.jpg') -- sent to init_thread
opt.path2std = paths.concat(opt.path2data, 'std.jpg')   -- sent to init_thread

local loadSize   = {3, opt.imageSize, opt.imageSize}
local sampleSize = {3, opt.imageSize, opt.imageSize}

if not paths.filep(opt.trainCache) and not paths.filep(testCache) then
    print('Creating train metadata')
    trainLoader = dataLoader{
        paths = {paths.concat(opt.path2data, 'train')},
        loadSize = loadSize,
        sampleSize = sampleSize,
        split = 100,
        verbose = true
    }
    print('Creating test metadata')
    testLoader = dataLoader{
        paths = {paths.concat(opt.path2data, 'test')},
        loadSize = loadSize,
        sampleSize = sampleSize,
        split = 0,
        verbose = true,
        forceClasses = trainLoader.classes -- force consistent class indices between trainLoader and testLoader
    }
    torch.save(opt.trainCache, trainLoader)
    torch.save(opt.testCache, testLoader)
    trainLoader = nil
    testLoader = nil
    collectgarbage()
end

-------------------------------------
-- Lunching Threads and recovering dataLoaders

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
opt.nBatchTrain = math.floor(nTrain / opt.batchSize)
opt.nBatchTest  = math.floor(nTest / opt.batchSize)

-------------------------------------
-- Model

print('Loading model : '..opt.netType)
paths.dofile(opt.path2model)

if opt.pretrain == 'no' then
    print('/!\\ Reseting weights in order to train from scratch')
    model:reset()
end

if opt.cuda then model:cuda() end

print('Input = '..opt.batchSize..'x3x'..model.imageSize..'x'..model.imageSize)
print(model)

assert(opt.imageSize == model.imageSize)
assert(opt.netType == model.name)
assert(model.params_conv > 0)

local inputs = torch.CudaTensor()
local targets = torch.CudaTensor()

print('Reshaping parameters and gradParameters')
local parameters, gradParameters = model:getParameters()

local criterion = nn.ClassNLLCriterion()
if opt.cuda then criterion:cuda() end

-- local confusion   = optim.ConfusionMatrix(nClasses)
local trainLogger = optim.Logger(paths.concat(opt.path2save, 'train.log'))
local testLogger  = optim.Logger(paths.concat(opt.path2save, 'test.log'))
local lossLogger  = optim.Logger(paths.concat(opt.path2save, 'loss.log'))

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
    print('==> Doing epoch on training data:')
    print("==> Online epoch # " .. epoch_id)

    cutorch.synchronize()
    model:training()

    tmEpoch:reset()

    top1_epoch = 0
    top5_epoch = 0
    loss_epoch = 0

    local shuffle = torch.randperm(nTrain) -- if global not upvalue

    batch_id = 1
    for i = 1, opt.nBatchTrain do
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

    top1_epoch = top1_epoch * 100 / (opt.nBatchTrain * opt.batchSize)
    top5_epoch = top5_epoch * 100 / (opt.nBatchTrain * opt.batchSize)
    loss_epoch = loss_epoch / opt.nBatchTrain

    trainLogger:add{
        ['top1 accuracy (%)'] = top1_epoch / opt.nBatchTrain,
        ['top5 accuracy (%)'] = top5_epoch / opt.nBatchTrain,
        ['avg loss'] = loss_epoch / opt.nBatchTrain
    }

    print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'Avg loss (per batch): %.2f \t '
                          .. 'Avg accuracy(%%):\t top-1 %.2f\ttop-5 %.2f\n',
        epoch_id, tmEpoch:time().real, loss_epoch, top1_epoch, top5_epoch))
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
    end
    top1 = top1 * 100 / opt.batchSize
    top5 = top5 * 100 / opt.batchSize

    lossLogger:add{
        ['batch id'] = batch_id,
        ['loss'] = loss
    }
    loss_epoch = loss_epoch + loss

    local current_lr = config.learningRate / (1 + config.evalCounter*config.learningRateDecay)

    print(string.format('Epoch: [%d][%d/%d]\t'
        .. 'Time(sec): Batch %.3f\tDataLoading %.3f\tLR: %.2e'
        .. '\tLoss: %.4f\tAccuracy(%%):\tTop1 %.2f\tTop5 %.2f',
        epoch_id, batch_id, opt.nBatchTrain, tmBatch:time().real, 
        dataLoadingTime, current_lr, loss, top1, top5))

    batch_id = batch_id + 1
    tmDataload:reset()
end

-------------------------------------
-- Testing

function test()
    print('==> Doing epoch on testing data:')
    print("==> Online epoch # " .. epoch_id)

    cutorch.synchronize()
    model:evaluate()

    tmEpoch:reset()

    top1_epoch = 0
    top5_epoch = 0
    loss_epoch = 0

    batch_id = 1
    for i = 1, opt.nBatchTest do
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

    top1_epoch = top1_epoch * 100 / (opt.nBatchTest * opt.batchSize)
    top5_epoch = top5_epoch * 100 / (opt.nBatchTest * opt.batchSize)
    loss_epoch = loss_epoch / opt.batchSize

    testLogger:add{
        ['top1 accuracy (%)'] = top1_epoch / opt.nBatchTest,
        ['top5 accuracy (%)'] = top5_epoch / opt.nBatchTest,
        ['avg loss'] = loss_epoch / opt.nBatchTest
    }
    print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f\t'
        .. 'Avg loss (per batch): %.2f \t '
        .. 'Avg accuracy(%%):\t top-1 %.2f\ttop-5 %.2f\n',
        epoch_id, tmEpoch:time().real, loss_epoch, top1_epoch, top5_epoch))
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
    end
    top1 = top1 * 100 / opt.batchSize
    top5 = top5 * 100 / opt.batchSize

    print(string.format('Epoch: Testing [%d][%d/%d]\t'
        .. 'Time(sec): Batch %.3f\tDataLoading %.3f\t'
        .. 'Loss: %.4f\tAccuracy(%%):\tTop1 %.2f\tTop5 %.2f',
        epoch_id, batch_id, opt.nBatchTest, tmBatch:time().real, dataLoadingTime,
        loss, top1, top5))

    loss_epoch = loss_epoch + loss
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
while epoch_id < opt.nb_epoch do
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

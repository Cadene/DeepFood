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
cmd:option('-lr',  8e-1, 'Learning Rate')
cmd:option('-lrd', 3e-4, 'Learning Rate Decay')
cmd:option('-wd',  1e-3, 'Weight Decay')
cmd:option('-m',   0.6, 'Momentum')
cmd:option('-lrf_conv', 10, 'lr will be divided by lrf_conv only for convolution layers')
cmd:option('-pretrain', 'yes', 'Options: yes | no')
opt = cmd:parse(arg or {})

opt.cuda = true
opt.batchSize = 60
opt.nb_epoch = 60
opt.seed = 1337
opt.path2data = '/home/cadene/data/UPMC_Food101_221_augmented' -- images 3*221*221
opt.save_model = false
opt.path2cache = '/home/cadene/doc/A/cache'
opt.path2save = '/home/cadene/doc/A/rslt_debug'
opt.threads = 3

opt.imageSize = 221

-------------------------------------
-- Info

idGPU = os.getenv('CUDA_VISIBLE_DEVICES')

print("Lunching using pid = "..unistd.getpid().." on CPU")
print("Lunching using GPU = "..idGPU)
print("Options : ", opt)

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
-- Dataset

-- function load_recipe101(path2data)
--     local path2esc = {'.', '..', '.DS_Store', '._.DS_Store'}
--     local path2train = path2data..'/train/'
--     local path2test  = path2data..'/test/'
--     local path2img, label = {}, {} -- all images names and label=[1,101]
--     local p2i_train, l_train = {}, {} -- only train
--     local p2i_test, l_test = {}, {}
--     local label2class = {} -- label (int) to class (string)
--     local is_in = function (string, path2esc)
--         for k, name in pairs(path2esc) do
--             if string == name then
--                 return true
--             end
--         end
--         return false
--     end
--     local label = 1
--     for _, class in pairs(paths.dir(path2train)) do
--         if not is_in(class, path2esc) then -- rm path2esc values
--             for _, path_img in pairs(paths.dir(path2train..class)) do
--                 if not is_in(path_img, path2esc) then
--                     path2img = paths.concat(path2train, class, path_img)
--                     table.insert(p2i_train, path2img)
--                     table.insert(l_train, label)
--                 end
--             end
--             label2class[label] = class
--             label = label + 1
--         end
--     end
--     label = 1
--     for _, class in pairs(paths.dir(path2test)) do
--         if not is_in(class, path2esc) then -- rm path2esc values
--             for _, path_img in pairs(paths.dir(path2test..class)) do
--                 if not is_in(path_img, path2esc) then
--                     path2img = paths.concat(path2test, class, path_img)
--                     table.insert(p2i_test, path2img)
--                     table.insert(l_test, label)
--                 end
--             end
--             label = label + 1
--         end
--     end
--     local trainset = {
--         path  = p2i_train,
--         label = l_train,
--         size  = #p2i_train
--     }
--     local testset = {
--         path  = p2i_test,
--         label = l_test,
--         size = #p2i_test
--     }
--     return trainset, testset, label2class
-- end

-- if paths.filep(opt.path2save..'/trainset.t7') then
--     trainset = torch.load(opt.path2save..'/trainset.t7')
--     testset  = torch.load(opt.path2save..'/testset.t7')
--     label2class = torch.load(opt.path2save..'/label2class.t7')
-- else
--     trainset, testset, label2class = load_recipe101(opt.path2data)
--     torch.save(opt.path2save..'/trainset.t7', trainset)
--     torch.save(opt.path2save..'/testset.t7', testset)
--     torch.save(opt.path2save..'/label2class.t7', label2class)
-- end
-- nb_class = #label2class

-------------------------------------
-- Threads

threads.serialization('threads.sharedserialize')
do 
    if opt.threads > 0 then
        local options = opt
        pool = threads.Threads(
            opt.threads,
            function(thread_id)
                print('Starting a new thread num ' .. thread_id)
                require 'torch'
                --require 'cutorch'
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

if opt.cuda then
    SpatialConvolution = cudnn.SpatialConvolution
    SpatialMaxPooling  = cudnn.SpatialMaxPooling
else
    SpatialConvolution = nn.SpatialConvolutionMM
    SpatialMaxPooling  = nn.SpatialMaxPooling
end

function load_overfeat()
    print('Creating overfeat')
    -- conv
    local conv = nn.Sequential()
    conv:add(SpatialConvolution(3, 96, 7, 7, 2, 2))
    conv:add(nn.ReLU(true))
    conv:add(SpatialMaxPooling(3, 3, 3, 3))
    conv:add(SpatialConvolution(96, 256, 7, 7, 1, 1))
    conv:add(nn.ReLU(true))
    conv:add(SpatialMaxPooling(2, 2, 2, 2))
    conv:add(SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
    conv:add(nn.ReLU(true))
    conv:add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    conv:add(nn.ReLU(true))
    conv:add(SpatialConvolution(512, 1024, 3, 3, 1, 1, 1, 1))
    conv:add(nn.ReLU(true))
    conv:add(SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1))
    conv:add(nn.ReLU(true))
    -- classifier
    local classif = nn.Sequential()
    classif:add(SpatialMaxPooling(3, 3, 3, 3))
    classif:add(SpatialConvolution(1024, 4096, 5, 5, 1, 1))
    classif:add(nn.ReLU(true))
    classif:add(nn.Dropout(0.5))
    classif:add(SpatialConvolution(4096, 4096, 1, 1, 1, 1))
    classif:add(nn.ReLU(true))
    classif:add(nn.Dropout(0.5))
    classif:add(SpatialConvolution(4096, nClasses, 1, 1, 1, 1))
    classif:add(nn.View(nClasses))
    classif:add(nn.LogSoftMax())
    -- model
    local model = nn.Sequential()
    model:add(conv)
    model:add(classif)

    if opt.pretrain == 'yes' then
        print('Loading overfeat weights')
        local m = model:get(1).modules
        local ParamBank = require 'ParamBank'
        local offset = 0
        ParamBank:init("net_weight_1")
        ParamBank:read(        0, {96,3,7,7},      m[offset+1].weight)
        ParamBank:read(    14112, {96},            m[offset+1].bias)
        ParamBank:read(    14208, {256,96,7,7},    m[offset+4].weight)
        ParamBank:read(  1218432, {256},           m[offset+4].bias)
        ParamBank:read(  1218688, {512,256,3,3},   m[offset+7].weight)
        ParamBank:read(  2398336, {512},           m[offset+7].bias)
        ParamBank:read(  2398848, {512,512,3,3},   m[offset+9].weight)
        ParamBank:read(  4758144, {512},           m[offset+9].bias)
        ParamBank:read(  4758656, {1024,512,3,3},  m[offset+11].weight)
        ParamBank:read(  9477248, {1024},          m[offset+11].bias)
        ParamBank:read(  9478272, {1024,1024,3,3}, m[offset+13].weight)
        ParamBank:read( 18915456, {1024},          m[offset+13].bias)
        -- ParamBank:read( 18916480, {4096,1024,5,5}, m[offset+16].weight)
        -- ParamBank:read(123774080, {4096},          m[offset+16].bias)
        -- ParamBank:read(123778176, {4096,4096,1,1}, m[offset+18].weight)
        -- ParamBank:read(140555392, {4096},          m[offset+18].bias)
        -- ParamBank:read(140559488, {1000,4096,1,1}, m[offset+20].weight)
        -- ParamBank:read(144655488, {1000},          m[offset+20].bias)
    end
    model.imageSize = 221
    model.name = 'overfeat'
    model.params_conv = 18916480
    return model
end

-- load_vgg16 for finetuning
function load_vgg16()
    local model = torch.load('vgg16/model0.t7')
    model:remove(40) -- cudnn.SoftMax
    model:remove(39) -- nn.Linear(4096,1000)
    model:add(nn.Linear(4096, nClasses))
    model:add(nn.LogSoftMax())
    model:get(33):reset()
    model:get(36):reset()
    model.imageSize = 224
    model.name = 'vgg16'
    return model
end

print('Loading model')
local model = load_overfeat()

if opt.cuda then
    -- cudnn.convert(model, cudnn)
    model:cuda()
end

print(model.name..' is loaded !')
--assert(opt.imageSize == model.imageSize)
print('Input = '..opt.batchSize..'x3x'..model.imageSize..'x'..model.imageSize)
print(model)

local inputs = torch.CudaTensor()
local targets = torch.CudaTensor()

print('Reshaping parameters and gradParameters')
local parameters, gradParameters = model:getParameters()

local criterion = nn.ClassNLLCriterion()
if opt.cuda then criterion:cuda() end

local confusion   = optim.ConfusionMatrix(nClasses)
local trainLogger = optim.Logger(paths.concat(opt.path2save, 'train.log'))
local testLogger  = optim.Logger(paths.concat(opt.path2save, 'test.log'))

-------------------------------------
-- Optimizer

-- sgd
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

    batch_id = 1
    for i = 1, opt.epochSize do
        local indexStart = (i-1) * opt.batchSize + 1
        local indexEnd = (indexStart + opt.batchSize - 1)
        pool:addjob(
            function()
                local inputs, labels = trainLoader:get(indexStart, indexEnd)
                return inputs, labels
            end,
            trainBatch
        )
    end

    pool:synchronize()
    cutorch.synchronize()

    top1_epoch = top1_epoch + top1
    top5_epoch = top5_epoch + top5
    loss_epoch = loss_epoch + loss

    trainLogger:add{
        ['top1 accuracy (%)'] = top1_epoch / opt.epochSize,
        ['top5 accuracy (%)'] = top5_epoch / opt.epochSize,
        ['avg loss'] = loss_epoch / opt.epochSize
        --['time (seconds)'] = tm.time().real
    }
    if opt.save_model then
        print('# ... saving model')
        torch.save(paths.concat(opt.path2save, 'model'..epoch_id..'.t7'), model)
    end
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
        local df_do   = criterion:backward(outputs, targets)
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
    print('# --------------------- #')
    print('# ... testing model ... #')
    print('# --------------------- #')
    print('')
    collectgarbage()
    local timer = torch.Timer()
    t_outputs, t_targets = {}, {}
    model:evaluate()
    local batch_id = 1
    for i = 1, trainset.size, opt.batchSize do
        print('Processing batch num '..batch_id)
        if i + opt.batchSize > testset.size then
            b_size = testset.size - i
        else
            b_size = opt.batchSize
        end
        inputs  = torch.zeros(b_size, 3, model.imageSize, model.imageSize)
        targets = torch.zeros(b_size)
        for j = 1, b_size do
            path2img   = paths.concat(opt.path2data,
                label2class[testset.label[i+j]],
                testset.path[i+j])
            inputs[j]  = torch.cdiv(image.load(path2img) - mean, std)
            targets[j] = testset.label[i+j]
        end
        if opt.cuda then
            inputs  = inputs:cuda()
            targets = targets:cuda()
        end
        outputs = model:forward(inputs)
        _, amax = outputs:max(2)
        table.insert(t_outputs, amax:resizeAs(targets))
        table.insert(t_targets, targets:clone())
        print('> seconds : '..timer:time().real)
        batch_id = batch_id + 1
    end
    -- print(confusion)
    confusion:zero()
    for i = 1, #t_outputs do
        confusion:batchAdd(t_outputs[i], t_targets[i])
    end
    confusion:updateValids()
    print('> perf test : '..(confusion.totalValid * 100))
    testLogger:add{['top 1 accuracy (%)'] = confusion.totalValid * 100}
end

-------------------------------------
-- Setting epochs

-- you can interrupt the training process with a SIGUSR1
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

function sanitize (net)
    local list = net:listModules()
    for _,val in ipairs(list) do
        for name,field in pairs(val) do
            if torch.type(field) == 'cdata' then val[name] = nil end
            if (name == 'output' or name == 'gradInput') then
                val[name] = field.new()
            end
        end
    end
end

print('Saving model')
sanitize(model)
torch.save(opt.save..'/model_final.t7', model)

print('End of training')
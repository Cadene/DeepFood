require 'torch'
require 'dataset'
require 'image'
require 'xlua'
local threads = require 'threads'

local tm = torch.Timer()
--local tmBatch = torch.Timer()
--local tmLoad = torch.Timer()

local cmd = torch.CmdLine()
cmd:option('-path2train', '/home/cadene/data/UPMC_Food101_224_augmented/train', '')
cmd:option('-path2cache', 'cache_meanstd', '')
cmd:option('-threads', 8, 'Number of threads.')
cmd:option('-batchSize', 60, 'Number of images a thread load and sum per call.')
cmd:option('-sample_pc', 100, 'Pourcents of the trainset used (by default 100%).')
cmd:option('-imageSize', 224, 'Image size.')
local opt = cmd:parse(arg or {})

print(opt)

os.execute('mkdir -p ' .. opt.path2cache)

if not paths.filep(opt.path2cache..'/trainLoader.t7') then
    print('Creating train metadata')
    trainLoader = dataLoader{
        paths = {opt.path2train},
        sampleSize = {3, opt.imageSize, opt.imageSize},
        split = 100,
        verbose = true
    }
else
    print('/!\\ Loading train metadata')
    trainLoader = torch.load(opt.path2cache..'/trainLoader.t7')
end

local nTrain = trainLoader:size()
local classes = trainLoader.classes
local nSample = math.ceil(nTrain * opt.sample_pc / 100)
local nBatch = math.floor(nSample / opt.batchSize)
local nSampleReal = nBatch * opt.batchSize
print('nTrain: '..nTrain)
print('#classes: '..#classes)
print('/!\\ nSample: '..nSample..' but nSampleReal: '..nSampleReal)

torch.save(opt.path2cache..'/trainLoader.t7', trainLoader)
trainLoader = nil
collectgarbage()

threads.serialization('threads.sharedserialize')
do 
    if opt.threads > 0 then
        local options = opt
        pool = threads.Threads(
            opt.threads,
            function(thread_id)
                print('Starting a new thread # ' .. thread_id)
                require 'torch'
                require 'dataset'
            end,
            function(thread_id)
                opt = options
                sampleHook = function(self, path)
                    collectgarbage()
                    local output = image.load(path, 3, 'float')
                    return output
                end
                trainLoader = torch.load(opt.path2cache..'/trainLoader.t7')
                trainLoader.sampleHook = sampleHook
            end
        )
    else
        pool = {}
        function pool:addjob(f1, f2) f2(f1()) end
        function pool:synchronize() end
    end
end

local shuffle = torch.randperm(nTrain) -- or nSample ?


print('Creating mean')
tm:reset()
local mean = torch.zeros(3, opt.imageSize, opt.imageSize)
local count = 0
for i = 1, nBatch do
    pool:addjob(
        function()
            local indexStart = (i-1) * opt.batchSize + 1
            local indexEnd = (indexStart + opt.batchSize - 1)
            local inputs, _ = trainLoader:get(indexStart, indexEnd, shuffle)
            local outputs = inputs:sum(1)
            return outputs[1]
        end,
        function(mean_batch)
            --local tm_load = tmLoad:time().real
            --tmBatch:reset()
            mean = mean + mean_batch
            count = count + opt.batchSize
            xlua.progress(count, nSample)
            --print('Time: Batch: '..tmBatch:time().real..' & Load: '..tm_load)
            --tmLoad:reset()
        end
    )
end
pool:synchronize()
mean = mean / nSampleReal
torch.save(opt.path2train..'/../mean.t7', mean)
local tm_mean = tm:time().real

local mean_batch = torch.Tensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
for i = 1, opt.batchSize do
    mean_batch[i] = mean:clone()
end

print('Creating std')
tm:reset()
local std = torch.zeros(3, opt.imageSize, opt.imageSize)
local count = 0
for i = 1, nBatch do
    pool:addjob(
        function()
            local indexStart = (i-1) * opt.batchSize + 1
            local indexEnd = (indexStart + opt.batchSize - 1)
            local inputs, _ = trainLoader:get(indexStart, indexEnd, shuffle)
            local outputs = (inputs - mean_batch):pow(2):sum(1)
            return outputs[1]
        end,
        function(std_batch)
            std = std + std_batch
            count = count + opt.batchSize
            xlua.progress(count, nSample)
        end
    )
end
pool:synchronize()
std = (std / nSampleReal):sqrt()
torch.save(opt.path2train..'/../std.t7', std)
local tm_std = tm:time().real

print('Summary mean std on '..opt.threads..' threads with batchSize='..opt.batchSize..':')
print(' Mean: '..nSample..' images takes'..tm_mean..' seconds .')

-- THE END
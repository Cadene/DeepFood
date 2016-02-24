require 'torch'
require 'dataset'
local threads = require 'threads'

local tm = torch.Timer()

local cmd = torch.CmdLine()
cmd:option('-name', 'UPMC_Food101', 'Name of the trainset')
cmd:option('-path2data', '/home/cadene/data/UPMC_Food101/images', '')
cmd:option('-path2augm', '/home/cadene/data/UPMC_Food101_test', '')
cmd:option('-path2cache', 'cache', '')
cmd:option('-imageSize', 256, '256x256^')
cmd:option('-cropSize', 224, '224x224')
cmd:option('-threads', 4, '')
local opt = cmd:parse(arg or {})

os.execute('mkdir -p ' .. opt.path2augm)


print('Creating original train metadata')
trainLoader = dataLoader{
    paths = {paths.concat(opt.path2data, 'train')},
    sampleSize = {0, 0, 0}, -- ugly hack
    split = 100,
    verbose = true
}

print('Creating original test metadata')
testLoader = dataLoader{
    paths = {paths.concat(opt.path2data, 'val')},
    sampleSize = {0, 0, 0}, -- ugly hack
    split = 0,
    verbose = true,
    forceClasses = trainLoader.classes -- force consistent class indices between trainLoader and testLoader
}

local nTrain = trainLoader:size()
local nTest = testLoader:size()
local classes = trainLoader.classes
print('nTrain: '..nTrain)
print('nTest: '..nTest)
print('#classes: '..#classes)

torch.save(opt.path2cache..'/trainLoader.t7', trainLoader)
torch.save(opt.path2cache..'/testLoader.t7', testLoader)

trainLoader = nil
testLoader = nil
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
                testLoader = torch.load(opt.path2cache..'/testLoader.t7')
                trainLoader.sampleHook = sampleHook
                testLoader.sampleHook = sampleHook
            end
        )
    else
        pool = {}
        function pool:addjob(f1, f2) f2(f1()) end
        function pool:synchronize() end
    end
end

print('Creating classes augmented dir')
for i = 1, #classes do
    os.execute('mkdir -p '..paths.concat(opt.path2augm,'train',classes[i]))
    os.execute('mkdir -p '..paths.concat(opt.path2augm,'test',classes[i]))
end

print('Creating augmented trainset')
tm:reset()
for i = 1, nTrain do
    pool:addjob(
        function()
            local imgPath, class = trainLoader:getPath(i)
            local split = imgPath:split('/')
            local imgName = split[#split]
            for _,gravity in pairs{'center','northwest','southwest','northeast','southeast'} do
                os.execute('mogrify -resize "'..opt.imageSize..'x'..opt.imageSize..'^"'
                    ..' -write "'..paths.concat(opt.path2augm,'train',class,gravity..'_'..imgName)..'"'
                    ..' -gravity '..gravity
                    ..' -crop '..opt.cropSize..'x'..opt.cropSize..'+0+0 +repage'
                    ..' "'..imgPath..'"')
                os.execute('mogrify -resize "'..opt.imageSize..'x'..opt.imageSize..'^"'
                    ..' -write "'..paths.concat(opt.path2augm,'train',class,gravity..'_flop_'..imgName)..'"'
                    ..' -gravity '..gravity
                    ..' -flop -crop '..opt.cropSize..'x'..opt.cropSize..'+0+0 +repage'
                    ..' "'..imgPath..'"')
            end
            print(' '..imgName)
        end
    )
end
local tm_augm_train = tm:time().real

print('Creating augmented testset')
for i = 1, nTest do
    pool:addjob(
        function()
            local imgPath, class = testLoader:getPath(i)
            local split = imgPath:split('/')
            local imgName = split[#split]
            for _,gravity in pairs{'center','northwest','southwest','northeast','southeast'} do
                os.execute('mogrify -resize "'..opt.imageSize..'x'..opt.imageSize..'^"'
                    ..' -write "'..paths.concat(opt.path2augm,'test',class,gravity..'_'..imgName)..'"'
                    ..' -gravity '..gravity
                    ..' -crop '..opt.cropSize..'x'..opt.cropSize..'+0+0 +repage'
                    ..' "'..imgPath..'"')
                os.execute('mogrify -resize "'..opt.imageSize..'x'..opt.imageSize..'^"'
                    ..' -write "'..paths.concat(opt.path2augm,'test',class,gravity..'_flop_'..imgName)..'"'
                    ..' -gravity '..gravity
                    ..' -flop -crop '..opt.cropSize..'x'..opt.cropSize..'+0+0 +repage'
                    ..' "'..imgPath..'"')
            end
            print(' '..imgName)
        end
    )
end
local tm_augm_test = tm:time().real

print('Summary data augmentation *10 on '..opt.threads..' threads:')
print(' Trainset: '..nTrain..' images takes'..tm_augm_train..' seconds .')
print(' Testset: '..nTest..' images takes'..tm_augm_test..' seconds.')

-- local mean = torch.zeros(3, 224, 224)
-- local std  = torch.zeros(3, 224, 224)




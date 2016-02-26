require 'image'

paths.dofile('dataset.lua')
-- paths.dofile('util.lua') dataParallel

-------------------------------------
-- Setting mean std

print('Loading mean and std')
local mean = torch.load(opt.path2mean)
local std  = torch.load(opt.path2std)

-------------------------------------
-- Setting loading hook

local loadImage = function(path)
    local input = image.load(path,3,'float')
    return input
end

local sampleHook = function(self, path)
    collectgarbage()
    local input = loadImage(path)
    local output = torch.cdiv(input - mean, std)
    return output
end

--------------------------------------
-- Setting trainLoader

print('Loading train metadata from cache')
trainLoader = torch.load(opt.trainCache)
trainLoader.sampleHook = sampleHook
assert(trainLoader.paths[1] == paths.concat(opt.path2data, 'train'),
      'cached files dont have the same path as opt.path2data. Remove your cached files at: '
         .. opt.trainCache .. ' and rerun the program')
collectgarbage()

--------------------------------------
-- Setting testLoader

print('Loading test metadata from cache')
testLoader = torch.load(opt.testCache)
testLoader.sampleHook = sampleHook
assert(testLoader.paths[1] == paths.concat(opt.path2data, 'test'),
      'cached files dont have the same path as opt.path2data. Remove your cached files at: '
         .. opt.testCache .. ' and rerun the program')
collectgarbage()



require 'image'

paths.dofile('dataset.lua')
-- paths.dofile('util.lua') dataParallel

local trainCache = paths.concat(opt.path2cache, 'trainCache.t7')
local testCache = paths.concat(opt.path2cache, 'testCache.t7')

local path2mean = paths.concat(opt.path2data, 'mean.jpg')
local path2std = paths.concat(opt.path2data, 'std.jpg')

local loadSize   = {3, opt.imageSize, opt.imageSize}
local sampleSize = {3, opt.imageSize, opt.imageSize}

-------------------------------------
-- Setting mean std

print('Loading mean and std')
local mean = image.load(path2mean)
local std  = image.load(path2std)

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
-- Setting trainset

if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHook = sampleHook
   assert(trainLoader.paths[1] == paths.concat(opt.path2data, 'train'),
          'cached files dont have the same path as opt.path2data. Remove your cached files at: '
             .. trainCache .. ' and rerun the program')
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {paths.concat(opt.path2data, 'train')},
      loadSize = loadSize,
      sampleSize = sampleSize,
      split = 100,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   trainLoader.sampleHook = sampleHook
end
collectgarbage()

--------------------------------------
-- Setting testset

if paths.filep(testCache) then
   print('Loading test metadata from cache')
   testLoader = torch.load(testCache)
   testLoader.sampleHook = sampleHook
   assert(testLoader.paths[1] == paths.concat(opt.path2data, 'test'),
          'cached files dont have the same path as opt.path2data. Remove your cached files at: '
             .. testCache .. ' and rerun the program')
else
   print('Creating test metadata')
   testLoader = dataLoader{
      paths = {paths.concat(opt.path2data, 'test')},
      loadSize = loadSize,
      sampleSize = sampleSize,
      split = 0,
      verbose = true,
      forceClasses = trainLoader.classes -- force consistent class indices between trainLoader and testLoader
   }
   torch.save(testCache, testLoader)
   testLoader.sampleHook = sampleHook
end
collectgarbage()





-- mean = image.load(opt.path2dir..'/mean.jpg')
-- std  = image.load(opt.path2dir..'/std.jpg')
-- trainset = torch.load(opt.path2save..'/trainset.t7')
-- testset  = torch.load(opt.path2save..'/testset.t7')
-- label2class = torch.load(opt.path2save..'/label2class.t7')


-- function loadBatch(i)
--     collectgarbage()
--     --print('Loading i='..i, 'threadid='..__threadid)
--     if i + opt.batchSize > trainset.size then
--         b_size = trainset.size - i
--     else
--         b_size = opt.batchSize
--     end
--     local inputs  = torch.zeros(b_size, 3, opt.imageSize, opt.imageSize) -- beware of the 4d tensor
--     local targets = torch.zeros(b_size)
--     for j = 1, b_size do
--         local img_id = shuffle[i+j]
--         -- print('loading image num '..shuffle[i+j]..'/'..trainset.size..'\t'..j..'/'..b_size)
--         path2img = paths.concat(opt.path2dir,
--             label2class[trainset.label[img_id]], 
--             trainset.path[img_id])
--         local img = image.load(path2img)
--         if img:size(1) == 1 then
--             blackandwhite = img[1]:clone()
--             img = torch.Tensor(3, opt.imageSize, opt.imageSize)
--             img[1] = blackandwhite
--             img[2] = blackandwhite
--             img[3] = blackandwhite
--         end
--         inputs[j]  = torch.cdiv(img - mean, std)
--         targets[j] = trainset.label[img_id]
--     end
--     return inputs, targets
-- end


--shuffle = torch.randperm(trainLoader:size()) -- TODO
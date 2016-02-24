require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'

opt = {}
opt.path2networks = 'networks'
nClasses = 101

require('networks/MinMaxTopInstancesPooling')

model = torch.load(opt.path2networks..'/vgg16.t7')
cudnn.convert(model, cudnn)

local weldon = nn.Sequential()
for i = 1, 31 do
    weldon:add(model:get(i)) -- copy conv layers
end

-- Converting first fc layer into a conv layer
weldon:add(cudnn.SpatialConvolution(512,4096,7,7))
weldon:get(32).bias = model:get(33).bias:clone()
weldon:get(32).weight = torch.reshape(model:get(33).weight:clone(), 4096, 512, 7, 7)
weldon:add(model:get(34)) -- add ReLU

-- Transfert layer
weldon:add(cudnn.SpatialConvolution(4096,nClasses,1,1,1,1))

weldon:add(MinMaxTopInstancesPooling(3))
weldon:add(nn.Reshape(nClasses))

model = weldon
collectgarbage()

model.imageSize = 224
model.name = 'weldon_vgg16'
model.params_conv = 14714688

-- local conv_nodes = model:findModules('cudnn.SpatialConvolution')
-- for i = 1, #conv_nodes do
--     local params, gradParams = conv_nodes[i]:getParameters()
--     print(i, params:size(1))
--     model.params_conv = model.params_conv + params:size(1)
-- end
-- print(model.params_conv)
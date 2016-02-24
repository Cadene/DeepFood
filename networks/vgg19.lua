require 'torch'
require 'nn'
require 'cutorch'
require 'cudnn'

-- opt = {}
-- opt.path2networks = '/home/cadene/doc/DeepFood/networks'
-- nClasses = 101

model = torch.load(opt.path2networks..'/vgg19.t7')
cudnn.convert(model, cudnn)

model:remove(46) -- cudnn.SoftMax
model:remove(45) -- nn.Linear(4096,1000)

model:add(nn.Linear(4096, nClasses))
model:add(nn.LogSoftMax())

model:get(42):reset()
model:get(39):reset()

model.imageSize = 224
model.name = 'vgg19'
model.params_conv = 20024384

-- local conv_nodes = model:findModules('cudnn.SpatialConvolution')
-- for i = 1, #conv_nodes do
--     local params, gradParams = conv_nodes[i]:getParameters()
--     print(i, params:size(1))
--     model.params_conv = model.params_conv + params:size(1)
-- end
-- print(model.params_conv)

require 'torch'
require 'nn'
require 'cudnn'

model = torch.load(opt.path2networks..'/vgg16.t7')
cudnn.convert(model, cudnn)

model:remove(40) -- cudnn.SoftMax
model:remove(39) -- nn.Linear(4096,1000)

model:add(nn.Linear(4096, nClasses))
model:add(nn.LogSoftMax())

model:get(33):reset()
model:get(36):reset()

model.imageSize = 224
model.name = 'vgg16'
model.params_conv = 14714688

-- local conv_nodes = model:findModules('cudnn.SpatialConvolution')
-- for i = 1, #conv_nodes do
--     local params, gradParams = conv_nodes[i]:getParameters()
--     print(i, params:size(1))
--     model.params_conv = model.params_conv + params:size(1)
-- end
-- print(model.params_conv)

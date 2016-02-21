require 'torch'
require 'nn'
require 'cudnn'

model = torch.load(opt.path2networks..'/overfeat.t7')
cudnn.convert(model, cudnn)

model:remove(24) -- cudnn.SoftMax
model:remove(23) -- nn.View(1000)
model:remove(22) -- cudnn.SpatialConvolution(4096 -> 1000, 1x1)

model:add(cudnn.SpatialConvolution(4096, nClasses, 1, 1, 1, 1))
model:add(nn.View(nClasses))
model:add(nn.LogSoftMax())

model:get(19):reset()
model:get(16):reset()

model.imageSize = 221
model.name = 'overfeat'
model.params_conv = 18916480

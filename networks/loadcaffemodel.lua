require 'torch'
require 'loadcaffe'
require 'nn'

path = '/home/cadene/doc/DeepFail/networks/vgg19'

model = loadcaffe.load(path..'/deploy.prototxt', path..'/net.caffemodel')

print(model)

torch.save('/home/cadene/doc/DeepFood/networks/vgg19.t7',model)
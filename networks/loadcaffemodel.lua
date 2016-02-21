require 'torch'
require 'loadcaffe'
require 'nn'

path = '/home/cadene/doc/DeepFood/networks/vgg16'

model = loadcaffe.load(path..'/deploy.prototxt', path..'/net.caffemodel')

print(model)

torch.save(path..'/../vgg16.t7',model)
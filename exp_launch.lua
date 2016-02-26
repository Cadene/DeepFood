require 'os'
local unistd  = require "posix.unistd"

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 UPMC_Food101 Launching script')
cmd:text()
cmd:text('Options:')
cmd:option('-netType', 'vgg16', '')
cmd:option('-GPU', 0, '')
cmd:option('-batchSize', 26, '')
cmd:option('-lr', 1e-1, '')
cmd:option('-lrd', 3e-4, '')
cmd:option('-m',        0.6,  'momentum')
cmd:option('-wd',     1e-3, 'weight decay')
cmd:option('-batchSize', 26, 'Size of a batch (60 for overfeat, 20 for vgg19)')
cmd:option('-imageSize', 224, 'w and h of an image to load')
cmd:option('-lrf_conv', 1, 'lr will be divided by lrf_conv only for convolution layers')
cmd:option('-pretrain', 'no', 'Options: yes (finetuning) | no (from scratch)')

local opt = cmd:parse(arg or {})

local fname = 'GPU'..opt.GPU

local exec = 'echo "CUDA_VISIBLE_DEVICES='..opt.GPU..' th main.lua'
for key, value in pairs(opt) do
  if key ~= 'GPU' then
    exec = exec..' -'..key..' '..value
  end
end
exec = exec..'" > '..fname..'.sh'

print(exec)

os.execute(exec)
unistd.sleep(1)
os.execute('chmod 755 '..fname..'.sh')
os.execute('nohup ./'..fname..'.sh > log/'..fname..'.log &')
-- os.execute('rm '..fname..'.sh')

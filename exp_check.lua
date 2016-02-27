require 'os'
require 'torch'

local split = require 'split'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 UPMC_Food101 Checking script')
cmd:text()
cmd:text('Options:')
cmd:option('-gpus', '0,1,2,3,4,5,6,7', 'Options: 0,1,2,3,4,5,6,7 | all')

local opt = cmd:parse(arg or {})

local path = '/home/cadene/doc/DeepFood'
local checkpoint = path
local log = paths.concat(path,'log')

local folder_tab = {}
if opt.gpus == 'all' then
  for folder in paths.iterdirs(path) do
    table.insert(folder_tab, folder)
  end
else
  for _,idGPU in pairs(split(opt.gpus,',')) do
    table.insert(folder_tab, 'GPU'..idGPU)
  end
end
for _,folder in pairs(folder_tab) do
  local opt = torch.load(paths.concat(checkpoint,folder,'opt.t7'))
  print("\n"..'# '..folder)
  print('')
  print('netType: '..opt.netType)
  print('pretrain: '..opt.pretrain)
  print('lr: '..opt.lr)
  print('lrd: '..opt.lrd)
  print('m: '..opt.m)
  print('wd: '..opt.wd)
  print('')
  local f = assert(io.open(paths.concat(checkpoint,folder,'launchdate.log'),"r"))
  print(f:read("*all"))
  f:close()
  local f = assert(io.open(paths.concat(checkpoint,folder,'train.log'),"r"))
  print(f:read("*all"))
  f:close()
  local f = assert(io.open(paths.concat(checkpoint,folder,'test.log'),"r"))
  print(f:read("*all"))
  f:close()
  local f = io.open(paths.concat(log,folder..'.log'),"r")
  if f~=nil then
    local len = f:seek('end')
    f:seek('set', len-1024/2)
    print(f:read("*a"))
    f:close()
  else
    print('No such file '..paths.concat(log,folder..'.log'))
  end
end

require 'os'
require 'torch'
local unistd  = require "posix.unistd"

local split = require 'split'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 UPMC_Food101 Killing script')
cmd:text()
cmd:text('Options:')
cmd:option('-all', 'no', 'Kill all processus')
cmd:option('-pids', 'none', 'Processus id, Options: none | 1453,1234,535')
cmd:option('-gpus', '0,1,2,5', 'GPU id')
cmd:option('-interrupt', 'no', 'Interrupt training at the end of the epoch')

local opt = cmd:parse(arg or {})

local path = '/home/cadene/doc/DeepFood'

function kill(pid, idGPU)
  if opt.interrupt ~= 'no' then
    local sig = require "posix.signal"
    print('Interrupting processus '..pid)
    sig.kill(pid, sig.SIGUSR1)
  else
    print('Killing processus '..pid)
    os.execute('kill -9 '..pid)
    unistd.sleep(1)
    os.execute('rm -rf '..path..'/GPU'..idGPU)
    os.execute('rm -rf '..path..'/GPU'..idGPU..'.sh')
    os.execute('rm '..path..'/log/GPU'..idGPU..'.log')
  end
end


if opt.all == 'yes' then
  print('Killing all processus')
  os.execute("ps -ef | grep luajit | awk '{print $2}' | xargs kill -9")
  return
end

if opt.pids ~= 'none' then
  local pids = split(opt.pids, ',')
  for _,pid in pairs(pids) do
    kill(pid)
  end
  return
end

if opt.gpus ~= 'none' then
  for _,idGPU in pairs(split(opt.gpus, ',')) do
    local folder = 'GPU'..idGPU
    local f = assert(io.open(path..'/'..folder..'/pid.log',"r"))
    local pid = f:read("*all")
    f:close()
    kill(pid, idGPU)
  end
  return
end


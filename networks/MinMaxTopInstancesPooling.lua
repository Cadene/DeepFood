-- compatible CUDA


local MinMaxTopInstancesPooling, parent = torch.class('MinMaxTopInstancesPooling', 'nn.Module')

-- n: number of top instances
function MinMaxTopInstancesPooling:__init(n)
   parent.__init(self)
   self.n = n

   self.indicesMax = torch.Tensor()
   self.indicesMin = torch.Tensor()
end

function MinMaxTopInstancesPooling:updateOutput(input)
   -- backward compatibility

   if #(#input) == 3 then -- image
    local numChannels = (#input)[1]
    self.output:typeAs(input):resize(numChannels,1,1)
    self.output:zero()
    self.indicesMax:resize(numChannels,self.n,1)
    self.indicesMax:zero()
    self.indicesMin:resize(numChannels,self.n,1)
    self.indicesMin:zero()
    for i=1, numChannels do
      local sortValues, sortIndex = torch.sort(input[i]:reshape((#input)[2]*(#input)[3]),1,true)
      local numValues = (#sortValues)[1]
      local maxValue = 0
      local minValue = 0
      for k=1, self.n do
        -- max
        maxValue = maxValue + sortValues[k]
        self.indicesMax[i][k] = sortIndex[k]

        -- min
        minValue = minValue + sortValues[numValues - k+1]
        self.indicesMin[i][k] = sortIndex[numValues - k+1]
      end
      --print('maxValue',maxValue,'minValue',minValue,'n',self.n)
      self.output[i] = (maxValue + minValue) / (2 * self.n)
    end
  elseif #(#input) == 4 then -- batch
    local batchSize = (#input)[1]
    local numChannels = (#input)[2]
    self.output:typeAs(input):resize(batchSize,numChannels,1,1)
    self.output:zero()
    self.indicesMax:resize(batchSize,numChannels,self.n,1)
    self.indicesMax:zero()
    self.indicesMin:resize(batchSize,numChannels,self.n,1)
    self.indicesMin:zero()
    for i=1, batchSize do
      for j=1, numChannels do
        local sortValues, sortIndex = torch.sort(input[i][j]:reshape((#input)[3]*(#input)[4]),1,true)
        local numValues = (#sortValues)[1]
        local maxValue = 0
        local minValue = 0
        for k=1, self.n do
          -- max
          maxValue = maxValue + sortValues[k]
          self.indicesMax[i][j][k] = sortIndex[k]

          -- min
          minValue = minValue + sortValues[numValues - k+1]
          self.indicesMin[i][j][k] = sortIndex[numValues - k+1]
        end
        --print('maxValue',maxValue,'minValue',minValue,'n',self.n)
        self.output[i][j] = (maxValue + minValue) / (2 * self.n)
      end
    end
  else
    print('error in MinMaxTopInstancesPooling:updateOutput')
   end
   return self.output
end

function MinMaxTopInstancesPooling:updateGradInput(input, gradOutput)

  --print('gradOutput',gradOutput)

  self.gradInput:typeAs(input):resizeAs(input)
  self.gradInput:zero()
  if #(#input) == 3 then -- 1 image
    local numChannels = (#input)[1]
    for i=1, numChannels do
      for k=1, self.n do
        local x = math.floor((self.indicesMax[i][k][1]-1) / (#input)[3]) + 1
        local y = (self.indicesMax[i][k][1]-1) % (#input)[3] + 1
        self.gradInput[i][x][y] = self.gradInput[i][x][y] + gradOutput[i][1][1]

        local x = math.floor((self.indicesMin[i][k][1]-1) / (#input)[3]) + 1
        local y = (self.indicesMin[i][k][1]-1) % (#input)[3] + 1
        self.gradInput[i][x][y] = self.gradInput[i][x][y] + gradOutput[i][1][1]
      end
    end
  elseif #(#input) == 4 then -- batch
    local batchSize = (#input)[1]
    local numChannels = (#input)[2]
     for i=1, batchSize do
      for j=1, numChannels do
        for k=1, self.n do
          local x = math.floor((self.indicesMax[i][j][k][1]-1) / (#input)[4]) + 1
          local y = (self.indicesMax[i][j][k][1]-1) % (#input)[4] + 1
          self.gradInput[i][j][x][y] = self.gradInput[i][j][x][y] + gradOutput[i][j][1][1]

          local x = math.floor((self.indicesMin[i][j][k][1]-1) / (#input)[4]) + 1
          local y = (self.indicesMin[i][j][k][1]-1) % (#input)[4] + 1
          self.gradInput[i][j][x][y] = self.gradInput[i][j][x][y] + gradOutput[i][j][1][1]
        end
      end
    end
  else
    print('error in MinMaxTopInstancesPooling:updateGradInput')
  end
  return self.gradInput
end

function MinMaxTopInstancesPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indicesMax:resize()
   self.indicesMax:storage():resize(0)
   self.indicesMin:resize()
   self.indicesMin:storage():resize(0)
end

function MinMaxTopInstancesPooling:__tostring__()
   local s =  string.format('%s(%d)', torch.type(self), self.n)
   return s
end

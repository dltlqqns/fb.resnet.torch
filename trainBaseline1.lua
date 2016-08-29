--[[
local M = {}
local Trainer = torch.class('Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
  self.model = model
  self.
end

function Trainer:train(epoch, dataloader)
  
  for n, sample in dataloader:run() do
    -- Copy samples to GPU
    self:copySample(sample)
    
    -- Foward / backward
    local outputs = self.model:forward(self.inputs)
    local loss = self.criterion:forward(self.model.output, self.target)
    
    self.model:zeroGradParameters()
    self.criterion:backward(self.model.output, self.target)
    self.model:backward
    
    -- Optimize
    
    --
  end
  
end

function Trainer:test()
end

return M.Trainer
--]]
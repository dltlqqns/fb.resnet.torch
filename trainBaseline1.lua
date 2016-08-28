local M = {}
local Trainer = torch.class('Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
  self.model = model
  self.
end

function Trainer:train()
end

function Trainer:test()
end

return M.Trainer
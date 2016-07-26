local opts = require 'opts'
local models = require 'models/init'

opt = opts.parse({})
--opt.dataset = 'imagenet'
--opt.netType = 'resnet'
opt.depth = 34

local model = models.setup(opt, nil)
model:training()

local input = torch.CudaTensor(3,3,224,224)
local output model:forward(input)
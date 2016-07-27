local opts = require 'opts'
local models = require 'models/init'

opt = opts.parse({})
--opt.dataset = 'imagenet'
--opt.netType = 'resnet'
opt.depth = 152
local batchSize = 16
local imageWidth = 340 --256
local imageHeight = 340 --256

local model = models.setup(opt, nil)
model:training()

local input = torch.CudaTensor(batchSize,3,imageHeight,imageWidth)
local output = model:forward(input)
print(input:size())
print(output:size())
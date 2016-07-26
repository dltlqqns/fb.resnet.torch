require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'

local opts = require('opts')
local opt = opts.parse(arg)
local model = require('models/basic')(opt)
local criterion = nn.MSECriterion():cuda()

local input = torch.CudaTensor()
input:resize(10,3,16,16):fill(1)
--print(input)
local output = model:forward(input)
--print(output:size())
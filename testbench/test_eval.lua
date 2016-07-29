require 'torch'
local eval = require 'eval'

-- getPred
--[[
local hm = torch.zeros(2,3,5,5)
hm[1][1][3][4] = 1
hm[1][1][3][5] = 2
hm[1][2][4][5] = 1
hm[1][3][1][2] = 1
hm[2][1][1][3] = 3
--print(hm)
local pred, visible = eval.getPred_hm(hm, 'max')
print(pred)
print(visible)
--]]

-- getAccuracy
--[[
local batchSize = 2
local nPart = 3
local normalized_dists = torch.rand(batchSize, nPart, 1)
local pred_invisible = {}
local invisible = torch.ByteTensor(batchSize, nPart, 1):bernoulli(0.5)
local thres = 0.5
local acc = eval.getAccuracy(normalized_dists, pred_invisible, invisible, thres, 'normal')
print(normalized_dists)
print(invisible)
print(acc)
--]]

-- getPerformance
---[[
local output = torch.zeros(2,3,5,5)
output[1][1][3][4] = 1
output[1][1][3][5] = 2
output[1][2][4][5] = 1
output[1][3][1][2] = 1
output[2][1][1][3] = 3
local sample = {}
sample.parts_hm = torch.Tensor({{{1,5},{4,5},{1,2}},{{1,3},{1,1},{1,1}}})
local acc = eval.getPerformance(output,sample, 'mpii')
print(acc)
--]]
local Dataset = require('datasets/mpii')
local opts = require('opts')
local vis = require('visualize')

local imageInfo = torch.load('gen/mpii.t7')
local opt = opts.parse(arg)
local split = 'train'
local dataset = Dataset(imageInfo, opt, split)

-- Get size
print(dataset:size())

-- Get data from idx
local idx = 2
local sample = dataset:get(idx)
image.display(sample.input)

-- Preprocess
sample_preprocessed = dataset:preprocess()(sample)
local heatmap = torch.zeros(sample_preprocessed.target:size(2),sample_preprocessed.target:size(3))
for iPart = 1, 16 do
  heatmap:add(sample_preprocessed.target[iPart])
end
image.display(heatmap)
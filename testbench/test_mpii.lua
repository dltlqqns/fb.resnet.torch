local Dataset = require('datasets/mpii')
local opts = require('opts')

local imageInfo = torch.load('gen/mpii.t7')
local opt = opts.parse(arg)
local split = 'train'
local dataset = Dataset(imageInfo, opt, split)

-- Get size
print(dataset:size())

-- Get data from idx
local idx = 2
local data = dataset:get(idx)
image.display(data.input)

-- Preprocess
data_preprocessed = dataset:preprocess(data)
image.display(data_preprocessed.input)
local heatmap = torch.zeros(data_preprocessed.target:size(2),data_preprocessed.target:size(3))
for iPart = 1, 16 do
  heatmap:add(data_preprocessed.target[iPart])
end
image.display(heatmap)
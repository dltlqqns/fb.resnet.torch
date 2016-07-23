local image = require 'image'
local ffi = require 'ffi'
local t_shg = require 'datasets/transforms_ym.lua'
local M = {}
local mpiiDataset = torch.class('resnet.mpiiDataset', M)

function mpiiDataset:__init(imageInfo, opt, split)
	self.imageInfo = imageInfo[split]
  self.opt = opt
  self.split = split
  self.nPart = imageInfo.parts:size(2)  -- ??
  print(self.nPart) -- ??
end

function mpiiDataset:get(i)
  -- Generate sample
  local image = self.loadImage(paths.concat(opt.datasetDir, 'images', ffi.string(self.imageInfo.imagePaths[i]:data())))
  local image_cropped = t_shg.crop(image, imageInfo.centers[i], imageInfo.scales[i], 0, self.opt.inputRes)
  local joint = {}
  for j = 1, self.nPart do
    joint[j] = t_shg.transform(imageInfo.parts[i][j], imageInfo.centers[i], imageInfo.scales[i], 0, self.opt.outputRes) --?? indexing
  end
  
  return {
    input = image_cropped,
    joint = joint,
  }
end

function mpiiDataset:loadImage(path)
  local ok, input = pcall(function()
      return image.load(path, 3, 'float')
  end)
  assert(ok, 'image loading error')
  return input
end


function mpiiDataset:size()
  return self.imageInfo.centers:size(1)
end

function mpiiDataset:preprocess(sample)
  -- Data augmentation
  local s = 
  local f = 
  local r = 
  
  local input = t.(sample.image_cropped, s,f,r)
  local heatmap = generateHeatmap(sample.joint, s,f,r)
  
  return {
    input = input,
    target = heatmap,
  }
end

return M.mpiiDataset

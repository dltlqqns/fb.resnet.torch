local image = require 'image'
local ffi = require 'ffi'
local t = require 'datasets/transforms_ym.lua'
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
  local input = self.loadImage(paths.concat(opt.datasetDir, 'images', ffi.string(self.imageInfo.imagePaths[i]:data())))
  
  return {
    input = input,
    joint = imageInfo.parts[i],
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
  
  local w1 = math.ceil((sample.input:size(3) - size)/2)
  local h1 = math.ceil((sample.input:size(2) - size)/2)
  local input = image.crop(sample.input, w1,h1,w1+size,w2+size)
  local heatmap = torch.zeros(self.nPart, self.opt.outputRes, self.opt.outputRes)
  for iPart = 1, self.nPart do
    heatmap[iPart] = drawGaussain(self.opt.outputRes, transform(), self.opt.sigma)
  end
  
  --[[
  local image_cropped = t.crop(image, imageInfo.centers[i], imageInfo.scales[i], 0, self.opt.inputRes)
  
  -- Data augmentation
  local s = 
  local f = 
  local r = 
  
  local input = t.(sample.image_cropped, s,f,r)
  local heatmap = generateHeatmap(sample.joint, s,f,r)
  --]]
  
  return {
    input = input,
    target = heatmap,
  }
end

return M.mpiiDataset

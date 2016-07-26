local image = require 'image'
local ffi = require 'ffi'
local t = require 'datasets/transforms_ym'
local M = {}
local mpiiDataset = torch.class('resnet.mpiiDataset', M)

function mpiiDataset:__init(imageInfo, opt, split)
	self.imageInfo = imageInfo[split]
  self.opt = opt
  self.split = split
  self.nPart = imageInfo[split].parts:size(2)
end

function mpiiDataset:get(i)
  -- Generate sample
  local path = paths.concat(self.opt.datasetDir, 'images', ffi.string(self.imageInfo.imagePaths[i]:data()))
  local input = mpiiDataset:loadImage(path)
  local center_yx = torch.zeros(2)
  center_yx[1] = self.imageInfo.centers[i][2]
  center_yx[2] = self.imageInfo.centers[i][1]
  local joint_yx = torch.zeros(self.nPart,2)
  joint_yx[{{},1}] = self.imageInfo.parts[i][{{},2}]
  joint_yx[{{},2}] = self.imageInfo.parts[i][{{},1}]
  
  return {
    input = input,
    joint_yx = joint_yx,
    center_yx = center_yx,
    scale = self.imageInfo.scales[i],
    visible = self.imageInfo.visibles[i],
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

function mpiiDataset:preprocess()
  -- TODO: color normalize
  -- TODO: data augmentation
  return function(sample)
    -- Crop input image
    local input = t.crop(sample.input, sample.center_yx, sample.scale, self.opt.inputRes)
    -- Generate heatmap
    local heatmap = torch.zeros(self.nPart, self.opt.outputRes, self.opt.outputRes)
    local tt = t.getTransformOrig2Crop(sample.center_yx, sample.scale, self.opt.outputRes)
    for iPart = 1, self.nPart do
      if sample.visible[iPart]~=0 then
        local p = t.transform(sample.joint_yx[iPart], tt)
        heatmap[iPart] = t.drawGaussian(self.opt.outputRes, p, self.opt.sigma)
      end
    end
    --input:add() ??
    --input:div() ??
    
    --[[
    -- Data augmentation
    local s = 
    local f = 
    local r = 
    
    local input = t.(sample.image_cropped, s,f,r)
    local heatmap = generateHeatmap(sample.joint_yx, s,f,r)
    --]]
    
    return {
      input = input,
      target = heatmap,
    }
  end
end

return M.mpiiDataset

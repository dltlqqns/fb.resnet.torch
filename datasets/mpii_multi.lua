local image = require 'image'
local ffi = require 'ffi'
local M = {}
local mpiimultiDataset = torch.class('resnet.mpiimultiDataset', M)

function mpiimultiDataset:__init(imageInfo, opt, split)
  self.imageInfo = imageInfo[split]
  self.opt = opt
  self.split = split
  self.nPart = 16
end

function mpiimultiDataset:get(i)
  local path = paths.concat(self.opt.datasetDir, 'images', ffi.string(self.imageInfo.imagePaths[i]:data()))
  local input = mpiimultiDataset:loadImage(path)
  local bb = self.imageInfo.bbs[i]
  
  return {
    input = input,
    bb = bb,
    --labels = labels,
    --joints = joints,
  }
end

function mpiimultiDataset:loadImage(path)
  local ok, input = pcall(function()
    return image.load(path, 3, 'float')
  end)
  assert(ok, 'image loading error')
  return input
end

function mpiimultiDataset:size()
end

function mpiimultiDataset:preprocess()
end

return M.mpiimultiDataset

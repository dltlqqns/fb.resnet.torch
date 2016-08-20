local image = require 'image'
local ffi = require 'ffi'
local M = {}
local mpiimultiDataset = torch.class('resnet.mpiimultiDataset', M)

function mpiimultiDataset:__init(imageInfo, opt, split)
end

function mpiimultiDataset:get(i)
  local path = paths.concat()
  local input = mpiimultiDataset:loadImage(path)
  return {
    input = input,
    bb = bb,
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

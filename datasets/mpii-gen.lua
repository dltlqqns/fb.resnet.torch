local ffi = require('ffi')
local M = {}

-- input: split | train or val
function getImageLabels(split)
  
  assert(split=='train' or split=='valid', 'wrong input type')
  
  -- Read split data from file
  local refDir = '/home/yumin/codes/pose-hg-train/data/mpii'
  local root = hdf5.open(paths.concat(refDir, 'annot', split .. '.h5'))
  --print(root:read('center'):all())
  local centers = torch.FloatTensor(root:read('center'):all():size()):copy(root:read('center'):all())
  local scales = torch.FloatTensor(root:read('scale'):all():size()):copy(root:read('scale'):all())
  local parts = torch.FloatTensor(root:read('part'):all():size()):copy(root:read('part'):all())
  local visibles = torch.FloatTensor(root:read('visible'):all():size()):copy(root:read('visible'):all())
  
  local nImg = centers:size(1)
  local maxLength = 15
  
  -- Load image file paths
  local imagePaths = torch.CharTensor(nImg, maxLength):zero()
  namesFile = io.open(paths.concat(refDir, 'annot', split .. '_images.txt'))
  local idx = 1
  for line in namesFile:lines() do
    -- Set image paths
    ffi.copy(imagePaths[idx]:data(), line)
    idx = idx + 1
  end
  namesFile:close()
  
  return imagePaths, centers, scales, parts, visibles
end

function M.exec(opt, cacheFile)
  
  local trainImagePaths, trainCenters, trainScales, trainParts, trainVisibles = getImageLabels('train')
  local valImagePaths, valCenters, valScales, valParts, valVisibles = getImageLabels('valid')
    
  local info = {
      basedir = opt.datasplitDir,
      train = {
        imagePaths = trainImagePaths,
        centers = trainCenters,
        scales = trainScales,
        parts = trainParts,
        visibles = trainVisibles,
      },
      val = {
        imagePaths = valImagePaths,
        centers = valCenters,
        scales = valScales,
        parts = valParts,
        visibles = valVisibles,
      },
    }
  
  print(info.train.centers:size())
  print("Saving list of images to " .. cacheFile)
  torch.save(cacheFile, info)
  
  local info_small = {
      basedir = opt.datasplitDir,
      train = {
        imagePaths = trainImagePaths:narrow(1,1,1000),
        centers = trainCenters:narrow(1,1,1000),
        scales = trainScales:narrow(1,1,1000),
        parts = trainParts:narrow(1,1,1000),
        visibles = trainVisibles:narrow(1,1,1000),
      },
      val = {
        imagePaths = valImagePaths,
        centers = valCenters,
        scales = valScales,
        parts = valParts,
        visibles = valVisibles,
      },
    }
  
  print(info_small.train.centers:size())
  print("Saving list of small images to ...")
  local _,_,filename_without_extension = string.find(cacheFile, "^(.*)%.[^%.]*$")
  torch.save(filename_without_extension .. '_small.t7', info_small)
  return info
end

return M

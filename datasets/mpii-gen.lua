ffi = require 'ffi'

local M = {}

function findImages()
  local imagePaths = torch.CharTensor()
  
  for i = 1, #paths do
    ffi.copy(imagePaths[i].data(), paths[i])
  end
  
  return imagePaths, labels
end

function M.exec(opt, cacheFile)
  local trainDir = paths.concat(opt.datasetDir,) 
  local valDir = paths.concat(opts.datasetDir,)
  
  local valImagePaths, valLabels = findImages(valDir)
  local trainImagePaths, trainLabels = findImages(trainDir)
    
  local info = {
      basedir = opt.datasetDir,
      train = {
        imagePaths = trainImagePaths,
        labels = trainLabels,
      },
      val = {
        imagePaths = valImagePaths,
        labels = valLabels,
      },
    }
  
  print("Saving list of images to " .. cacheFile)
  torch.save(cacheFile, info)
  return info
end

return M

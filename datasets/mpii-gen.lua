local M = {}

function findImages()
  local imagePath = torch.CharTensor()
  
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

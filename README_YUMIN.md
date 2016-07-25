transform_ym.lua
  padncrop(input, lt, br)
  crop(input, center, scale, res)
  drawGaussian(center, sigma, res)
  getTransformOrig2Crop(center, scale, res)
  getTransformCrop2Orig(center, scale, res)
  transform(pt, t)

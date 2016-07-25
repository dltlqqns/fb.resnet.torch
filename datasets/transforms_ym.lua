require 'image'

local M = {}

--[[
function M.Compose(transforms)
  return function(input)
    for _, transform in ipairs(transforms) do
      input = transform(input)
    end
  end
end

function M.ColorNormalize(meanstd)
  return function(input)
    local output = input:clone()
    for i = 1, input:size(1) do
      output[i]:add(-meanstd.mean[i])
      output[i]:div(meanstd.std[i])
    end
    return output
  end
end
--]]

function M.getTransformOrig2Crop(center, scale, res)
  -- original to cropped coordinate
  local w = scale * 200
  local t = torch.eye(3)
  t[1][1] = res / w
  t[2][2] = res / w
  t[1][3] = res * (-center[1]/w + 0.5) + 0.5 -- 0.5 different from SGH
  t[2][3] = res * (-center[2]/w + 0.5) + 0.5 -- 0.5 different from SGH
  
  return t
end

function M.getTransformCrop2Orig(center, scale, res)
  local t = M.getTransformOrig2Crop(center, scale, res)
  t = torch.inverse(t)
  return t
end

function M.transform(pt, t)
  -- original to cropped coordinate
  local pt_ = torch.ones(3)
  pt_[1] = pt[1]
  pt_[2] = pt[2]
  local trans_pt = (t * pt_):sub(1,2):int()
  
  return trans_pt
end

function M.padncrop(input, lt, br)
  assert(input:nDimension()==3, 'wrong input format in function padncrop')
  
  -- Pad original image
  local padx = math.max(0, 1-lt[2], br[2]-input:size(3))
  local pady = math.max(0, 1-lt[1], br[1]-input:size(2))
  local padded = torch.zeros(3, 2*pady+input:size(2), 2*padx+input:size(3))
  padded:sub(1,3,pady+1,pady+input:size(2),padx+1,padx+input:size(3)):copy(input)
  -- Crop image from padded image
  local lt_padded = {pady+lt[1], padx+lt[2]}
  local br_padded = {pady+br[1], padx+br[2]}
  local output = padded:sub(1,3,lt_padded[1],br_padded[1],lt_padded[2],br_padded[2]):clone()
  
  return output
end

function M.crop(input, center, scale, res)
  assert(input:nDimension()==3, 'wrong input format in function crop')
  
  local lt = M.transform({1,1}, M.getTransformCrop2Orig(center, scale, res))
  local br = M.transform({res,res}, M.getTransformCrop2Orig(center, scale, res))
  
  -- Crop
  local cropped = M.padncrop(input, lt, br)  
  -- Resize
  local resized = image.scale(cropped, res, res)
  
  return resized
end

function M.drawGaussian(res, center, sigma)
  -- Return (res x res) image with gaussian heatmap of (center, sigma)
  assert(sigma-math.floor(sigma)==0,'sigma should be integer valued')
  local size = 6*sigma + 1
  local g = image.gaussian(size, 0.25)
  lt_o = {math.max(center[1] - 3*sigma,1), math.max(center[2] - 3*sigma,1)}
  br_o = {math.min(center[1] + 3*sigma,res), math.min(center[2] + 3*sigma,res)}
  lt_g = {math.max(1,lt_o[1]-center[1]+3*sigma+1), math.max(1,lt_o[2]-center[2]+3*sigma+1)}
  br_g = {math.min(size,br_o[1]-center[1]+3*sigma+1), math.min(size,br_o[2]-center[2]+3*sigma+1)}
  
  local output = torch.zeros(res,res)
  output:sub(lt_o[1],br_o[1],lt_o[2],br_o[2]):copy(g:sub(lt_g[1],br_g[1],lt_g[2],br_g[2]))

  return output
end

return M
require 'image'

local M = {}

function M.Compose(transforms)
  return function(input, label)
    for _, transform in ipairs(transforms) do
      input, label = transform(input, label)
    end
    return input
  end
end

function M.ColorNormalize(meanstd)
  return function(input, label)
    local output = input:clone()
    for i = 1, 3 do
      output[i]:add(-meanstd.mean[i])
      output[i]:div(meanstd.std[i])
    end
    return output, label
  end
end

function M.getTransformOrig2Crop(center_yx, scale, res)
  -- original to cropped coordinate
  local w = scale * 200
  local t = torch.eye(3)
  t[1][1] = res / w
  t[2][2] = res / w
  t[1][3] = res * (-center_yx[1]/w + 0.5) + 0.5 -- 0.5 different from SGH
  t[2][3] = res * (-center_yx[2]/w + 0.5) + 0.5 -- 0.5 different from SGH
  
  return t
end

function M.getTransformCrop2Orig(center_yx, scale, res)
  local t = M.getTransformOrig2Crop(center_yx, scale, res)
  t = torch.inverse(t)
  return t
end

function M.transform(pt_yx, t)
  -- original to cropped coordinate
  local pt_yx_ = torch.ones(3)
  pt_yx_[1] = pt_yx[1]
  pt_yx_[2] = pt_yx[2]
  local trans_pt_yx = (t * pt_yx_):sub(1,2):int()
  
  return trans_pt_yx
end

function M.padncrop(input, lt, br)
  -- lt <= . <= br
    assert(input:nDimension()==3, 'wrong input format in function padncrop')
    
  -- Manipulate image
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

function M.crop(input, center_yx, scale, res)
  assert(input:nDimension()==3, 'wrong input format in function crop')
  
  local lt = M.transform({1,1}, M.getTransformCrop2Orig(center_yx, scale, res))
  local br = M.transform({res,res}, M.getTransformCrop2Orig(center_yx, scale, res))
  
  -- Crop
  local cropped = M.padncrop(input, lt, br)  
  -- Resize
  local resized = image.scale(cropped, res, res)
  
  return resized
end

function M.drawGaussian(res, center_yx, sigma)
  -- Return (res x res) image with gaussian heatmap of (center, sigma)
  assert(sigma-math.floor(sigma)==0,'sigma should be integer valued')
  local size = 6*sigma + 1
  local g = image.gaussian(size, 0.25)
  local lt_o = {math.max(center_yx[1] - 3*sigma,1), math.max(center_yx[2] - 3*sigma,1)}
  local br_o = {math.min(center_yx[1] + 3*sigma,res), math.min(center_yx[2] + 3*sigma,res)}
  local lt_g = {math.max(1,lt_o[1]-center_yx[1]+3*sigma+1), math.max(1,lt_o[2]-center_yx[2]+3*sigma+1)}
  local br_g = {math.min(size,br_o[1]-center_yx[1]+3*sigma+1), math.min(size,br_o[2]-center_yx[2]+3*sigma+1)}
  
  --[[
  print('hello')
  print(br_o[1], lt_o[1], br_o[2], lt_o[2])
  print(br_g[1], lt_g[1], br_g[2], lt_g[2])
  print(br_o[1]-lt_o[1], br_o[2]-lt_o[2])
  print(br_g[1]-lt_g[1], br_g[2]-lt_g[2])
  --]]

  local output = torch.zeros(res,res)
  if(br_o[1]-lt_o[1]>0 and br_o[2]-lt_o[2]>0) then
    output:sub(lt_o[1],br_o[1],lt_o[2],br_o[2]):copy(g:sub(lt_g[1],br_g[1],lt_g[2],br_g[2]))
  end

  return output
end

return M

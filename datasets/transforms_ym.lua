require 'image'

local M = {}

function M.Compose(transforms)
  return function(input, joint_yx, visible)
    for _, transform in ipairs(transforms) do
      input, joint_yx, visible = transform(input, joint_yx, visible)
    end
    return input, joint_yx, visible
  end
end

function M.ColorNormalize(meanstd)
  return function(input, joint_yx, visible)
    local output = input:clone()
    for i = 1, 3 do
      output[i]:add(-meanstd.mean[i])
      output[i]:div(meanstd.std[i])
    end
    return output, joint_yx, visible
  end
end

function M.Pad(input, joint_yx, padx, pady)
  if padx > 0 or pady > 0 then
    local nPart = joint_yx:size(1)
    -- pad, image
		local temp = input.new(3, input:size(2) + 2*pady, input:size(3) + 2*padx)
    temp:zero()
        :narrow(2, pady+1, input:size(2))
        :narrow(3, padx+1, input:size(3))
        :copy(input)
    input = temp
    
    -- pad, joint
    local offset = torch.zeros(1,2)
    offset[1][1] = pady
    offset[1][2] = padx
    joint_yx:add(torch.expand(offset,nPart,2))
  end
    
	return input, joint_yx
end

function M.Crop(lt,br)
	return function(input, joint_yx, visible)
    local nPart = joint_yx:size(1)
    local offset = torch.Tensor(1,2)
    
	 	-- Pad
    local padx = math.max(0, 1-lt[2], br[2]-input:size(3))
    local pady = math.max(0, 1-lt[1], br[1]-input:size(2))
    input, joint_yx = M.Pad(input, joint_yx, padx, pady)
        
		-- Crop
		-- crop, image
    -- one-base
    local x1 = padx + lt[2]
    local y1 = pady + lt[1]
    local x2 = padx + br[2]
    local y2 = pady + br[1]
		local output = image.crop(input, x1-1, y1-1, x2, y2) -- zero-base, x2,y2: non-inclusive
		-- crop, joint
    offset = torch.zeros(1,2)
    offset[1][1] = y1
    offset[1][2] = x1
    joint_yx:add(-torch.expand(offset,nPart,2)):add(1)

		return output, joint_yx, visible
	end
end

function M.Resize(res)
  return function(input, joint_yx, visible)
    --assert(input:size(2)==input:size(3), 'current implementation only allows square input')
    local resized = image.scale(input, res, res)
    joint_yx:add(-1):mul(res/input:size(2)):add(1)
    return resized, joint_yx, visible
  end
end

function M.Rotate(deg)
  return function(input, joint_yx, visible)
    if deg ~= 0 then
      local angle = (torch.uniform()-0.5) *deg * math.pi / 180
      local res = input:size(2)
      assert(input:size(2)==input:size(3), 'current implementation only allows square input')
      -- image
      input = image.rotate(input, angle, 'bilinear')
      -- joint
      local r = torch.eye(3)
      local c, s = math.cos(angle), math.sin(angle)
      r[1][1] = c
      r[1][2] = -s
      r[2][1] = s
      r[2][2] = c
      local t = torch.eye(3)
      t[1][3] = -res/2
      t[2][3] = -res/2
      local t_inv = torch.eye(3)
      t_inv[1][3] = res/2
      t_inv[2][3] = res/2
      for iPart = 1 , joint_yx:size(1) do
        joint_yx[iPart] = M.transform(joint_yx[iPart], t_inv * r * t)
      end
    end
    return input, joint_yx, visible
  end
end

function M.Flip()
  return function(input, joint_yx, visible)
    assert(input:size(2)==input:size(3), 'current implementation only allows square input')
    local res = input:size(2)
    bFlip = torch.uniform() < 0.5
    if bFlip then
      local flipped = image.hflip(input)
      -- part id change
      matchedParts = {
        {1,6},    {2,5},    {3,4},
        {11,16},  {12,15},  {13,14}
      }
      joint_yx:narrow(2,2,1):mul(-1):add(res):add(1)
      for i = 1, #matchedParts do
        local idx1, idx2 = unpack(matchedParts[i])
        local tmp = joint_yx:narrow(1, idx1, 1):clone()
        joint_yx:narrow(1, idx1, 1):copy(joint_yx:narrow(1, idx2, 1))
        joint_yx:narrow(1, idx2, 1):copy(tmp)
        tmp = visible[idx1]
        visible[idx1] = visible[idx2]
        visible[idx2] = tmp
      end
      return flipped, joint_yx, visible
    else
      return input, joint_yx, visible
    end
  end
end

function M.ColorJitter()
  return function(input, joint_yx, visible)
    input:narrow(1,1,1):mul(torch.uniform(0.8,1.2)):clamp(0,1)
    input:narrow(1,2,1):mul(torch.uniform(0.8,1.2)):clamp(0,1)
    input:narrow(1,3,1):mul(torch.uniform(0.8,1.2)):clamp(0,1)
    return input, joint_yx, visible
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

-- TODO
function M.generateHeatmap()
  local heatmap
  
  return heatmap
end

function M.drawGaussian(res, center_yx, sigma)
  -- Return (res x res) image with gaussian heatmap of (center, sigma)
  -- TODO: allow float values for center_yx
  assert(sigma-math.floor(sigma)==0,'sigma should be integer valued')
  assert((center_yx-torch.floor(center_yx)):apply(function(x) return x==0 end):mean()==1,'center_yx should be integer valued in current implementation')
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

--[[
function M.padncrop(input, lt, br)
  -- lt <= . <= br
    assert(input:nDimension()==3, 'wrong input format in function padncrop')
    
  -- Manipulate image
  -- Pad original image
  local padx = math.max(0, 1-lt[2], br[2]-input:size(3))
  local pady = math.max(0, 1-lt[1], br[1]-input:size(2))
  local padded = torch.zeros(3, 2*pady+input:size(2), 2*padx+input:size(3))
  padded:sub(1,3,pady+1,pady+input:size(2),padx+1,padx+input:size(3))
        :copy(input)
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
--]]

return M

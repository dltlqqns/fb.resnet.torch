require 'image'
local t = require 'datasets/transforms_ym'
local vis = require 'visualize'

-- Tests
local opts = require 'opts'
local opt = opts.parse({})
local datasets = require 'datasets/init'
local dataset = datasets.create(opt, 'val')
local idx = 3
local sample = dataset:get(idx)
local input = sample.input
local joint_yx = sample.joint_yx
local visible = sample.visible
local padx = 10
local pady = 20
input, joint_yx = t.Pad(input, joint_yx, padx, pady)
local output = input:clone()
local joint_output = joint_yx:clone()
local visible_output = visible:clone()

-- Show original image
--local im = vis.drawSkeleton(input, {}, joint_yx, visible)
--image.display(im)

-- New version
local tt
local scale
local center_yx
local deg
local inputRes = 256

print(sample.visible)

--2. ColorNormalize
--[[
local meanstd = {
    mean = {1,0.5,0.1},
    std = {1,1,1},
  }
tt = t.ColorNormalize(meanstd)
input, joint_yx, visible = tt(input, joint_yx, visible)
--]]

--3. Pad
--[[
local padded = input:clone()
local joint_padded = joint_yx:clone()
padded, joint_padded = t.Pad(input, joint_yx, padx, pady)
im = vis.drawSkeleton(padded, {}, joint_padded, visible)
image.display(im)
--]]

--[[
--4. Crop
scale = sample.scale
center_yx = sample.center_yx
local lt = t.transform({1,1}, t.getTransformCrop2Orig(center_yx, scale, inputRes))
local br = t.transform({inputRes,inputRes}, t.getTransformCrop2Orig(center_yx, scale, inputRes))

input, joint_yx, visible = t.Crop(lt, br)(input, joint_yx, visible)
im = vis.drawSkeleton(input, {}, joint_yx, visible)
image.display(im)

--5. Resize
input, joint_yx, visible = t.Resize(inputRes)(input, joint_yx, visible)
im = vis.drawSkeleton(input, {}, joint_yx, visible)
image.display(im)

--6. Rotate
deg = 30
input, joint_yx, visible = t.Rotate(deg)(input, joint_yx, visible)
im = vis.drawSkeleton(input, {}, joint_yx, visible)
image.display(im)

--7. Flip
input, joint_yx, visible = t.Flip()(input, joint_yx, visible)
im = vis.drawSkeleton(input, {}, joint_yx, visible)
image.display(im)

--8. ColorJitter
input, joint_yx, visible = t.ColorJitter()(input, joint_yx, visible)
im = vis.drawSkeleton(input, {}, joint_yx, visible)
image.display(im)
--]]

--1. Compose
--[[
tt = t.Compose({t.Crop(lt, br),
                t.Resize(inputRes),
                t.Rotate(deg),
                t.Flip(),
                t.ColorJitter(),
              })
output, joint_output, visible_output = tt(output, joint_output, visible_output)
im = vis.drawSkeleton(output, {}, joint_output, visible_output)
image.display(im)
--]]

-------------------------------------------------------------------
--9. circlePatch
local radius = 5
local patch = t.circlePatch(radius)
print(patch)

--10. drawActivation
hm = t.drawActivation(43, {20,30}, 1, 'gaussian')
image.display(hm)

hm = t.drawActivation(43, {20,30}, 2, 'uniform')
image.display(hm)

--11. drawDistActivation
parts_hm = torch.zeros(5,2)
parts_hm[1][1] = 11
parts_hm[1][2] = 22
parts_hm[2][1] = 13
parts_hm[2][2] = 24
parts_hm[3][1] = 15
parts_hm[3][2] = 26
parts_hm[4][1] = 17
parts_hm[4][2] = 28
parts_hm[5][1] = 19 
parts_hm[5][2] = 30
hms = t.drawDistActivation(43, parts_hm, 1, 2, 'uniform')
assert(hms:size(1)==9,'wrong number of channels')
for i = 1, hms:size(1) do
  image.display(hms[i])
end

--12. generateHeatmap
hms = t.generateHeatmap()

-------------------------------------------------------------------
-- Old version
--[[
-- Set parameters
local center = {256,256}
local scale = 1.28 --2.56
local res = 128 --512

--1. getTransformCrop2Orig
local t1 = t.getTransformOrig2Crop(center,scale,res)
print(t1)

--2. getTransformOrig2Crop
local t2 = t.getTransformCrop2Orig(center,scale,res)
print(t2)

--3. transform
local pt = {1,1}
local trans_pt1 = t.transform(pt, t1)
print(trans_pt1)
local trans_pt2 = t.transform(pt, t2)
print(trans_pt2)

--4. padncrop
--local im_padncrop = t.padncrop(im, {51,101},{150,600})
--local im_padncrop = t.padncrop(im, {1,1},{150,600})
local im_padncrop = t.padncrop(im, {-10,-10},{600,600})
image.display(im_padncrop)

--5. crop
local im_crop = t.crop(im, center, scale, res)
image.display(im_crop)

--6. drawGaussian
local res_g = 61
local center_g = {1,100} --{10,25}
local sigma_g = 20
local im_gaussian = t.drawGaussian(res_g, center_g, sigma_g)
image.display(im_gaussian)
--]]
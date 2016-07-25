require 'image'
local t = require 'datasets/transforms_ym'

-- Tests
--local im = image.load('nadal.png', 3, 'float')
local im = image.lena()

-- Show original image
image.display(im)

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

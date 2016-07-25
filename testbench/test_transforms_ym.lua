require 'image'
local t = require 'datasets/transforms_ym.lua'

local im = image.load('nadal.png', 3, 'float')

-- Show original image
image.display(im)

-- Tests

--crop
im_crop = t.crop()
image.display(im_crop)

--scale
im_scale = t.scale()
image.display(im_crop)

--rotation
im_scale = t.rotate()
image.display(im_crop)

--color
im_scale = t.color()
image.display(im_crop)
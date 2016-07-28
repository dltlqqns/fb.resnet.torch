require 'image'
local lena = image.lena()
local visualize = require('visualize')

-- colorHM [0~1]:[blue~red]
--[[
image.display(visualize.colorHM(torch.rand(100,100)))
image.display(visualize.colorHM(torch.zeros(100,100)))
image.display(visualize.colorHM(torch.ones(100,100)))
image.display(visualize.colorHM(15*torch.ones(100,100)))
image.display(visualize.colorHM(-1*torch.ones(100,100)))
--]]

-- compileImages
--[[
image.display(lena)
image.display(visualize.compileImages({lena, lena},2,2,lena:size(2)))
--]]

-- drawOutput
---[[
image.display(visualize.drawOutput(image.scale(lena,64), torch.rand(16,64,64)))
--]]
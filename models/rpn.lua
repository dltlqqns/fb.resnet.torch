local nn = require 'nn'
require 'cunn'

local function createModel(opt)
	local model = nn.Sequential()
	-- preresnet layers with ~16stride
	model:add()
	
	-- 3x3 convolution
	model:add(Convolution(512,256,3,3,1,1,1,1)) -- dimension
	model:add(ReLU(true))	-- true for in-place
	-- Classification label & target box regression
	model:add(nn.ConcatTable()
			:add(Convolution(256,18,1,1,1,1,1,1))
			:add(Convolution(256,36,1,1,1,1,1,1)))
	return model	
end

return createModel

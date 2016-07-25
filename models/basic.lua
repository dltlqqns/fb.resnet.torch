
local function createModel(opt)
  local model = nn.Sequential()
  
  -- Define network
  model:add( cudnn.SpatialConvolution(3,16,3,3,1,1,1,1) )
  model:add( nn.SpatialBatchNormalization(16) )
  model:add( cudnn.ReLU )
  model:add( cudnn.SpatialMaxPooling(3,3,2,2,0,0) )
  model:add( cudnn.SpatialConvolution(16,16,3,3,1,1,1,1) )
  model:add( nn.SpatialBatchNormalization(16) )
  model:add( cudnn.ReLU )
  model:add( cudnn.SpatialMaxPooling(3,3,2,2,0,0) )
  
  -- Initialize
  
  -- 
  
  model:cuda()
  return model
end

return createModel
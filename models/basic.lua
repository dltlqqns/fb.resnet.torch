
local function createModel(opt)
  local model = nn.Sequential()
  
  -- Define network
  model:add( cudnn.SpatialConvolution(3,16,3,3,1,1,1,1) )
  model:add( cudnn.SpatialBatchNormalization(16) )
  model:add( cudnn.ReLU() )
  model:add( cudnn.SpatialMaxPooling(3,3,2,2,1,1) )
  model:add( cudnn.SpatialConvolution(16,16,3,3,1,1,1,1) )
  model:add( cudnn.SpatialBatchNormalization(16) )
  model:add( cudnn.ReLU() )
  model:add( cudnn.SpatialMaxPooling(3,3,2,2,1,1) )
  
  -- Initialize
  local function ConvInit(name)
    for k,v in pairs(model:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0, math.sqrt(2/n))
      if cudnn.version >= 4000 then
        v.bias = nil
        v.gradBias = nil
      else
        v.bias:zero()
      end
    end
  end
  
  local function BNInit(name)
    for k,v in pairs(model:findModules(name)) do
      v.weight:fill(1)
      v.bias:zero()
    end
  end
  
  ConvInit('cudnn.SpatialConvolution')
  --ConvInit('nn.SpatialConvolution')
  BNInit('cudnn.SpatialBatchNormalization')
  --BNInit('nn.SpatialBatchNormalization')
  -- 
  
  model:cuda()
  return model
end

return createModel
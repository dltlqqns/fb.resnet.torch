require 'nngraph'
local nn = require 'nn'
local cunn = require 'cunn'

------------------------------------------------------------------
-- ResNet-18
-- 1 layer = two residual unit(conv + identity)
local layers = { 
  {},
  { filters= 64, kW=3, kH=3, padW=1, padH=1, conv_steps=2 },
  { filters=128, kW=3, kH=3, padW=1, padH=1, conv_steps=2 },
  { filters=256, kW=3, kH=3, padW=1, padH=1, conv_steps=2 },
  { filters=512, kW=3, kH=3, padW=1, padH=1, conv_steps=2 },
}

local anchor_nets = {
  { kW=3, n=256, input=4 },   -- input refers to the 'layer' defined above
  { kW=3, n=256, input=5 },
  { kW=5, n=256, input=5 },
  { kW=7, n=256, input=5 },
}
------------------------------------------------------------------

local function AnchorNetwork(nInputPlane, n, kernelWidth)
  local net = nn.Sequential()
  net:add(cudnn.SpatialConvolution(nInputPlane, n, kernelWidth,kernelWidth, 1,1))
  net:add(cudnn.ReLU())
  net:add(cudnn.SpatialConvolution(n, 3 * (2 + 4), 1, 1))  -- aspect ratios { 1:1, 2:1, 1:2 } x { class, left, top, width, height }
  return net
end

local function createModel(opt)
  -- load pretrained resnet
  local pretrained = torch.load(opt.pretrained)

  -- remove classification layer
  pretrained:remove(#pretrained.modules)
  pretrained:remove(#pretrained.modules)
  pretrained:remove(#pretrained.modules)

  -- convert to nngraph
  local input = nn.Identity()()
  local prev = input
  local conv_outputs = {}
  for l = 1, #pretrained.modules do
    local net = pretrained.modules[l]
    prev = net(prev)
    if pretrained.modules[l].__typename == 'nn.Sequential' or
      pretrained.modules[l].__typename == 'nn.ReLU' or
      pretrained.modules[l].__typename == 'cudnn.ReLU' then
      table.insert(conv_outputs, prev)
    end
  end

  -- debug
  --[[
  for i = 1, #conv_outputs do
    print(conv_outputs[i])
  end
  --]]

  -- add anchor network
  local proposal_outputs = {}
  for i,a in ipairs(anchor_nets) do
    table.insert(proposal_outputs, AnchorNetwork(layers[a.input].filters, a.n, a.kW)(conv_outputs[a.input]))
  end
  table.insert(proposal_outputs, conv_outputs[#conv_outputs])

  -- create nngraph
  local model = nn.gModule({input},proposal_outputs)
  
  -- debug. visualize
  --graph.dot(model.fg,'TEST', 'tmp/fg')

  return model
end

return createModel

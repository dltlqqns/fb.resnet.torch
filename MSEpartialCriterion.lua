local MSEpartialCriterion, parent = torch.class('nn.MSEpartialCriterion', 'nn.Criterion')

function MSEpartialCriterion:__init(p,sizeAverage)
   parent.__init(self)
   if sizeAverage ~= nil then
     self.sizeAverage = sizeAverage
   else
     self.sizeAverage = true
   end
   self.p = p or 0.3
   self.mask = torch.Tensor()
end

local function generateMask(target, p)
   local mask = torch.CudaTensor() -- If you don't want to predetermine CudaTensor or Tensor, consider inline to updateOutput function
   mask:resizeAs(target):zero()
   mask:bernoulli(p)
   mask[target:gt(0)] = 1
   return mask
end

function MSEpartialCriterion:updateOutput(input, target)
   self.mask = generateMask(target, self.p)

   self.output_tensor = self.output_tensor or input.new(1)
   local input_masked = input:clone():cmul(self.mask)
   local target_masked = target:clone():cmul(self.mask)
   input.THNN.MSECriterion_updateOutput(
      input_masked:cdata(),
      target_masked:cdata(),
      self.output_tensor:cdata(),
      self.sizeAverage
   )
   self.output = self.output_tensor[1]
   return self.output
end

function MSEpartialCriterion:updateGradInput(input, target)
   input.THNN.MSECriterion_updateGradInput(
      input:cdata(),
      target:cdata(),
      self.gradInput:cdata(),
      self.sizeAverage
   )
   self.gradInput:cmul(self.mask)
   return self.gradInput
end

return MSEpartialCriterion

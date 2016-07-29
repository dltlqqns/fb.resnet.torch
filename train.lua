--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'
local eval = require 'eval'
local visualize = require 'visualize'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   --local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local lossSum, accSum, distSum = 0.0, 0.0, 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1)
      local loss = self.criterion:forward(self.model.output, self.target)
      local acc = eval.getPerformance(output, sample, self.opt.dataset)

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      optim.sgd(feval, self.params, self.optimState)

      lossSum = lossSum + loss*batchSize
      accSum = accSum + acc*batchSize
      N = N + batchSize

      -- draw heatmap
      if n == 1 then
        local hm = visualize.drawOutput(sample.input[1], output[1])
        --image.display(hm)
        image.save('checkpoints/train_hm.png', hm)
        --local hm_gt = visualize.drawOutput(sample.input[1], sample.target[1])
        --image.display(hm_gt)
      end
      
      --
      print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Loss %1.4f  Acc %1.4f'):format(
         epoch, n, trainSize, timer:time().real, dataTime, loss, acc))

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

   return lossSum / N, accSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local lossSum, accSum = 0.0, 0.0
   local N = 0

   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1) / nCrops
      local loss = self.criterion:forward(self.model.output, self.target)
      local acc = eval.getPerformance(output, sample, self.opt.dataset)

      lossSum = lossSum + loss*batchSize
      accSum = accSum + acc*batchSize
      N = N + batchSize
      
      -- draw heatmap
      if n == 1 then
        local hm = visualize.drawOutput(sample.input[1], output[1])
        --image.display(hm)
        image.save('checkpoints/val_hm.png', hm)
        --local hm_gt = visualize.drawOutput(sample.input[1], sample.target[1])
        --image.display(hm_gt)
      end
      
      --
      print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  Loss %1.4f Acc %1.4f'):format(
         epoch, n, size, timer:time().real, dataTime, loss, acc))

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()

   print((' * Finished epoch # %d     loss: %1.4f  acc: %1.4f\n'):format(
      epoch, lossSum / N, accSum / N))

   return lossSum / N, accSum / N
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or torch.CudaTensor()

   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'imagenet' then
      decay = math.floor((epoch - 1) / 30)
   elseif self.opt.dataset == 'cifar10' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   end
   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer

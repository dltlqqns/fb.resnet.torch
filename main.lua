--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'gnuplot'
require 'hdf5'

local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestAcc = -math.huge
local trainLosses = checkpoint and checkpoint.trainLosses or {}
local trainAccs = checkpoint and checkpoint.trainAccs or {}
local testLosses = checkpoint and checkpoint.testLosses or {}
local testAccs = checkpoint and checkpoint.testAccs or {}
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainLoss, trainAcc = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local testLoss, testAcc = trainer:test(epoch, valLoader)

   -- Save accuracy
   trainLosses[epoch] = trainLoss
   trainAccs[epoch] = trainAcc
   testLosses[epoch] = testLoss
   testAccs[epoch] = testAcc

   local bestModel = false
   if testAcc > bestAcc then
      bestModel = true
      bestAcc = testAcc
      print(' * Best model ', testAcc)
   end
   
  checkpoints.save(epoch, model, trainer.optimState, bestModel, trainLosses, testLosses, trainAccs, testAccs, opt)
  if epoch >= 2 then
    checkpoints.saveplot(trainLosses, testLosses, opt, 'loss')
    checkpoints.saveplot(trainAccs, testAccs, opt, 'acc')
  end
end

print(string.format(' * Finished acc: %6.3f', bestAcc))

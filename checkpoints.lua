--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

local checkpoint = {}

function checkpoint.latest(opt)
   if opt.resume == 'none' then
      return nil
   end

   local latestPath = paths.concat(opt.resume, 'latest.t7')
   if not paths.filep(latestPath) then
      return nil
   end

   print('=> Loading checkpoint ' .. latestPath)
   local latest = torch.load(latestPath)
   local optimState = torch.load(paths.concat(opt.resume, latest.optimFile))
   return latest, optimState
end

function checkpoint.save(epoch, model, optimState, isBestModel, trainLosses, testLosses, trainAccs, testAccs, opt)
   local function saveModel(m)
      local modelFile = 'model_' .. epoch .. '.t7'
      local optimFile = 'optimState_' .. epoch .. '.t7'

      torch.save(paths.concat(opt.save, modelFile), m)
      torch.save(paths.concat(opt.save, optimFile), optimState)
      torch.save(paths.concat(opt.save, 'latest.t7'), {
         epoch = epoch,
         modelFile = modelFile,
         optimFile = optimFile,
		 trainLosses = trainLosses,
		 testLosses = testLosses,
		 trainAccs = trainAccs,
		 testAccs = testAccs,
      })

      if isBestModel then
         torch.save(paths.concat(opt.save, 'model_best.t7'), m)
      end
   end

   -- Remove temporary buffers to reduce checkpoint size
   model:clearState()

   -- Don't save the DataParallelTable for easier loading on other machines
   if torch.type(model) == 'nn.DataParallelTable' then
      saveModel(model:get(1))
   else
      saveModel(model)
   end

   -- Re-use gradInput buffers if the option is set. This is necessary because
   -- of the model:clearState() call clears sharing.
   if opt.shareGradInput then
      local models = require 'models/init'
      models.shareGradInput(model)
   end
end

function checkpoint.saveplot(trainY, testY, opt, id)
  -- linear scale
   local h1 = gnuplot.pdffigure(paths.concat(opt.save,opt.expID .. id .. '.pdf'))
   gnuplot.plot({'train', torch.Tensor(trainY), '-'},{'val', torch.Tensor(testY), '-'})
   gnuplot.grid(true)
   gnuplot.xlabel('Iteration')
   gnuplot.ylabel(id)
   gnuplot.plotflush(h1)
   
   -- log scale
   local h2 = gnuplot.pdffigure(paths.concat(opt.save,opt.expID .. id .. '_logscale.pdf'))
   gnuplot.plot({'train', torch.log(torch.Tensor(trainY)), '-'},{'val', torch.log(torch.Tensor(testY)), '-'})
   gnuplot.grid(true)
   gnuplot.xlabel('Iteration')
   gnuplot.ylabel(id .. ' (log-scale)')
   gnuplot.plotflush(h2)
end

return checkpoint

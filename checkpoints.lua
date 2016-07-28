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

function checkpoint.save(epoch, model, optimState, isBestModel, trainTop1s, testTop1s, opt)
   local function saveModel(m)
      local modelFile = 'model_' .. epoch .. '.t7'
      local optimFile = 'optimState_' .. epoch .. '.t7'

      torch.save(paths.concat(opt.save, modelFile), m)
      torch.save(paths.concat(opt.save, optimFile), optimState)
      torch.save(paths.concat(opt.save, 'latest.t7'), {
         epoch = epoch,
         modelFile = modelFile,
         optimFile = optimFile,
		 trainTop1s = trainTop1s,
		 testTop1s = testTop1s,
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
   local h = gnuplot.pdffigure(paths.concat(opt.save,opt.expID .. id .. '.pdf'))
   gnuplot.plot({'train', torch.Tensor(trainY), '-'},{'val', torch.Tensor(testY), '-'})
   gnuplot.grid(true)
   gnuplot.xlabel('Iteration')
   gnuplot.ylabel('Accuracy')
   gnuplot.plotflush(h)
end

function checkpoint.saveHeatmap(hm)
  local im = drawOutput(torch.Tensor(3,hm:size(1),hm:size(2)),hm)
  return im
end

return checkpoint

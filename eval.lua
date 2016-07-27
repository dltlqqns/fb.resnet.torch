local t = require 'datasets/transforms_ym'
local M = {}

-- Pose estimation function from heatmap to joint positions on the heatmap coord
function M.getPred_hm(hm)
  -- hm: (nSample x nPart x H x W) heatmaps
  -- preds_hm: (nSample x nPart x 2). yx order
  assert(hm:nDimension()==4, 'wrong input format in function getPred')
  local max, idx = torch.max(hm:view(hm:size(1),hm:size(2),hm:size(3)*hm:size(4)),3)
  local preds_hm = torch.repeatTensor(idx,1,1,2):float()
  preds_hm[{{},{},1}]:add(-1):div(hm:size(4)):floor():add(1)
  preds_hm[{{},{},2}]:apply(function(x) return (x-1)%hm:size(4) end)
  return preds_hm
end

-- Return accuracy of parts for one sample
function M.detected_hm(pred_hm, target_hm, thres, dist)
  -- pred_hm: (nPart x 2). yx order. pred_hm = -1 for invisible parts
  -- target_hm : (nPart x 2). yx order. target_hm = -1 for invisible parts
  -- dist: (nPart x 1). dist = -1 for invisible parts
  assert(pred_hm:nDimension()==2 and target_hm:nDimension()==2, 'wrong input format')
  
  local nPart = pred_hm:size(1)
  local iRSHO = 
  local iLHIP = 
  
  -- Get distance
  dist = M.getDistance_hm(pred_hm, target_hm)
  
  -- Get reference distance (rsho to lhip)
  local refdist = torch.norm(target_hm[iRSHO] - target_hm[iLHIP])
  -- Get accuracy
  -- take care of invisible parts!!
  local acc = torch.zeros(nPart)
  
  return acc
end

-- Return distance from GT of parts for one sample
function M.getDistance_hm(pred_hm, target_hm)
  -- pred_hm: (nPart x 2). yx order
  -- target_hm : (nPart x 2). yx order
  -- target_hm = -1 for invisible parts. corresponding distance = -1
  -- dist: (nPart x 1)
  assert(pred_hm:nDimension()==2 and target_hm:nDimension()==2, 'wrong input format')
  local nPart = pred_hm:size(1)
  local invisible = target_hm:lt(0):sum(2):squeeze()
  local dist = -torch.ones(nPart, 2)
  for iPart = 1, nPart do
    if not invisible[iPart] then
      dist[iPart] = torch.norm(pred_hm[iPart] - target_hm[iPart])
    end
  end
  print(dist)
  
  return dist
end

-- Return average accuracy and distance on the heatmap coord over all samples
function M.getPerformance(output, sample)
  assert(output:nDimension()==4, 'wrong input format in function getPerformance')
  local nSample = output:size(1)
  local nPart = output:size(2)
  local thres = 0.5 -- pckh
  
  -- Get prediction on the heatmap coord
  local preds_hm = M.getPred_hm(output)
  -- Get GT on the heatmap coord
  local targets_hm = sample.parts_hm
  
  -- Get distance and accuracy
  local dists = torch.zeros(nSample, nPart)
  local dets = torch.zeros(nSample, nPart)
  for iSample = 1, nSample do
    dets[iSample], dists[iSample] = M.detected_hm(preds_hm[iSample], targets_hm[iSample], thres)
  end
  
  -- Average over both samples and parts
  return dets:float():mean(), dists:float():mean()
end

function M.drawPCK(preds_orig, target_orig)
end

return M

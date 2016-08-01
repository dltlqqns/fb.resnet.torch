local M = {}

-- Pose estimation function from heatmap to joint positions on the heatmap coord
function M.getPred_hm(hm, type)
  -- hm: (nSample x nPart x H x W) heatmaps
  -- pred_hm: (nSample x nPart x 2). yx order
  -- visible: (nSample x nPart)
  assert(hm:nDimension()==4, 'wrong input format in function getPred')
  local preds_hm
  local pred_visible
  if type=='max' then
    local max, idx = torch.max(hm:view(hm:size(1),hm:size(2),hm:size(3)*hm:size(4)),3)
    preds_hm = torch.repeatTensor(idx,1,1,2):float()
    preds_hm[{{},{},1}]:add(-1):div(hm:size(4)):floor():add(1)
    preds_hm[{{},{},2}]:apply(function(x) return (x-1)%hm:size(4)+1 end)
    pred_visible = torch.ByteTensor(hm:size(1), hm:size(2)):fill(1)
  end
  return preds_hm, pred_visible
end

function M.getAccuracy(normalized_dists, pred_invisible, invisible, thres, type)
  local acc
  invisible[invisible:gt(1)]:fill(1)
  if type=='normal' then
    local visible = 1 - invisible
    local dets = normalized_dists:lt(thres):cmul(visible)
    local nVisibles = visible:float():sum(1)
    local valid = 1-nVisibles:eq(0) -- to address dividing by 0
    acc = dets:float():sum(1)[valid]:cdiv(nVisibles[valid]:float()):mean()
  elseif type== 'occlusion-aware' then
    error('under construction')
  else
    error('wrong type')
  end
  return acc
end

-- Return average accuracy and distance on the heatmap coord over all samples
function M.getPerformance(output, sample, dataset)
  assert(output:nDimension()==4, 'wrong input format in function getPerformance')
  local nSample = output:size(1)
  local nPart = output:size(2)
  local thres = 0.5 -- pckh
  local iRSHO, iLHIP
  if dataset=='mpii' then
    iRSHO, iLHIP = 3,14 -- ??
  else
    error('unsupported dataset')
    iRSHO, iLHIP = 1,2
  end
  
  -- Get prediction on the heatmap coord
  local preds_hm, pred_invisible = M.getPred_hm(output, 'max')
  -- Get GT on the heatmap coord
  local targets_hm = sample.parts_hm:float()
  local invisible = targets_hm:eq(-1):sum(3)
  invisible[invisible:gt(1)]:fill(1)
  
  -- Get accuracy
  local dists = (preds_hm - targets_hm):pow(2):sum(3):sqrt()
  local dists_ref = (targets_hm[{{},{iRSHO},{}}] - targets_hm[{{},{iLHIP},{}}]):pow(2):sum(3):sqrt()
  dists_ref[dists_ref:eq(0)] = 1 -- prevent to divide by 0
  local normalized_dists = dists:cdiv(dists_ref:repeatTensor(1,dists:size(2),1))
  local acc = M.getAccuracy(dists, pred_invisible, invisible, thres, 'normal')
  
  -- Average over both samples and parts
  return acc
end

function M.drawPCK(preds_orig, target_orig)
end

return M

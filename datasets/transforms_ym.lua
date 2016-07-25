require 'image'

local M = {}

function M.Compose(transforms)
  return function(input)
    for _, transform in ipairs(transforms) do
      input = transform(input)
    end
  end
end

function M.ColorNormalize(meanstd)
  return function(input)
    local output = input:clone()
    for i = 1, input:size(1) do
      output[i]:add(-meanstd.mean[i])
      output[i]:div(meanstd.std[i])
    end
    return output
  end
end


return M
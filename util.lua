--[[
function slice_oper(t, dim, type)
  -- output dimension of coord dim is 1
  -- oper: all | any
  
  local oper
  if type == 'all' then
    oper = function(x) return all(x) end
  elseif type == 'any' then
    oper = function(x) return any(x) end
  end
  
  local output
  if t.dim() == 2 and dim == 1 then -- any(t,1) of matlab
    output = t:new(1,t:size(2))
    for i = 1, t:size(2) do
      output[1][i] = oper(t:select(2,i))
    end
  elseif t.dim() == 2 and dim == 2 then -- any(t,2) of matlab
    output = t:new(t:size(1),1)
    for i = 1, t:size(1) do
      output[i][1] = oper(t:select(1,i))
    end
  elseif t.dim == 3 and dim == 1 then
    output = t:new(1,t:size(2),t:size(3))
    for i = 1, t:size(2) do
      for j = 1, t:size(3) do
        output[1][i][j] = oper(t:select(3,j):select(2,i))
      end
    end
  elseif t.dim == 3 and dim == 2 then
    output = t:new(t:size(1),2,t:size(3))
    for i = 1, t:size(1) do
      for j = 1, t:size(3) do
        output[i][1][j] = oper(t:select(3,j):select(1,i))
      end
    end
  elseif t.dim == 3 and dim == 3 then
    output = t:new(t:size(1),t:size(2),1)
    for i = 1, t:size(1) do
      for j = 1, t:size(2) do
        output[i][j][1] = oper(t:select(2,j):select(1,i))
      end
    end
  end
  
  return output
end

function boolean2float(input)
  assert(torch.typename(input)=='torch.ByteTensor')
  local output = torch.zeros(input:size())[input]:fill(1)
  return output
end
--]]
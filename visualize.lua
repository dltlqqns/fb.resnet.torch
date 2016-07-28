local M = {}

-- copied from pose-hg-demo util.lua
function M.colorHM(x)
    -- Converts a one-channel grayscale image to a color heatmap image
    local function gauss(x,a,b,c)
        return torch.exp(-torch.pow(torch.add(x,-b),2):div(2*c*c)):mul(a)
    end
    local cl = torch.zeros(3,x:size(1),x:size(2))
    cl[1] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
    cl[2] = gauss(x,1,.5,.3)
    cl[3] = gauss(x,1,.2,.3)
    cl[cl:gt(1)] = 1
    return cl
end

-- copied from pose-hg-demo img.lua
function M.compileImages(imgs, nrows, ncols, res)
    -- Assumes the input images are all square/the same resolution
    local totalImg = torch.zeros(3,nrows*res,ncols*res)
    for i = 1,#imgs do
        local r = torch.floor((i-1)/ncols) + 1
        local c = ((i - 1) % ncols) + 1
        totalImg:sub(1,3,(r-1)*res+1,r*res,(c-1)*res+1,c*res):copy(imgs[i])
    end
    return totalImg
end

-- copied and modified from pose-hg-demo util.lua
function M.drawOutput(input, hms)
	--assert(hms:size(1)==16,'wrong input in function drawOutput')

    local colorHms = {}
    local inp64 = image.scale(input,64):mul(.3)
    for i = 1,hms:size(1) do 
        colorHms[i] = M.colorHM(hms[i])
        colorHms[i]:mul(.7):add(inp64)
    end
    local totalHm = M.compileImages(colorHms, 4, 4, 64)
    local im = image.scale(totalHm,756)
    return im
end

return M
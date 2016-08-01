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
    local sz = hms:size(3)
    local inp = image.scale(input,sz):mul(.3)
    for i = 1,hms:size(1) do 
        colorHms[i] = M.colorHM(hms[i])
        colorHms[i]:mul(.7):add(inp)
    end
    local totalHm = M.compileImages(colorHms, 4, 4, sz)
    local im = image.scale(totalHm,756)
    return im
end

-- copied from pose-hg-demo/img.lua
-- modified from xy order to yx order
function M.drawLine(img,pt1,pt2,width,color)
    -- I'm sure there's a line drawing function somewhere in Torch,
    -- but since I couldn't find it here's my basic implementation
    local color = color or {1,1,1}
    local m = torch.dist(pt1,pt2)
    local dx = (pt2[2] - pt1[2])/m
    local dy = (pt2[1] - pt1[1])/m
    for j = 1,width do
        local start_pt1 = torch.Tensor({pt1[1] + (-width/2 + j-1)*dx, pt1[2] - (-width/2 + j-1)*dy})
        start_pt1:ceil()
        for i = 1,torch.ceil(m) do
            local x_idx = torch.ceil(start_pt1[2]+dx*i)
            local y_idx = torch.ceil(start_pt1[1]+dy*i)
            if y_idx - 1 > 0 and x_idx -1 > 0 and y_idx < img:size(2) and x_idx < img:size(3) then
                img:sub(1,1,y_idx-1,y_idx,x_idx-1,x_idx):fill(color[1])
                img:sub(2,2,y_idx-1,y_idx,x_idx-1,x_idx):fill(color[2])
                img:sub(3,3,y_idx-1,y_idx,x_idx-1,x_idx):fill(color[3])
            end
        end 
    end
    img[img:gt(1)] = 1

    return img
end

-- copied from pose-hg-demo/util.lua
function M.drawSkeleton(input, hms, coords, visible)

    local im = input:clone()

    local pairRef = {
        {1,2},      {2,3},      {3,7},
        {4,5},      {4,7},      {5,6},
        {7,9},      {9,10},
        {14,9},     {11,12},    {12,13},
        {13,9},     {14,15},    {15,16}
    }
    
    local partNames = {'RAnk','RKne','RHip','LHip','LKne','LAnk',
                       'Pelv','Thrx','Neck','Head',
                       'RWri','RElb','RSho','LSho','LElb','LWri'}
    local partColor = {1,1,1,2,2,2,0,0,0,0,3,3,3,4,4,4}

    local actThresh = 0.002

    -- Loop through adjacent joint pairings
    for i = 1,#pairRef do
        --if hms[pairRef[i][1]]:mean() > actThresh and hms[pairRef[i][2]]:mean() > actThresh then
            -- Set appropriate line color
            local color
            if partColor[pairRef[i][1]] == 1 then color = {0,.3,1}
            elseif partColor[pairRef[i][1]] == 2 then color = {1,.3,0}
            elseif partColor[pairRef[i][1]] == 3 then color = {0,0,1}
            elseif partColor[pairRef[i][1]] == 4 then color = {1,0,0}
            else color = {.7,0,.7} end

            -- Draw line
            if(visible[pairRef[i][1]]==1 and visible[pairRef[i][2]]==1) then
              im = M.drawLine(im, coords[pairRef[i][1]], coords[pairRef[i][2]], 4, color, 0)
            end
        --end
    end

    return im
end

return M

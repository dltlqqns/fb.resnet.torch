local datasets = require('datasets/mpii_multi-gen')
local opts = require('opts')

opt = opts.parse({})

local dataset = datasets.exec(opt, cacheFile)
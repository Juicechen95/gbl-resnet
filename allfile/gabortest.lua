--require('mobdebug').start()
require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'math'
require 'image'
require 'xlua'
-- require 'SpatialConvolutionPoly'

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


-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)


for n, sample in trainLoader:run() do 
	if n <4000 then
		--image.display{image = sample.input,zoom = 5,legend = n}
		print(sample.target)
		else 
		break
		end
	
	end
--[[
images = {}
for i,file in ipairs(files) do
   -- load each image
   table.insert(images, image.load(file))
end

print('Loaded images:')
print(images)

-- Display a of few them
for i = 1,math.min(#files,10) do
   image.display{image=images[i], legend=files[i]}
end
-- The trainer handles the training loop and evaluation on validation set
--local trainer = Trainer(model, criterion, opt, optimState)
--local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)
--]]

-- main.lua
--require('mobdebug').start()
require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'math'
require 'image'
-- require 'SpatialConvolutionPoly'

local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train_cifar'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)
local optimState = checkpoint and torch.load(checkpoint.optimFile) or nil

-- Create model
local model, criterion = models.setup(opt, checkpoint)
model:cuda()
print(model)
-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)


--visualizing filters
if opt.visual then
	print('visual')
	filter1 =  model:get(1).weight
	img0 = image.toDisplayTensor{input = filter1,  padding = 2, scaleeach = 800, nrow = 18}
	--img1 = image.scale(img0,400,400)
	path1 = paths.concat(opt.save,'filter1.png')
	image.save(path1, img0)
end
	

-- The trainer handles the training loop and evaluation on validation set

	local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, trainLoader)--valLoader)
	print('test only now!!!!!!')
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge

-- log results to files
accLogger = optim.Logger(paths.concat(opt.save, 'accuracy.log'))
errLogger = optim.Logger(paths.concat(opt.save, 'error.log'   ))

for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local testTop1, testTop5, testLoss = trainer:test(epoch, valLoader)

   local bestModel = false
   if testTop1 < bestTop1 then
      bestModel = true
      bestTop1 = testTop1
      bestTop5 = testTop5
      print(' * Best model ', testTop1, testTop5)
   end
   --checkpoints.save(opt, epoch, model, trainer.optimState, bestModel)
   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)

   -- update logger
   accLogger:add{['% train accuracy'] = trainTop1, ['% test accuracy'] = testTop1}
   errLogger:add{['% train error']    = trainLoss, ['% test error']    = testLoss}

   -- plot logger
   accLogger:style{['% train accuracy'] = '-', ['% test accuracy'] = '-'}
   errLogger:style{['% train error']    = '-', ['% test error']    = '-'}
   accLogger:plot()
   errLogger:plot()
end

print(string.format(' * Finished top1: %6.3f  top5: %6.3f', bestTop1, bestTop5))

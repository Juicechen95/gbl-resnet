-- resnet-lgbp.lua

local nn = require 'nn'
require 'math'
require 'cunn'
require 'cutorch'
require 'RandomBinaryConvolution'
require 'GaborInitializedConv'
require'Gaborconv2'
require'Gbconv_lr'
require'BCM'


local bcm = cudnn.BCM
local RandomBinaryConvolution = cudnn.RandomBinaryConvolution
local Convolution = cudnn.SpatialConvolution
local Gbconv = cudnn.Gaborconv2 --return square sum of real and image
--local Gbconv2 = cudnn.Gaborconv2 -- return half real and half image gabor kernels
local Gbconvlr = cudnn.Gbconv_lr
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Sigmoid = cudnn.Sigmoid
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
	
   local function gaborblock(nChIn,nChOut,gbsz)
	  local r = nn.Sequential() --real part square of gabor
		r:add(nn.ConcatTable()
			:add(Gbconv(nChIn,nChOut,gbsz,gbsz,1,1,(gbsz-1)/2,(gbsz-1)/2,1))
			:add(Gbconv(nChIn,nChOut,gbsz,gbsz,1,1,(gbsz-1)/2,(gbsz-1)/2,1)))
		r:add(nn.CMulTable(true))
	  local i = nn.Sequential() --real part square of gabor
		i:add(nn.ConcatTable()
			:add(Gbconv(nChIn,nChOut,gbsz,gbsz,1,1,(gbsz-1)/2,(gbsz-1)/2,2))
			:add(Gbconv(nChIn,nChOut,gbsz,gbsz,1,1,(gbsz-1)/2,(gbsz-1)/2,2)))
		i:add(nn.CMulTable(true))
		
      return nn.Sequential()
		--:add(SBatchNorm(nChIn))  ??????
		:add(nn.ConcatTable()
			:add(r)
			:add(i))
		:add(nn.CAddTable(true))
		--:add(ReLU(true))
   end
	
  local function basicblock(nChIn,nChOut,sz)
      local s = nn.Sequential()
      local shareConv = RandomBinaryConvolution(nChIn,nChOut,sz,sz,1,1,(sz-1)/2,(sz-1)/2)
      s:add(SBatchNorm(nChIn))
--if opt.gbconv == 2 then
--		s:add(Gbconv2(nChIn,nChOut,gbsz,gbsz,1,1,(gbsz-1)/2,(gbsz-1)/2,1))
      -- s:add(ReLU())
      s:add(shareConv)
      s:add(ReLU())
      s:add(Convolution(nChOut,nChIn,1,1))
      local identity = nn.Identity()
     local output = nn.Sequential():add(nn.ConcatTable():add(s):add(identity)):add(nn.CAddTable(true))
      --local output = s
		return output
   end
	
   	local sz = opt.convSize
	local gbsz = opt.gbsz --gabor convsize
	local gbst = opt.gbst ---gabor stride
  	local nInputPlane = opt.nInputPlane
 	local nChan =opt.numChannels  --number of intermediate channels  
    local ngbout = opt.ngbout--gabor kernels
	local nChOut1 = opt.nbcout--bcm output channels
	local nChOut2 = opt.numWeights --number of fixed binary kernels
    local mode = opt.mode ---theta,gamma,lambda
if opt.gblr == 0 then
	 Gbconv2 = Gbconv
else 
	 Gbconv2 = Gbconvlr
end
   -- define model to train
   model = nn.Sequential()
	
	if opt.gbconv ==1 then
--lgbppattern
   model:add(gaborblock(nInputPlane,ngbout,gbsz))
    --model:add(SBatchNorm(ngbout))
    model:add(RandomBinaryConvolution(ngbout,nChOut2,sz,sz,1,1,(sz-1)/2,(sz-1)/2))
    model:add(ReLU())
    model:add(Convolution(nChOut2,nChan,1,1))
	
	elseif opt.gbconv ==2 then
	 --gabor patternGbconv2(nInputPlane,ngbout, bsz,gbsz,1,1,(gbsz-1)/2,(gbsz-1)/2))
   model:add(Gbconv2(nInputPlane,ngbout,gbsz,gbsz,1,1,(gbsz-1)/2,(gbsz-1)/2,1))
	model:add(Convolution(ngbout,nChan,3,3))
        elseif opt.gbconv == 3 then
model:add(Gbconv2(nInputPlane,ngbout,gbsz,gbsz,gbst,gbst,(gbsz-1)/2,(gbsz-1)/2,mode, opt.gblr))
	model:add(bcm(ngbout,nChan,sz,sz,1,1,(sz-1)/2,(sz-1)/2))
	end
	
   model:add(SBatchNorm(nChan))
   model:add(ReLU(true))

   for stages = 1,opt.depth do
      model:add(basicblock(nChan,nChOut2,sz))
      -- model:add(Max(3,3,2,2,1,1))
   end



   -- stage 3 : standard 2-layer neural network
if gbst == 2 then
	model:add(Avg(3,3,3,3))--cifar10 retry add(Avg(4,4,4,4))--svhn :add(Avg(2,2,2,2))--gbst=2
	model:add(nn.Reshape(nChan*5*5))--cifar10 retry add(nn.Reshape(nChan*4*4))--svhn :add(nn.Reshape(nChan*8*8))
	model:add(nn.Dropout(0.5))
   	model:add(nn.Linear(nChan*5*5, opt.full))--cifar10 retry add(nn.Linear(nChan*4*4, opt.full))--svhn :add(nn.Linear(nChan*8*8, opt.full))--opt.view, opt.full))
elseif gbst == 3 then
	model:add(Avg(2,2,2,2))--gbst=2
	model:add(nn.Reshape(nChan*5*5))
	model:add(nn.Dropout(0.5))
   	model:add(nn.Linear(nChan*5*5, opt.full))
elseif gbst == 1 then
   model:add(Avg(5,5,5,5))
   model:add(nn.Reshape(nChan*opt.view))--view = 6*6
   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(nChan*opt.view, opt.full))
end
   model:add(cudnn.ReLU())
   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(opt.full,opt.nClasses))
   model:cuda()

   return model
end

return createModel

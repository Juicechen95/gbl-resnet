-- GaborInitialized_conv_layer.lua
--fixed gabor
require'cutorch'
local THNN = require 'nn.THNN'
local GaborInitializedConv, parent = torch.class('cudnn.GaborInitializedConv', 'cudnn.SpatialConvolution')

function GaborInitializedConv:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, ri)
  self.ri = ri
   parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   self:reset()
end

function GaborInitializedConv:reset()
	--local numElements = self.nInputPlane*self.nOutputPlane*self.kW*self.kH
	self.weight = torch.CudaTensor(self.nOutputPlane,self.nInputPlane,self.kW,self.kH)
		for j = 1, self.nOutputPlane do -- for RGB, filters are the same
			local theta =  0+(180/self.nOutputPlane)*j
			local tmp = gabor2(self.kW, 3,theta, 0,1.68,0.5, self.ri)
			for i = 1,self.nInputPlane do 
				self.weight[j][i] = tmp:clone()--gabor2(self.kW, 3,theta, 0,1.68,0.5, self.ri):clone()
			    	--[[self.weight[j][2] = tmp:clone()--self.weight[j][1]
			    	self.weight[j][3] = tmp:clone()--self.weight[j][1]--]]
			end
			
		end
	
	self.bias = nil
	self.gradBias = nil	
	--self.gradWeight = torch.CudaTensor(self.nOutputPlane, self.nInputPlane, self.kH, self.kW):fill(0) 	
end

function gabor2(Sx, lambda, theta, shi, sigma,gamma, R)
	local pi = 3.14
	local Sy = Sx
	sigma = 0.56*lambda
		Gabor = torch.CudaTensor(Sx,Sy)
		for x = 1,Sx do
			for y = 1,Sy do
		xPrime =  (x-Sx/2-1)*math.cos(theta) + (y-Sy/2-1)*math.sin(theta)	--equation 1
		yPrime = -(x-Sx/2-1)*math.sin(theta)  + (y-Sy/2-1)*math.cos(theta)	--equation 2
					if R == 1 then --r=1 means real part of gabor
						Gabor[x][y] = math.exp(-1/(sigma*3)*((xPrime^2)+(yPrime^2 * gamma^2 )))*math.cos(2*pi*xPrime/lambda  + shi)	-- equation 3
						else
						Gabor[x][y] = math.exp(-1/(sigma*3)*((xPrime^2)+(yPrime^2 * gamma^2 )))*math.sin(2*pi*xPrime/lambda  + shi)	-- equation 3	 
					end
			end
		end
--[[print('gabor')
print(Gabor)--]]	
		return(Gabor)
end

function GaborInitializedConv:accGradParameters(input, gradOutput, scale)
end

function GaborInitializedConv:updateParameters(learningRate)
end

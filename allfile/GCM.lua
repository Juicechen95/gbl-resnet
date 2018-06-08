--GCM_layer.lua   half real and half image
--fixed gabor
require'cutorch'
local THNN = require 'nn.THNN'
local GCM, parent = torch.class('cudnn.GCM', 'cudnn.SpatialConvolution')

function GCM:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, mode, theta, ga)
  --self.ri = ri
   parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   --self:reset()
self.mode = mode
self.theta = theta
self.ga = ga
print('self.theta')
print(self.theta)
print('self.ga')
print(self.ga)
print('begin reset')
print('self.mode')
print(self.mode)
self:reset()
end

function GCM:reset()
		print('in reset')
	self.weight = torch.CudaTensor(self.nOutputPlane,self.nInputPlane,self.kW,self.kH)

	if self.mode == 't1' then
		print('in t1')
			for i = 1, self.nInputPlane do 
			   self.weight[1][i] = gabor2(self.kW, 3,self.theta, 0,1.68,self.ga, 1)
			   self.weight[2][i] = gabor2(self.kW, 3,self.theta, 0,1.68,self.ga, 0)
  			end
 		 
	elseif self.mode == '7l2' then--gbsz7 l=2
		print('in t1')
		for j = 1, self.nOutputPlane,2 do -- for RGB, filters are the same
			local theta =  (180/self.nOutputPlane)*(j-1)
			print(j)
			print(theta)
			for i = 1,self.nInputPlane do 
			   self.weight[j][i] = gabor2(self.kW, 2,theta, 0,1.68,0.5, 1)
			   self.weight[j+1][i] = gabor2(self.kW, 2,theta, 0,1.68,0.5, 0)
  			end
 		 end
	elseif self.mode == '5l2' then--gbsz5 l=2
		print('in t1')
		local n = 0
		for j = 1, self.nOutputPlane,4 do -- for RGB, filters are the same
			local theta =  (360/self.nOutputPlane)*n
			print(j)
			print(theta)
			for i = 1,self.nInputPlane do 
			   self.weight[j][i] = gabor2(self.kW, 2,theta, 0,1.68,0.5, 1)
			   self.weight[j+1][i] = gabor2(self.kW, 2,theta, 0,1.68,0.5, 0)
			   self.weight[j+2][i] = gabor2(self.kW, 2,theta, 0,1.68,1, 1)
			   self.weight[j+3][i] = gabor2(self.kW, 2,theta, 0,1.68,1, 0)
  			end
			n = n+1
 		 end

	elseif self.mode == 't2' then
		print('in t2')
		local n = 0
		for j = 1, self.nOutputPlane,2 do -- for RGB, filters are the same
			local theta =  (180/(self.nOutputPlane/2))*n
			print('j')
			print(j)
			print(theta)
			for i = 1,self.nInputPlane do 
			   self.weight[j][i] = gabor2(self.kW, 3,theta, 0,1.68,0.5, 1)
			   self.weight[j+1][i] = gabor2(self.kW, 3,theta, 0,1.68,0.5, 0)	
  			end
			n = n+1
 		 end
	elseif self.mode == 't3' then
		print('in t3')
		local n = 0
		for j = 1, self.nOutputPlane,2 do -- for RGB, filters are the same
			local theta =  (360/(self.nOutputPlane/2))*n
			print('j')
			print(j)
			print(theta)
			for i = 1,self.nInputPlane do 
			   self.weight[j][i] = gabor2(self.kW, 3,theta, 0,1.68,0.5, 1)
			   self.weight[j+1][i] = gabor2(self.kW, 3,theta, 0,1.68,0.5, 0)	
  			end
			n = n+1
 		 end
	elseif self.mode == 'ti' then
		print('in t3')
		local n = 0
		for j = 1, self.nOutputPlane,2 do -- for RGB, filters are the same
			local theta =  (360/(self.nOutputPlane/2))*n
			print('j')
			print(j)
			print(theta)
			for i = 1,self.nInputPlane do 
			   self.weight[j][i] = gabor2(self.kW, 3,theta+i, 0,1.68,0.5, 1)
			   self.weight[j+1][i] = gabor2(self.kW, 3,theta+i, 0,1.68,0.5, 0)	
  			end
			n = n+1
 		 end
	elseif self.mode == 'g' then
		local n = 0
		for j = 1, self.nOutputPlane,2 do -- for RGB, filters are the same
			local ga = (1/(self.nOutputPlane/2))*n
			print('j')
			print(j)			
			print('gamma')
			print(ga)
			for i = 1,self.nInputPlane do 
			   self.weight[j][i] = gabor2(self.kW, 3,30, 0,1.68,ga, 1)
			   self.weight[j+1][i] = gabor2(self.kW, 3,30, 0,1.68,ga, 0)	
  			end
			n = n+1
 		 end
	elseif self.mode == 'l' then
		local n = 0
		for j = 1, self.nOutputPlane,2 do -- for RGB, filters are the same
			local la = 2+(3/(self.nOutputPlane/2))*n
			print('j')
			print(j)			
			print('lambda')
			print(la)
			for i = 1,self.nInputPlane do 
			   self.weight[j][i] = gabor2(self.kW, la,30, 0,1.68,0.5, 1)
			   self.weight[j+1][i] = gabor2(self.kW, la,30, 0,1.68,0.5, 0)	
  			end
			n = n+1
 		 end
	elseif self.mode == 'tg1' then
		local n = 0
		for j = 1, self.nOutputPlane,4 do -- for RGB, filters are the same
			local theta = (180/(self.nOutputPlane/4))*n
			print('j')
			print(j)
			print(theta)
			for i = 1,self.nInputPlane do 
			   self.weight[j][i] = gabor2(self.kW, 3,theta, 0,1.68,0.5, 1)
			   self.weight[j+1][i] = gabor2(self.kW, 3,theta, 0,1.68,0.5, 0)
			   self.weight[j+2][i] = gabor2(self.kW, 3,theta, 0,1.68,1, 1)
			   self.weight[j+3][i] = gabor2(self.kW, 3,theta, 0,1.68,1, 0)	
  			end
			n = n+1
 		 end
	elseif self.mode == 'tg2' then
		local n = 0
		for j = 1, self.nOutputPlane,4 do -- for RGB, filters are the same
			local theta = (180/(self.nOutputPlane/4))*n
			local ga = (1/(self.nOutputPlane/4))*n
			print('j')
			print(j)
			print(theta)
			for i = 1,self.nInputPlane do 
			   self.weight[j][i] = gabor2(self.kW, 3,theta, 0,1.68,ga, 1)
			   self.weight[j+1][i] = gabor2(self.kW, 3,theta, 0,1.68,ga, 0)
			   self.weight[j+2][i] = gabor2(self.kW, 3,theta, 0,1.68,ga, 1)
			   self.weight[j+3][i] = gabor2(self.kW, 3,theta, 0,1.68,ga, 0)	
  			end
			n = n+1
 		 end
	elseif self.mode == 'tl1' then
		local n = 0
		for j = 1, self.nOutputPlane,4 do -- for RGB, filters are the same
			local theta = (180/(self.nOutputPlane/4))*n
			print('j')
			print(j)
			print(theta)
			for i = 1,self.nInputPlane do 
			   self.weight[j][i] = gabor2(self.kW, 2,theta, 0,1.68,0.5, 1)
			   self.weight[j+1][i] = gabor2(self.kW, 2,theta, 0,1.68,0.5, 0)
			   self.weight[j+2][i] = gabor2(self.kW, 5,theta, 0,1.68,0.5, 1)
			   self.weight[j+3][i] = gabor2(self.kW, 5,theta, 0,1.68,0.5, 0)	
  			end
			n = n+1
 		 end
	elseif self.mode == 'tl2' then
		local n = 0
		for j = 1, self.nOutputPlane,4 do -- for RGB, filters are the same
			local theta = (180/(self.nOutputPlane/4))*n
			local la = 2+(3/(self.nOutputPlane/4))*n
			print('j')
			print(j)
			print(theta)
			for i = 1,self.nInputPlane do 
			   self.weight[j][i] = gabor2(self.kW, la,theta, 0,1.68,0.5, 1)
			   self.weight[j+1][i] = gabor2(self.kW, la,theta, 0,1.68,0.5, 0)
			   self.weight[j+2][i] = gabor2(self.kW, la,theta, 0,1.68,0.5, 1)
			   self.weight[j+3][i] = gabor2(self.kW, la,theta, 0,1.68,0.5, 0)	
  			end
			n = n+1
 		 end
	elseif self.mode == 'lg' then
		local n = 0
		for j = 1, self.nOutputPlane,4 do -- for RGB, filters are the same
			local la = 2+(3/(self.nOutputPlane/4))*n
			local ga = (1/(self.nOutputPlane/4))*n
			print('j')
			print(j)
			print(theta)
			for i = 1,self.nInputPlane do 
			   self.weight[j][i] = gabor2(self.kW, la,30, 0,1.68,ga, 1)
			   self.weight[j+1][i] = gabor2(self.kW, la,30, 0,1.68,ga, 0)
			   self.weight[j+2][i] = gabor2(self.kW, la,30, 0,1.68,ga, 1)
			   self.weight[j+3][i] = gabor2(self.kW, la,30, 0,1.68,ga, 0)	
			end
			n=n+1
 		 end
	elseif self.mode == 'tlg' then
		local n = 0
		for j = 1, self.nOutputPlane,4 do -- for RGB, filters are the same
			local la = 2+(3/(self.nOutputPlane/4))*n
			local ga = (1/(self.nOutputPlane/4))*n
			local theta = (180/(self.nOutputPlane/4))*n
			print('j')
			print(j)
			print(theta)
			for i = 1,self.nInputPlane do 
			   self.weight[j][i] = gabor2(self.kW, la,theta, 0,1.68,ga, 1)
			   self.weight[j+1][i] = gabor2(self.kW, la,theta, 0,1.68,ga, 0)
			   self.weight[j+2][i] = gabor2(self.kW, la,theta, 0,1.68,ga, 1)
			   self.weight[j+3][i] = gabor2(self.kW, la,theta, 0,1.68,ga, 0)	
  			end
			n = n+1
 		 end
	elseif self.mode == 'tig' then ----theta changing with rgb
		local n = 0
		for j = 1, self.nOutputPlane,2 do -- for RGB, filters are the same
			local la = 2+(3/(self.nOutputPlane/4))*n
			local ga = (1/(self.nOutputPlane/4))*n
			local theta = (180/(self.nOutputPlane/4))*n
			print('j')
			print(j)
			print(theta)
			for i = 1,self.nInputPlane do 
			   self.weight[j][i] = gabor2(self.kW, 3,theta+i, 0,1.68,0.5, 1)
			   self.weight[j+1][i] = gabor2(self.kW, 3,theta+i, 0,1.68,0.5, 0)
			   self.weight[j+2][i] = gabor2(self.kW, 3,theta+i, 0,1.68,1, 1)
			   self.weight[j+3][i] = gabor2(self.kW, 3,theta+i, 0,1.68,1, 0)	
  			end
			n = n+1
 		 end
	elseif self.mode == 'tig2' then ----theta changing with rgb
		local n = 0
		for j = 1, self.nOutputPlane,2 do -- for RGB, filters are the same
			--local la = 2+(3/(self.nOutputPlane/4))*n
			--local ga = (1/(self.nOutputPlane/4))*n
			local theta = (360/(self.nOutputPlane/4))*(j-1)
			print('j')
			print(j)
			print(theta)
			for i = 1,self.nInputPlane do 
			   self.weight[j][i] = gabor2(self.kW, 10,theta+i, 0,1.68,0.5, 1)
			   self.weight[j+1][i] = gabor2(self.kW, 10,theta+i, 0,1.68,0.5, 0)
			   self.weight[j+2][i] = gabor2(self.kW, 10,theta+i, 0,1.68,1, 1)
			   self.weight[j+3][i] = gabor2(self.kW, 10,theta+i, 0,1.68,1, 0)	
  			end
			n = n+1
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
		return(Gabor)
end

	function GCM:accGradParameters(input, gradOutput, scale)
	end

	function GCM:updateParameters(learningRate)
	end



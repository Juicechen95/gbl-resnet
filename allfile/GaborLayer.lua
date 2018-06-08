-- Clement's old gabor function

require 'torch'
require 'image'
-- require 'lab'
require 'xlua'

-- For the similar Gabor code go to 
-- http://www.mathworks.com/matlabcentral/fileexchange/23253-gabor-filter/content/Gabor%20Filter/gabor_fn.m


-- Size should be odd number 
-- angle (in rad)
-- elipse_ratio = aspect ratio(0.5)

-- test:
--image.display{image=gabor(9,0.5,45,1,0.5), zoom =4}

function gabor1(size, sigma, angle, period, ellipse_ratio)--9,0.5,45,1,0.5
      -- init matrix
      local data = torch.zeros(size,size)

      -- image -> pixel
      period = period * size
      sigma = sigma * size

      -- set params
      local halfsize = math.floor(size/2)
      local sigma_x = sigma
      local sigma_y = sigma/ellipse_ratio

      for y=-halfsize,halfsize do
         for x=-halfsize,halfsize do
				
            x_angle = x*math.cos(angle) + y*math.sin(angle)
            y_angle = -x*math.sin(angle) + y*math.cos(angle)
            data[x+halfsize+1][y+halfsize+1] 
               = math.exp(-0.5*(x_angle^2/sigma_x^2 + y_angle^2/sigma_y^2))
                math.cos(2*math.pi*x_angle/period)
         end
      end
      
      -- return new tensor
      return data
   end


function gabor2(Sx, lambda, theta, shi, sigma, gamma)
local pi = 3.14
local Sy = Sx
sigma = 0.56*lambda
	Gabor = torch.Tensor(Sx,Sy)
	for x = 1,Sx do
		for y = 1,Sy do
	xPrime =  (x-Sx/2-1)*math.cos(theta) + (y-Sy/2-1)*math.sin(theta)	--equation 1
	yPrime = -(x-Sx/2-1)*math.sin(theta)  + (y-Sy/2-1)*math.cos(theta)	--equation 2
	Gabor[x][y] = math.exp(-1/(sigma*3)*((xPrime^2)+(yPrime^2 * gamma^2 )))*math.cos(2*pi*xPrime/lambda  + shi)	-- equation 3 
	end
		end	
	return(Gabor)
end

image.display{image=gabor2(9,3,45,0,1.68,0.5), zoom =4}
--image.display{image=gabor1(51,0.5,45,1,0.5), zoom =5}

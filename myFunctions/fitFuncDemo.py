import numpy
import cv2
from scipy import optimize
from scipy import ndimage
import matplotlib.pyplot as plt

numpy.seterr(over='raise')

################################################################################
def gaussSkewPlane(data, a0, a1, a2, height, muX, muY, sigmaX, sigmaY, rho):
	f = a0 + a1*data[:,0] + a2*data[:,1] + height*numpy.e**(-1.0/(2*(1-rho**2)) * (((data[:,0]-muX)/sigmaX)**2 + ((data[:,1]-muY)/sigmaY)**2 - 2*rho*(data[:,0]-muX)*(data[:,1]-muY)/(sigmaX*sigmaY)))
	return f
################################################################################


################################################################################
def initialGuess(data):
	data = data.astype('double')
	a0 = data[:,2].min()
	a1 = 0.0
	a2 = 0.0
	height = data[:,2].max()
	total = data[:,2].sum()
	muX = 1.0*(data[:,0]*data[:,2]).sum()/total
	muY = 1.0*(data[:,1]*data[:,2]).sum()/total
	subDataX = data[data[:,1]==int(muY),:]
	subDataY = data[data[:,0]==int(muX),:]
	sigmaX = numpy.sqrt(numpy.sum((subDataX[:,0]-muX)**2 * subDataX[:,2])/numpy.sum(subDataX[:,2]))
	sigmaY = numpy.sqrt(numpy.sum((subDataY[:,1]-muY)**2 * subDataY[:,2])/numpy.sum(subDataY[:,2]))
	rho = 0.0
	return a0,a1,a2,height,muX,muY,sigmaX,sigmaY,rho
################################################################################


################################################################################
def fitting(rawData):
	[row,col] = rawData.shape
	mask = numpy.zeros([row,col],dtype='uint8')
	x, y = numpy.indices(rawData.shape)
	data = numpy.column_stack((numpy.ndarray.flatten(x), numpy.ndarray.flatten(y), numpy.ndarray.flatten(rawData)))

	[a0,a1,a2,height,muX,muY,sigmaX,sigmaY,rho] = initialGuess(data)
	[params, pcov] = optimize.curve_fit(gaussSkewPlane, data[:,:2], data[:,2], [a0,a1,a2,height,muX,muY,sigmaX,sigmaY,rho])
	return params
################################################################################


rawData = numpy.zeros([100,100])
rawData[60,45] = 200
rawData = ndimage.gaussian_filter(rawData,sigma=20)
params = fitting(rawData)

a0     = params[0]
a1     = params[1]
a2     = params[2]
height = params[3]
muX    = params[4]
muY    = params[5]
sigmaX = numpy.abs(params[6])
sigmaY = numpy.abs(params[7])
rho    = params[8]
x, y = numpy.indices(rawData.shape)
fitData = a0 + a1*x +a2*y + height * numpy.e**(-1.0/(2*(1-rho**2)) * \
		  (((x-muX)/sigmaX)**2 + \
		  ((y-muY)/sigmaY)**2 - \
		  2*rho*(x-muX)*(y-muY)/(sigmaX*sigmaY)))
		   
plt.figure()
plt.subplot(121), plt.imshow(rawData)
plt.subplot(122), plt.imshow(fitData)
plt.show()

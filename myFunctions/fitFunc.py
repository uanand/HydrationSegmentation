import numpy
import cv2
from scipy import optimize

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
	#(((data[:,0]-muX)*(data[:,1]-muY)*data[:,2]).sum() / data[:,2].sum()) / (sigmaX*sigmaY)
	return a0,a1,a2,height,muX,muY,sigmaX,sigmaY,rho
################################################################################


################################################################################
def fitting(rawData):
	[row,col] = rawData.shape
	mask = numpy.zeros([row,col],dtype='uint8')
	x, y = numpy.indices(rawData.shape)
	data = numpy.column_stack((numpy.ndarray.flatten(x), numpy.ndarray.flatten(y), numpy.ndarray.flatten(rawData)))
	flag = True
	k = 1

	[a0,a1,a2,height,muX,muY,sigmaX,sigmaY,rho] = initialGuess(data)
	try:
		[params, pcov] = optimize.curve_fit(gaussSkewPlane, data[:,:2], data[:,2], [a0,a1,a2,height,muX,muY,sigmaX,sigmaY,rho])
	except:
		flag = False

	if (flag==True):
		a0     = params[0]
		a1     = params[1]
		a2     = params[2]
		height = params[3]
		muX    = params[4]
		muY    = params[5]
		sigmaX = numpy.abs(params[6])
		sigmaY = numpy.abs(params[7])
		rho    = params[8]

		if (sigmaX<0.5 or sigmaY<0.5):
			flag=False
		else:
			covMatrix = numpy.array([[sigmaX**2, rho*sigmaX*sigmaY],[rho*sigmaX*sigmaY, sigmaY**2]])
			eigVals, eigVecs = numpy.linalg.eig(covMatrix)

			majorIndex = numpy.argmax(eigVals); minorIndex = numpy.argmin(eigVals)
			majorAxis = 2*numpy.sqrt(k * eigVals[majorIndex])
			minorAxis = 2*numpy.sqrt(k * eigVals[minorIndex])
			theta = numpy.arctan2(eigVecs[majorIndex][1], eigVecs[majorIndex][0])
			if (theta<0):
				theta += 2*numpy.pi
			theta -= numpy.pi/2

			#fitData = a0 + a1*x +a2*y + height * numpy.e**(-1.0/(2*(1-rho**2)) * \
					   #(((x-muX)/sigmaX)**2 + \
					   #((y-muY)/sigmaY)**2 - \
					   #2*rho*(x-muX)*(y-muY)/(sigmaX*sigmaY)))

			cv2.ellipse(mask,center=(int(numpy.round(muY)),int(numpy.round(muX))),\
						axes=(int(numpy.round(majorAxis)),int(numpy.round(minorAxis))),\
						angle=numpy.rad2deg(theta),startAngle=0,endAngle=360,\
						color=1,thickness=-1)
	return mask.astype('bool'), flag
################################################################################

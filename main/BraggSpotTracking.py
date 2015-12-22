import sys
import os
import numpy
import cv2
from scipy import ndimage
from skimage.morphology import disk, white_tophat
from skimage.filter.rank import enhance_contrast, tophat
import matplotlib.pyplot as plt
import cPickle as pickle
from scipy import optimize

sys.path.append(os.path.abspath('../myFunctions'))
import myPythonFunc
import myCythonFunc
import fitFunc

sys.path.append(os.path.abspath('../myClasses'))
from EMImaging import dataProcessing
from EMImaging import segmentation

inputDir = r'/home/utkarsh/Desktop/Jingyu'
inputFile = r'/home/utkarsh/Desktop/Jingyu/fringesTest.tif'
magnification = 400000
numPixels = 1024 #1024,866,480
microscopeName = 'JOEL2010' #'JOEL2010','T12'
fps = 10
labelStackFlag = False
structure = [[0,1,0],[1,1,1],[0,1,0]]
[pixInNM,pixInAngstrom] = myPythonFunc.findPixelWidth(mag=magnification,numPixels=numPixels,microscopeName=microscopeName)

if (os.path.exists(inputDir+'/metaData')):
	metaData = pickle.load(open(inputDir+'/metaData','rb'))


################################################################################
################################################################################
# USER DEFINED FUNCTION TO LABEL BINARY STACK
################################################################################
################################################################################

################################################################################
def labelBraggPeaks(bImgDir, gImgDir, row, col, numFrames, thetaTh, rTickTh, missFramesTh, frameList, structure, labelDataDir, labelImgDir):
	braggLabelStack = numpy.zeros([row, col, numFrames], dtype='uint16')
	for frame in frameList:
		bImg = numpy.load(bImgDir+'/'+str(frame)+'.npy')
		gImg = numpy.abs(numpy.load(gImgDir+'/'+str(frame)+'.npy'))**2

		if (frame == frameList[0]):
			labelImg_0, numLabel_0, dictionary_0 = myPythonFunc.regionProps(bImg, gImg, structure=structure, theta=True, rTick=True)
			maxID = numLabel_0
			dictionary_0['frame'] = []
			for j in range(len(dictionary_0['id'])):
				dictionary_0['frame'].append(frame)
			braggLabelStack[:,:,frame-1] = labelImg_0
		else:
			labelImg_1, numLabel_1, dictionary_1 = myPythonFunc.regionProps(bImg, gImg, structure=structure, theta=True, rTick=True)
			for j in range(len(dictionary_1['id'])):
				flag = 0
				bImg_1_LabelN = labelImg_1==dictionary_1['id'][j]
				theta_1 = dictionary_1['theta'][j]
				rTick_1 = dictionary_1['rTick'][j]*pixInAngstrom
				frame_1 = frame
				for k in range(len(dictionary_0['id'])-1,-1,-1):
					theta_0 = dictionary_0['theta'][k]
					rTick_0 = dictionary_0['rTick'][k]*pixInAngstrom
					frame_0 = dictionary_0['frame'][k]
					if (numpy.abs(rTick_1-rTick_0)<=rTickTh and (numpy.abs(theta_1-theta_0)<thetaTh or (360-numpy.abs(theta_1-theta_0))<thetaTh)):
						braggLabelStack[:,:,frame-1] += (bImg_1_LabelN*dictionary_0['id'][k])
						dictionary_0['theta'][k] = theta_1
						dictionary_0['rTick'][k] = rTick_1/pixInAngstrom
						dictionary_0['frame'][k] = frame
						flag = 1
						break
				if (flag == 0):
					maxID += 1
					labelN_1 = bImg_1_LabelN*maxID
					braggLabelStack[:,:,frame-1] += labelN_1
					dictionary_0['id'].append(maxID)
					dictionary_0['theta'].append(theta_1)
					dictionary_0['rTick'].append(rTick_1/pixInAngstrom)
					dictionary_0['frame'].append(frame)
	if (braggLabelStack.max() < 256):
		braggLabelStack = braggLabelStack.astype('uint8')
	elif (braggLabelStack.max()<65536):
		braggLabelStack = braggLabelStack.astype('uint16')

	for frame in frameList:
		bImg = numpy.load(bImgDir+'/'+str(frame)+'.npy')
		gImg = numpy.abs(numpy.load(gImgDir+'/'+str(frame)+'.npy'))**2
		bImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(bImg))
		labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, structure=structure, centroid=True)
		bImg = myPythonFunc.normalize(bImg)
		for j in range(len(dictionary['id'])):
			bImgLabelN = labelImg == dictionary['id'][j]
			ID = numpy.max(bImgLabelN*braggLabelStack[:,:,frame-1])
			cv2.putText(bImg, str(ID), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=127, thickness=2, bottomLeftOrigin=False)
		finalImage = numpy.column_stack((bImg, myPythonFunc.normalize(numpy.log10(gImg+1))))
		numpy.save(labelDataDir+'/'+str(frame)+'.npy', braggLabelStack[:,:,frame-1])
		cv2.imwrite(labelImgDir+'/'+str(frame)+'.png', finalImage)
	numpy.save(labelDataDir+'/braggLabelStack.npy', braggLabelStack)
	del braggLabelStack
	return maxID
################################################################################

################################################################################
################################################################################



################################################################################
################################################################################
# CREATING THE RAW DATA STACK
################################################################################
################################################################################
#hP = dataProcessing(inputDir, inputFile, magnification, numPixels=numPixels, microscopeName=microscopeName)
##hP.createRawStack()
#hP.readTIFF(inputFile)
#del hP
################################################################################
################################################################################



################################################################################
################################################################################
# CREATING METADATA FOR THE INPUT FILE
################################################################################
################################################################################
#if (os.path.exists(inputDir+'/metaData')):
	#metaData = pickle.load(open(inputDir+'/metaData','rb'))
#qArr, qTcks, rTcks, pixelTicks = myPythonFunc.calculate_qArr(numpy.load(inputDir+'/output/data/dataProcessing/gImgRawStack/1.npy'),pixInAngstrom)
#metaData['qArr']=qArr
#if (labelStackFlag==False):
	#metaData['particleList'] = ['raw']
#else:
	#metaData['particleList'].append('raw')
#pickle.dump(metaData, open(inputDir+'/metaData', 'wb'))
################################################################################
################################################################################



################################################################################
################################################################################
# CALCULATING ORDERED PARAMTER FOR THE IMAGE STACK
################################################################################
################################################################################
#print "CALCULATING ORDERED PARAMETER"
#gaussBlurSize = 4
#qArr = metaData['qArr']
#particleList = metaData['particleList']
#frameList = metaData['frameList']

#for particle in particleList:
	#myPythonFunc.mkdir(inputDir+'/output/data/braggPeaks')
	#myPythonFunc.mkdir(inputDir+'/output/images/braggPeaks')
	#myPythonFunc.mkdir(inputDir+'/output/data/braggPeaks/kymograph')
	#myPythonFunc.mkdir(inputDir+'/output/images/braggPeaks/kymograph')
	#myPythonFunc.mkdir(inputDir+'/output/data/braggPeaks/kymograph/'+str(particle))
	#myPythonFunc.mkdir(inputDir+'/output/images/braggPeaks/kymograph/'+str(particle))
	#print "Processing particle", particle
	
	#ordParamList = []
	#for frame in frameList:
		#gImg = numpy.load(inputDir+'/output/data/dataProcessing/gImgRawStack/'+str(frame)+'.npy')
		#if (particle != 'raw'):
			#particleLabelImg = numpy.load(inputDir+'/output/data/particles/tracking/'+str(frame)+'.npy')
			#mask = ndimage.gaussian_filter((particleLabelImg==particle).astype('double'), sigma=gaussBlurSize)
			#if (mask.max()>0):
				#mask = (mask-mask.min())/(mask.max()-mask.min())
			#gImg = (gImg*mask).astype('uint8')
		#ordParamList.append(myPythonFunc.calculate_ordParam(gImg, qArr))
	#ordParamArr = numpy.array(ordParamList)
	#numpy.save(inputDir+'/output/data/braggPeaks/kymograph/'+str(particle)+'/ordParamArr.npy', ordParamArr)
################################################################################
################################################################################



################################################################################
################################################################################
# PLOTTING ORDERED PARAMETER
################################################################################
################################################################################
print "PLOTTING ORDERED PARAMETER"
particleList = metaData['particleList']
gamma = 0.4
ringRange = [1,5]
step_size = 10
rTicksManual = []

for particle in particleList:
	gImg = numpy.load(inputDir+'/output/data/dataProcessing/gImgRawStack/1.npy')
	outputDir = inputDir+'/output/images/braggPeaks/kymograph/'+str(particle)
	ordParamArr = numpy.load(inputDir+'/output/data/braggPeaks/kymograph/'+str(particle)+'/ordParamArr.npy')
	plt.figure()
	plt.subplot(311), plt.plot(ordParamArr[0,:]**gamma), plt.xlim(50,250), plt.minorticks_on()
	plt.subplot(312), plt.plot(ordParamArr[1,:]**gamma), plt.xlim(50,250), plt.minorticks_on()
	plt.subplot(313), plt.plot(ordParamArr[2,:]**gamma), plt.xlim(50,250), plt.minorticks_on()
	plt.savefig(inputDir+'/output/images/braggPeaks/kymograph/'+str(particle)+'/1dOrdPar.png', format='png', bbox_inches='tight')
	plt.close()
	myPythonFunc.plotOrderParam(outputDir, ordParamArr, gImg, pixInAngstrom, gamma, ringRange, step_size, manualXTicks=rTicksManual, fps=fps)
################################################################################
################################################################################



################################################################################
################################################################################
# DETECTION OF BRAGG DIFFRACTION PEAKS USING THE TOP-HAT TRANSFORM
################################################################################
################################################################################
#print "SEGMENTING OUT THE BRAGG SPOTS USING THE TOP-HAT TRANSFORM"
#gaussBlurSize = 3
#qTickToKeepList = [120]
#margin = 10
#[row,col,numFrames] = [metaData['row'], metaData['col'], metaData['numFrames']]
#particleList = metaData['particleList']
#frameList = metaData['frameList']

#myPythonFunc.mkdir(inputDir+'/output/images/braggPeaks/THTDetection')
#myPythonFunc.mkdir(inputDir+'/output/data/braggPeaks/THTDetection')
#myPythonFunc.mkdir(inputDir+'/output/images/braggPeaks/fft')
#myPythonFunc.mkdir(inputDir+'/output/data/braggPeaks/fft')
#myPythonFunc.mkdir(inputDir+'/output/images/braggPeaks/gImg')
#myPythonFunc.mkdir(inputDir+'/output/data/braggPeaks/gImg')

#for particle in particleList:
	#myPythonFunc.mkdir(inputDir+'/output/images/braggPeaks/THTDetection/'+str(particle))
	#myPythonFunc.mkdir(inputDir+'/output/data/braggPeaks/THTDetection/'+str(particle))
	#myPythonFunc.mkdir(inputDir+'/output/images/braggPeaks/fft/'+str(particle))
	#myPythonFunc.mkdir(inputDir+'/output/data/braggPeaks/fft/'+str(particle))
	#myPythonFunc.mkdir(inputDir+'/output/images/braggPeaks/gImg/'+str(particle))
	#myPythonFunc.mkdir(inputDir+'/output/data/braggPeaks/gImg/'+str(particle))
	
	#print particle
	#for frame in frameList:
		#gImg = numpy.load(inputDir+'/output/data/dataProcessing/gImgRawStack/'+str(frame)+'.npy')
		#if (particle != 'raw'):
			#particleLabelImg = numpy.load(inputDir+'/output/data/particles/tracking/'+str(frame)+'.npy')
			#mask = ndimage.gaussian_filter((particleLabelImg==particle).astype('double'), sigma=gaussBlurSize)
			#if (mask.max()>0):
				#mask = (mask-mask.min())/(mask.max()-mask.min())
			#gImg = (gImg*mask).astype('uint8')
		##else:
			##mask = numpy.zeros([row,col], dtype='uint8')
			##cv2.circle(mask, center=(col/2,row/2), radius=row-row/2-10, color=1, thickness=-1)
			##mask = mask.astype('double')
			##mask = ndimage.gaussian_filter(mask, sigma=gaussBlurSize+2)
			##if (mask.max()>0):
					##mask = (mask-mask.min())/(mask.max()-mask.min())
			##gImg = (gImg*mask).astype('uint8')
		#fImg = numpy.fft.fftshift(numpy.fft.fftn(gImg))
		#fImgAbs = numpy.abs(fImg)**2
		#fImgBlur=myPythonFunc.normalize(numpy.log10(ndimage.gaussian_filter(fImgAbs,gaussBlurSize)+1))
		#fImgTHT=white_tophat(fImgBlur, selem=disk(2*gaussBlurSize)+2)
		#bImg = fImgTHT>=myCythonFunc.threshold_kapur(fImgTHT.flatten())
		#[labelImg, numLabel, dictionary] = myPythonFunc.regionProps(bImg, fImgAbs, structure, qTick=True)
		#for spotNum, qTick in zip(dictionary['id'], dictionary['qTick']):
			#flag = 0
			#for qTickToKeep in qTickToKeepList:
				#if (qTick>=qTickToKeep-margin and qTick<=qTickToKeep+margin):
					#flag=1
			#if (flag == 0):
				#labelImg[labelImg==spotNum]=0
		#bImgFilter = labelImg.astype('bool')
		#bImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(bImgFilter))
		#finalImage = numpy.maximum(myPythonFunc.normalize(numpy.log10(fImgAbs+1),0,230), myPythonFunc.normalize(bImgBdry))
		#cv2.imwrite(inputDir+'/output/images/braggPeaks/gImg/'+str(particle)+'/'+str(frame)+'.png', gImg)
		#cv2.imwrite(inputDir+'/output/images/braggPeaks/fft/'+str(particle)+'/'+str(frame)+'.png', myPythonFunc.normalize(numpy.log10(fImgAbs+1)))
		#cv2.imwrite(inputDir+'/output/images/braggPeaks/THTDetection/'+str(particle)+'/'+str(frame)+'.png', finalImage)
		#numpy.save(inputDir+'/output/data/braggPeaks/gImg/'+str(particle)+'/'+str(frame)+'.npy', gImg)
		#numpy.save(inputDir+'/output/data/braggPeaks/fft/'+str(particle)+'/'+str(frame)+'.npy', fImg)
################################################################################
################################################################################



################################################################################
################################################################################
# CREATE BINARY IMAGE STACK
################################################################################
################################################################################
#print 'CREATING bImgStack'
#particleList = metaData['particleList']
#frameList = metaData['frameList']

#for particle in particleList:
	#for frame in frameList:
		#img = cv2.imread(inputDir+'/output/images/braggPeaks/THTDetection/'+str(particle)+'/'+str(frame)+'.png',0)
		#bImg = ndimage.binary_fill_holes(img==255)
		#numpy.save(inputDir+'/output/data/braggPeaks/THTDetection/'+str(particle)+'/'+str(frame)+'.npy', bImg)
################################################################################
################################################################################



################################################################################
################################################################################
# LABELING OF BRAGG PEAKS
################################################################################
################################################################################
#print "LABELLING SEGMENTED BRAGG SPOTS ACROSS TIME"
#[row,col,numFrames] = [metaData['row'], metaData['col'], metaData['numFrames']]
#particleList = metaData['particleList']
#frameList = metaData['frameList']

#myPythonFunc.mkdir(inputDir+'/output/data/braggPeaks/tracking')
#myPythonFunc.mkdir(inputDir+'/output/images/braggPeaks/tracking')

#thetaTh = 5
#rTickTh = 0.1
#missFramesTh = 20

#if ('braggParticleList' not in metaData.keys()):
	#metaData['braggParticleList'] = {}

#for particle in particleList:
	#myPythonFunc.mkdir(inputDir+'/output/data/braggPeaks/tracking/'+particle)
	#myPythonFunc.mkdir(inputDir+'/output/images/braggPeaks/tracking/'+particle)

	#bImgDir = inputDir+'/output/data/braggPeaks/THTDetection/'+str(particle)
	#gImgDir = inputDir+'/output/data/braggPeaks/fft/'+str(particle)
	#labelDataDir = inputDir+'/output/data/braggPeaks/tracking/'+str(particle)
	#labelImgDir = inputDir+'/output/images/braggPeaks/tracking/'+str(particle)

	#maxID = labelBraggPeaks(bImgDir, gImgDir, row, col, numFrames, thetaTh, rTickTh, missFramesTh, frameList, structure, labelDataDir, labelImgDir)
	#metaData['braggParticleList'][particle] = range(1,maxID+1)
#pickle.dump(metaData, open(inputDir+'/metaData', 'wb'))
##############################################################################
################################################################################




##############################################################
##############################################################
#REMOVING UNWANTED PARTICLES
##############################################################
##############################################################
#print "REMOVING UNWANTED PARTICLES"
#keepDict = {}
#removeDict = {'raw':[13,14]}

#particleList = metaData['particleList']
#frameList = metaData['frameList']

#for particle in particleList:
	#braggParticleList = metaData['braggParticleList'][particle]

	#try:
		#keepList = keepDict[particle]
	#except:
		#keepList = []
	#try:
		#removeList = removeDict[particle]
	#except:
		#removeList = []
	#if not removeList:
		#removeList = [s for s in braggParticleList if s not in keepList]

	#for frame in frameList:
		#labelImg = numpy.load(inputDir+'/output/data/braggPeaks/tracking/'+str(particle)+'/'+str(frame)+'.npy')
		#for i in removeList:
			#labelImg[labelImg==i] = 0
		#bImg = labelImg.astype('bool')
		#gImg = cv2.imread(inputDir+'/output/images/braggPeaks/fft/'+str(particle)+'/'+str(frame)+'.png',0)
		#label, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, structure=structure, centroid=True)
		#bImg = myPythonFunc.normalize(bImg)
		#for j in range(len(dictionary['id'])):
			#bImgLabelN = label == dictionary['id'][j]
			#ID = numpy.max(bImgLabelN*labelImg)
			#cv2.putText(bImg, str(ID), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=127, thickness=2, bottomLeftOrigin=False)
		#finalImage = numpy.column_stack((bImg, gImg))
		#cv2.imwrite(inputDir+'/output/images/braggPeaks/tracking/'+str(particle)+'/'+str(frame)+'.png', finalImage)
		#numpy.save(inputDir+'/output/data/braggPeaks/tracking/'+str(particle)+'/'+str(frame)+'.npy', labelImg)
		#numpy.save(inputDir+'/output/data/braggPeaks/THTDetection/'+str(particle)+'/'+str(frame)+'.npy', labelImg.astype('bool'))

	#for r in removeList:
		#try:
			#metaData['braggParticleList'][particle].remove(r)
		#except:
			#pass
#pickle.dump(metaData, open(inputDir+'/metaData', 'wb'))
##############################################################
##############################################################




##############################################################
##############################################################
#LABEL CORRECTION
##############################################################
##############################################################
#print "LABEL CORRECTION"
#[row, col, numFrames] = [metaData['row'], metaData['col'], metaData['numFrames']]
#qArr = metaData['qArr']
#frameList = metaData['frameList']
#particleList = metaData['braggParticleList']['raw']

#correctionList = [[]]

#for frame in frameList:
	#labelImg = numpy.load(inputDir+'/output/data/braggPeaks/tracking/raw/'+str(frame)+'.npy')
	#for i in range(len(correctionList)):
		#for j in range(len(correctionList[i])-1):
			#labelImg[labelImg==correctionList[i][j]] = correctionList[i][-1]
	#numpy.save(inputDir+'/output/data/braggPeaks/tracking/raw/'+str(frame)+'.npy', labelImg)
#for i in correctionList:
	#for j in i[:-1]:
		#try:
			#metaData['braggParticleList']['raw'].remove(j)
		#except:
			#pass

##frameWiseCorrection = {'raw':[[range(100,numFrames+1),7,60]],\
					  ##}
##for corrList in frameWiseCorrection:

#maxLabel = max(metaData['braggParticleList']['raw'])+1; counter=1

#newLabels = {}
#for particle in metaData['braggParticleList']['raw']:
	#newLabels[particle]=[]

#for frame in frameList:
	#particlesInFrame = numpy.unique(numpy.load(inputDir+'/output/data/braggPeaks/tracking/raw/'+str(frame)+'.npy'))[1:]
	#for p in particlesInFrame:
		#if not newLabels[p]:
			#newLabels[p] = [maxLabel, counter]
			#maxLabel+=1; counter+=1

#for frame in frameList:
	#labelImg = numpy.load(inputDir+'/output/data/braggPeaks/tracking/raw/'+str(frame)+'.npy')
	#for keys in newLabels.keys():
		#labelImg[labelImg==keys] = newLabels[keys][0]
	#for keys in newLabels.keys():
		#labelImg[labelImg==newLabels[keys][0]] = newLabels[keys][1]

	#bImg = labelImg.astype('bool')
	#gImg = cv2.imread(inputDir+'/output/images/braggPeaks/fft/'+str(frame)+'.png',0)
	#label, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, structure=structure, centroid=True)
	#bImg = myPythonFunc.normalize(bImg)
	#for j in range(len(dictionary['id'])):
		#bImgLabelN = label == dictionary['id'][j]
		#ID = numpy.max(bImgLabelN*labelImg)
		#cv2.putText(bImg, str(ID), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=127, thickness=2, bottomLeftOrigin=False)
	##[y,x] = numpy.unravel_index(qArr.argmin(), qArr.shape)
	##cv2.circle(bImg, center=(x,y), radius=5, color=255, thickness=-1)
	#finalImage = numpy.column_stack((bImg, gImg))
	#cv2.imwrite(inputDir+'/output/images/braggPeaks/tracking/raw/'+str(frame)+'.png', finalImage)
	#numpy.save(inputDir+'/output/data/braggPeaks/tracking/raw/'+str(frame)+'.npy', labelImg)
##############################################################
##############################################################




##############################################################
##############################################################
#FITTING EVERY IDENTIFIED BRAGG SPOT WITH A FIT FUNCTION
##############################################################
##############################################################
#print "PEAK FITTING"
#[row, col, numFrames] = [metaData['row'], metaData['col'], metaData['numFrames']]
#particleList = metaData['particleList']
#frameList = metaData['frameList']
#cropSize = 12

#myPythonFunc.mkdir(inputDir+'/output/data/braggPeaks/fitting')
#myPythonFunc.mkdir(inputDir+'/output/images/braggPeaks/fitting')
#bigMask = numpy.zeros([row,col], dtype='bool')
#failedBraggFitting={}

#outFile = open(inputDir+'/output/images/braggPeaks/fitting/fitFailLog.dat', 'wb')
#outFile.write("Particle Frame BraggSpotID\n")

#for particle in particleList:
	#myPythonFunc.mkdir(inputDir+'/output/data/braggPeaks/fitting/'+str(particle))
	#myPythonFunc.mkdir(inputDir+'/output/images/braggPeaks/fitting/'+str(particle))
	#failedBraggFitting[particle]={}

	#for frame in frameList:
		#print frame
		#labelImg = numpy.load(inputDir+'/output/data/braggPeaks/tracking/'+str(particle)+'/'+str(frame)+'.npy')
		#fImg = numpy.abs(numpy.load(inputDir+'/output/data/braggPeaks/fft/'+str(particle)+'/'+str(frame)+'.npy'))**2
		#labels = numpy.unique(labelImg)[1:]
		#failedBraggFitting[particle][frame] = []
		#for label in labels:
			#bImg = labelImg==label
			#temp1, temp2, dictionary = myPythonFunc.regionProps(bImg, fImg, structure=structure, centroid=True)
			#fImgCrop = fImg[dictionary['centroid'][0][0]-cropSize:dictionary['centroid'][0][0]+cropSize+1, dictionary['centroid'][0][1]-cropSize:dictionary['centroid'][0][1]+cropSize+1]
			#mask, flag = fitFunc.fitting(ndimage.gaussian_filter(fImgCrop,1))
			#bigMask[:] = False
			#bigMask[dictionary['centroid'][0][0]-cropSize:dictionary['centroid'][0][0]+cropSize+1, dictionary['centroid'][0][1]-cropSize:dictionary['centroid'][0][1]+cropSize+1] = mask
			#labelImg[labelImg==label] = 0
			#labelImg[bigMask==True] = label

			#if (flag == True):
				#labelImg[labelImg==label] = 0
				#labelImg[bigMask==True] = label
			#else:
				#failedBraggFitting[particle][frame].append(label)
				#outFile.write("%s %d %d\n" %(str(particle), frame, label))

		#bImg = labelImg.astype('bool')
		#bImgBdry = myPythonFunc.boundary(bImg)
		#gImg = cv2.imread(inputDir+'/output/images/braggPeaks/fft/'+str(particle)+'/'+str(frame)+'.png',0)

		#label, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, structure=structure, centroid=True)
		#bImg = myPythonFunc.normalize(bImg)
		#bImgBdry = myPythonFunc.normalize(bImgBdry)
		#for j in range(len(dictionary['id'])):
			#bImgLabelN = label == dictionary['id'][j]
			#ID = numpy.max(bImgLabelN*labelImg)
			#cv2.putText(bImg, str(ID), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=127, thickness=2, bottomLeftOrigin=False)
		#cv2.circle(bImg, center=(col/2,row/2), radius=5, color=255, thickness=-1)
		#finalImage = numpy.column_stack((bImg, numpy.maximum(gImg, bImgBdry)))
		#cv2.imwrite(inputDir+'/output/images/braggPeaks/fitting/'+str(particle)+'/'+str(frame)+'.png', finalImage)
		#numpy.save(inputDir+'/output/data/braggPeaks/fitting/'+str(particle)+'/'+str(frame)+'.npy', labelImg)
#outFile.close()
##############################################################
##############################################################



##############################################################
##############################################################
# CALCULATE MEASURES FOR ALL BRAGG REFLECTIONS
##############################################################
##############################################################
#print "FINDING MEASURES FOR ALL BRAGG REFLECTIONS"
#particleList = metaData['particleList']
#frameList = metaData['frameList']

#myPythonFunc.mkdir(inputDir+'/output/data/braggPeaks/measures')

#rTickMMM = True
#thetaMMM = True

#braggDataDict = {}
#for particle in particleList:
	#braggDataDict[particle] = {}
	#braggParticleList = metaData['braggParticleList'][particle]
	#myPythonFunc.mkdir(inputDir+'/output/data/braggPeaks/measures/'+str(particle))
	#outFile = open(inputDir+'/output/data/braggPeaks/measures/'+str(particle)+'/braggData.dat', 'wb')
	#for braggSpot in braggParticleList:
		#braggDataDict[particle][braggSpot] = {}
	#for frame in frameList:
		#outFile.write("%s %f " %(str(particle), 1.0*frame/fps))
		#labelImg = numpy.load(inputDir+'/output/data/braggPeaks/tracking/'+str(particle)+'/'+str(frame)+'.npy')
		#fImg = numpy.abs(numpy.load(inputDir+'/output/data/braggPeaks/fft/'+str(particle)+'/'+str(frame)+'.npy'))**2
		#for label in braggParticleList:
			#bImg = labelImg==label
			#if (bImg.max() == True):
				#temp1, temp2, dictionary = myPythonFunc.regionProps(bImg, fImg, structure, rTickMMM=rTickMMM, thetaMMM=thetaMMM)
				#braggDataDict[particle][label][frame] = dictionary
				#outFile.write("%f %f %f %f %f %f " %(dictionary['rTickMean'][0]*pixInAngstrom, dictionary['rTickMin'][0]*pixInAngstrom, dictionary['rTickMax'][0]*pixInAngstrom, dictionary['thetaMean'][0], dictionary['dThetaM'][0], dictionary['dThetaP'][0]))
				##outFile.write("%f %f " %(dictionary['rTick'][0]*pixInAngstrom, dictionary['theta'][0]))
			#else:
				#outFile.write("nan nan nan nan nan nan ")
		#outFile.write("\n")
	#outFile.close()
#pickle.dump(braggDataDict, open(inputDir+'/output/data/braggPeaks/measures/braggDataDict.dat', 'wb'))
##############################################################
##############################################################


##############################################################
##############################################################
# GENERATE CLEAN IMAGES WITH MASKED FFT
##############################################################
##############################################################
#print 'CREATING CLEAN IMAGES'
#[row, col, numFrames] = [metaData['row'], metaData['col'], metaData['numFrames']]
#particleList = metaData['particleList']
#frameList = metaData['frameList']

#myPythonFunc.mkdir(inputDir+'/output/images/braggPeaks/gImgClean')

#for particle in particleList:
	#myPythonFunc.mkdir(inputDir+'/output/images/braggPeaks/gImgClean/'+str(particle))
	#for frame in frameList:
		#print frame
		#fImg = numpy.load(inputDir+'/output/data/braggPeaks/fft/'+str(particle)+'/'+str(frame)+'.npy')
		#labelImg = numpy.load(inputDir+'/output/data/braggPeaks/tracking/'+str(particle)+'/'+str(frame)+'.npy')
		#cv2.circle(labelImg, center=(col/2,row/2), radius=100, color=255, thickness=-1)
		#mask = labelImg.astype('bool')
		#gImgClean = myPythonFunc.normalize(myPythonFunc.inverseFFT(fImg, mask))
		#cv2.imwrite(inputDir+'/output/images/braggPeaks/gImgClean/'+str(particle)+'/'+str(frame)+'.png', gImgClean)
##############################################################
##############################################################

import cv2
import cv
import numpy
import os.path
import scipy
import myCythonFunc
import myPythonFunc
import sys
import math
import cPickle as pickle
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.cm as cm
from scipy import ndimage
sys.setrecursionlimit(100000)

inputDir = "D:\\Utkarsh\\Zainul\\14Dislocation3Par"
inputFile = "D:\\Utkarsh\\Zainul\\14Dislocation3Par\\14Dislocation3Par.avi"

pixInAngstrom = 0.3458984375
# pixInAngstrom=0.4123046875
fontScale=1



##############################################################
##############################################################
#PRE-PROCESS - CREATING DATASETS FOR ANALYSIS
##############################################################
##############################################################
# sigma_bg = 20
# alpha = 0.6
# noiseSizeList = [4,8]
# particleSizeList = [20,60]
# methodList = ['medianOFgauss', 'gauss', 'median'] #'gaussOFmedian'

# print "Creating output directories"
# myPythonFunc.createOutputDir(inputDir)

# print "Counting number of frames"
# [row, col, numFrames] = myPythonFunc.countFrames(inputFile)

# gImgRawStack = myPythonFunc.readStack(inputFile, row, col, numFrames, sigma_bg, alpha, invert=False, backgroundSubtract=False, weight=False)
# gImgStack = myPythonFunc.readStack(inputFile, row, col, numFrames, sigma_bg, alpha, invert=True, backgroundSubtract=True, weight=True)
# fileName1 = inputDir+"/output/data/gImgRawStack.npy"
# fileName2 = inputDir+"/output/data/gImgStack.npy"
# numpy.save(fileName1, gImgRawStack)
# numpy.save(fileName2, gImgStack)

# for method in methodList:
	# if (os.path.exists(inputDir+'/output/images/particleIdentification/'+method) == False):
		# os.makedirs(inputDir+'/output/images/particleIdentification/'+method)
	# if (os.path.exists(inputDir+'/output/data/particleIdentification/'+method) == False):
		# os.makedirs(inputDir+'/output/data/particleIdentification/'+method)
	# if (method == 'medianOFgauss'):
		# for noiseSize in noiseSizeList:
			# for particleSize in particleSizeList:
				# for i in range(numFrames):
					# print method, noiseSize, particleSize, i+1
					# gImg = gImgStack[:,:,i]
					# if (i == 0):
						# gImgBlurSharpStack = numpy.zeros([row, col, numFrames], dtype='uint8')
					# gImgBlur = ndimage.gaussian_filter(gImg, sigma=2*noiseSize)
					# gImgBlurSharp = myCythonFunc.medianFilter2D_C(gImgBlur, size=[particleSize/3, particleSize/3])
					# gImgBlurSharpStack[:,:,i] = gImgBlurSharp
					# if (i == numFrames-1):
						# if (os.path.exists(inputDir+'/output/data/particleIdentification/'+method+'/'+str(noiseSize).zfill(2)+'_'+str(particleSize).zfill(3)) == False):
							# os.makedirs(inputDir+'/output/data/particleIdentification/'+method+'/'+str(noiseSize).zfill(2)+'_'+str(particleSize).zfill(3))
							# os.makedirs(inputDir+'/output/images/particleIdentification/'+method+'/'+str(noiseSize).zfill(2)+'_'+str(particleSize).zfill(3))
						# fileName = inputDir+'/output/data/particleIdentification/'+method+'/'+str(noiseSize).zfill(2)+'_'+str(particleSize).zfill(3)
						# numpy.save(fileName+'/gImgBlurSharpStack.npy', gImgBlurSharpStack)
						# del gImgBlurSharpStack
	# elif (method == 'gaussOFmedian'):
		# for particleSize in particleSizeList:
			# for noiseSize in noiseSizeList:
				# for i in range(numFrames):
					# print method, particleSize, noiseSize, i+1
					# gImg = gImgStack[:,:,i]
					# if (i == 0):
						# gImgSharpBlurStack = numpy.zeros([row, col, numFrames], dtype='uint8')
					# gImgSharp = myCythonFunc.medianFilter2D_C(gImg, size=[particleSize/3, particleSize/3])
					# gImgSharpBlur = ndimage.gaussian_filter(gImgSharp, sigma=2*noiseSize)
					# gImgSharpBlurStack[:,:,i] = gImgSharpBlur
					# if (i == numFrames-1):
						# if (os.path.exists(inputDir+'/output/data/particleIdentification/'+method+'/'+str(noiseSize).zfill(2)+'_'+str(particleSize).zfill(3)) == False):
							# os.makedirs(inputDir+'/output/data/particleIdentification/'+method+'/'+str(noiseSize).zfill(2)+'_'+str(particleSize).zfill(3))
							# os.makedirs(inputDir+'/output/images/particleIdentification/'+method+'/'+str(noiseSize).zfill(2)+'_'+str(particleSize).zfill(3))
						# fileName = inputDir+'/output/data/particleIdentification/'+method+'/'+str(noiseSize).zfill(2)+'_'+str(particleSize).zfill(3)
						# numpy.save(fileName+'/gImgSharpBlurStack.npy', gImgSharpBlurStack)
						# del gImgSharpBlurStack
	# elif (method == 'median'):
		# for particleSize in particleSizeList:
			# for i in range(numFrames):
				# print method, particleSize, i+1
				# gImg = gImgStack[:,:,i]
				# if (i == 0):
					# gImgSharpStack = numpy.zeros([row, col, numFrames], dtype='uint8')
				# gImgSharp = myCythonFunc.medianFilter2D_C(gImg, size=[particleSize/3, particleSize/3])
				# gImgSharpStack[:,:,i] = gImgSharp
				# if (i == numFrames-1):
					# if (os.path.exists(inputDir+'/output/data/particleIdentification/'+method+'/'+str(particleSize).zfill(3)) == False):
						# os.makedirs(inputDir+'/output/data/particleIdentification/'+method+'/'+str(particleSize).zfill(3))
						# os.makedirs(inputDir+'/output/images/particleIdentification/'+method+'/'+str(particleSize).zfill(3))
					# fileName = inputDir+'/output/data/particleIdentification/'+method+'/'+str(particleSize).zfill(3)
					# numpy.save(fileName+'/gImgSharpStack.npy', gImgSharpStack)
					# del gImgSharpStack
	# elif (method == 'gauss'):
		# for noiseSize in noiseSizeList:
			# for i in range(numFrames):
				# print method, noiseSize, i+1
				# gImg = gImgStack[:,:,i]
				# if (i == 0):
					# gImgBlurStack = numpy.zeros([row, col, numFrames], dtype='uint8')
				# gImgBlur = ndimage.gaussian_filter(gImg, sigma=2*noiseSize)
				# gImgBlurStack[:,:,i] = gImgBlur
				# if (i == numFrames-1):
					# if (os.path.exists(inputDir+'/output/data/particleIdentification/'+method+'/'+str(noiseSize).zfill(2)) == False):
						# os.makedirs(inputDir+'/output/data/particleIdentification/'+method+'/'+str(noiseSize).zfill(2))
						# os.makedirs(inputDir+'/output/images/particleIdentification/'+method+'/'+str(noiseSize).zfill(2))
					# fileName = inputDir+'/output/data/particleIdentification/'+method+'/'+str(noiseSize).zfill(2)
					# numpy.save(fileName+'/gImgBlurStack.npy', gImgBlurStack)
					# del gImgBlurStack
# del gImgStack
##############################################################
##############################################################



##############################################################
##############################################################
#PERFORMAING IMAGE SEGMENTATION
##############################################################
##############################################################
# print "PERFORMAING IMAGE SEGMENTATION"
# gImgStack = numpy.load(inputDir+'/output/data/gImgStack.npy')
# [row, col, numFrames] = gImgStack.shape

# noiseSizeList = [4]#[4,8]
# particleSizeList = [60]#[20,60]
# methodList = ['medianOFgauss']#['medianOFgauss', 'gauss', 'median'] #'gaussOFmedian'
# contourMethodList =  ['neighbors', 'edges', 'none']
# rawFilterList = ['median', 'none']
# frameList = range(1,numFrames+1)
# areaRange = [22500,300000]
# minNeighbors = 2
# minCircular = 0.0
# thresholdPartitions = (1,1)
# exclude = numpy.array([0], dtype='uint8')

# #Jingyu's 250k_10fps_12 minArea=37500, maxArea=360000
# #Jingyu's 250k_10fps_23_Frames11-787 minArea = 2000 maxArea = 90000
# #Guanhua's 7, minArea=300, maxArea=20000
# #Guanhua's 14, minArea=600, maxArea=90000

# for method in methodList:
	# if (method == 'medianOFgauss'):
		# for noiseSize in noiseSizeList:
			# for particleSize in particleSizeList:
				# datDirName = inputDir+'/output/data/particleIdentification/'+method+'/'+str(noiseSize).zfill(2)+'_'+str(particleSize).zfill(3)
				# imgDirName = inputDir+'/output/images/particleIdentification/'+method+'/'+str(noiseSize).zfill(2)+'_'+str(particleSize).zfill(3)
				# gImgBlurSharpStack = numpy.load(datDirName+'/gImgBlurSharpStack.npy')
				# for contourMethod in contourMethodList:
					# if (os.path.exists(imgDirName+'/'+contourMethod) == False):
						# os.makedirs(imgDirName+'/'+contourMethod)
						# for rawFilter in rawFilterList:
							# if (os.path.exists(imgDirName+'/'+contourMethod+'/'+rawFilter) == False):
								# os.makedirs(imgDirName+'/'+contourMethod+'/'+rawFilter)
				# for i in frameList:
					# print method, noiseSize, particleSize, i
					# for rawFilter in rawFilterList:
						# gImg = gImgStack[:,:,i-1]
						# gImgBlurSharp = gImgBlurSharpStack[:,:,i-1]
						# gImgBlurSharpMedian = (myPythonFunc.adaptiveThreshold(gImgBlurSharp, method='median', numPartitions=thresholdPartitions))*gImgBlurSharp
						
						# if (rawFilter == 'none'):
							# gImgInput = gImgBlurSharpMedian.astype('uint8')
						# elif (rawFilter == 'median'):
							# gImgBlur2 = ndimage.gaussian_filter(gImg, sigma=2)
							# gImgMedian = (myPythonFunc.adaptiveThreshold(gImgBlur2, method='median', numPartitions=thresholdPartitions))
							# gImgBlurSharpgImgMedian = gImgBlurSharpMedian*gImgMedian
							# gImgInput = gImgBlurSharpgImgMedian.astype('uint8')
							
						# allContours = (myCythonFunc.gradient(gImgInput.astype('double'), exclude)).astype('bool')
						# for contourMethod in contourMethodList:
							# denseContours = (myCythonFunc.cleanContour(allContours.astype('uint8'), contourMethod, minNeighbors)).astype('bool')
							# bImgFill = ndimage.binary_fill_holes(denseContours)
							# bImgInner = bImgFill
							# # bImgInner = (myCythonFunc.removeBoundaryParticles(bImgFill.astype('uint8'))).astype('bool')
							# bImgBig = myCythonFunc.areaThreshold(bImgInner.astype('uint8'), minArea=areaRange[0], maxArea=areaRange[1]).astype('bool')
							# bImgOpen = myPythonFunc.binary_opening(bImgBig, iterations=2*noiseSize)
							# bImg = myCythonFunc.areaThreshold(bImgOpen.astype('uint8'), minArea=areaRange[0], maxArea=areaRange[1])
							# bImgCircular = myCythonFunc.circularThreshold(bImg.astype('uint8'), minCircular).astype('bool')
							# bImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(bImgCircular))
							
							# labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImgCircular, gImg, centroid=True)
							# bImgTagged = myPythonFunc.normalize(bImgCircular)
							# for j in range(len(dictionary['id'])):
								# cv2.putText(bImgTagged, str(dictionary['id'][j]), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=1, bottomLeftOrigin=False)
								
							# finalImage = numpy.row_stack((numpy.column_stack((gImgInput, myPythonFunc.normalize(allContours), myPythonFunc.normalize(denseContours), myPythonFunc.normalize(bImgFill), myPythonFunc.normalize(bImgInner))), numpy.column_stack((myPythonFunc.normalize(bImgBig), myPythonFunc.normalize(bImgOpen), myPythonFunc.normalize(bImgCircular), bImgTagged, numpy.maximum(gImg, bImgBdry)))))
							# cv2.imwrite(imgDirName+'/'+contourMethod+'/'+rawFilter+'/'+str(i).zfill(4)+'.tiff', finalImage)
				# del gImgBlurSharpStack
	# elif (method == 'median'):
		# for particleSize in particleSizeList:
			# datDirName = inputDir+'/output/data/particleIdentification/'+method+'/'+str(particleSize).zfill(3)
			# imgDirName = inputDir+'/output/images/particleIdentification/'+method+'/'+str(particleSize).zfill(3)
			# gImgSharpStack = numpy.load(datDirName+'/gImgSharpStack.npy')
			# for contourMethod in contourMethodList:
				# if (os.path.exists(imgDirName+'/'+contourMethod) == False):
					# os.makedirs(imgDirName+'/'+contourMethod)
					# for rawFilter in rawFilterList:
						# if (os.path.exists(imgDirName+'/'+contourMethod+'/'+rawFilter) == False):
							# os.makedirs(imgDirName+'/'+contourMethod+'/'+rawFilter)
			# for i in frameList:
				# print method, particleSize, i
				# for rawFilter in rawFilterList:
					# gImg = gImgStack[:,:,i-1]
					# gImgSharp = gImgSharpStack[:,:,i-1]
					# gImgSharpMedian = (myPythonFunc.adaptiveThreshold(gImgSharp, method='median', numPartitions=thresholdPartitions))*gImgSharp
					
					# if (rawFilter == 'none'):
						# gImgInput = gImgSharpMedian.astype('uint8')
					# elif (rawFilter == 'median'):
						# gImgBlur2 = ndimage.gaussian_filter(gImg, sigma=2)
						# gImgMedian = (myPythonFunc.adaptiveThreshold(gImgBlur2, method='median', numPartitions=thresholdPartitions)).astype('uint8')
						# gImgSharpgImgMedian = gImgSharpMedian*gImgMedian
						# gImgInput = gImgSharpgImgMedian.astype('uint8')
						
					# allContours = (myCythonFunc.gradient(gImgInput.astype('double'), exclude)).astype('bool')
					# for contourMethod in contourMethodList:
						# denseContours = (myCythonFunc.cleanContour(allContours.astype('uint8'), contourMethod, minNeighbors)).astype('bool')
						# bImgFill = ndimage.binary_fill_holes(denseContours)
						# bImgInner = bImgFill
						# # bImgInner = (myCythonFunc.removeBoundaryParticles(bImgFill.astype('uint8'))).astype('bool')
						# bImgBig = myCythonFunc.areaThreshold(bImgInner.astype('uint8'), minArea=areaRange[0], maxArea=areaRange[1]).astype('bool')
						# bImgOpen = myPythonFunc.binary_opening(bImgBig, iterations=2*noiseSize)
						# bImg = myCythonFunc.areaThreshold(bImgOpen.astype('uint8'), minArea=areaRange[0], maxArea=areaRange[1])
						# bImgCircular = myCythonFunc.circularThreshold(bImg.astype('uint8'), minCircular).astype('bool')
						# bImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(bImgCircular))
						
						# labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImgCircular, gImg, centroid=True)
						# bImgTagged = myPythonFunc.normalize(bImgCircular)
						# for j in range(len(dictionary['id'])):
							# cv2.putText(bImgTagged, str(dictionary['id'][j]), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=1, bottomLeftOrigin=False)
							
						# finalImage = numpy.row_stack((numpy.column_stack((gImgInput, myPythonFunc.normalize(allContours), myPythonFunc.normalize(denseContours), myPythonFunc.normalize(bImgFill), myPythonFunc.normalize(bImgInner))), numpy.column_stack((myPythonFunc.normalize(bImgBig), myPythonFunc.normalize(bImgOpen), myPythonFunc.normalize(bImgCircular), bImgTagged, numpy.maximum(gImg, bImgBdry)))))	
						# cv2.imwrite(imgDirName+'/'+contourMethod+'/'+rawFilter+'/'+str(i).zfill(4)+'.tiff', finalImage)
			# del gImgSharpStack
	# elif (method == 'gauss'):
		# for noiseSize in noiseSizeList:
			# datDirName = inputDir+'/output/data/particleIdentification/'+method+'/'+str(noiseSize).zfill(2)
			# imgDirName = inputDir+'/output/images/particleIdentification/'+method+'/'+str(noiseSize).zfill(2)
			# gImgBlurStack = numpy.load(datDirName+'/gImgBlurStack.npy')
			# for contourMethod in contourMethodList:
				# if (os.path.exists(imgDirName+'/'+contourMethod) == False):
					# os.makedirs(imgDirName+'/'+contourMethod)
				# for rawFilter in rawFilterList:
					# if (os.path.exists(imgDirName+'/'+contourMethod+'/'+rawFilter) == False):
						# os.makedirs(imgDirName+'/'+contourMethod+'/'+rawFilter)
			# for i in frameList:
				# print method, noiseSize, i
				# for rawFilter in rawFilterList:
					# gImg = gImgStack[:,:,i-1]
					# gImgBlur = gImgBlurStack[:,:,i-1]
					# gImgBlurMedian = (myPythonFunc.adaptiveThreshold(gImgBlur, method='median', numPartitions=thresholdPartitions))*gImgBlur
					
					# if (rawFilter == 'none'):
						# gImgInput = gImgBlurMedian.astype('uint8')
					# elif (rawFilter == 'median'):
						# gImgBlur2 = ndimage.gaussian_filter(gImg, sigma=2)
						# gImgMedian = (myPythonFunc.adaptiveThreshold(gImgBlur2, method='median', numPartitions=thresholdPartitions)).astype('uint8')
						# gImgBlurgImgMedian = gImgBlurMedian*gImgMedian
						# gImgInput = gImgBlurgImgMedian.astype('uint8')
					
					# allContours = (myCythonFunc.gradient(gImgInput.astype('double'), exclude)).astype('bool')
					# for contourMethod in contourMethodList:
						# denseContours = (myCythonFunc.cleanContour(allContours.astype('uint8'), contourMethod, minNeighbors)).astype('bool')
						# bImgFill = ndimage.binary_fill_holes(denseContours)
						# bImgInner = bImgFill
						# # bImgInner = (myCythonFunc.removeBoundaryParticles(bImgFill.astype('uint8'))).astype('bool')
						# bImgBig = myCythonFunc.areaThreshold(bImgInner.astype('uint8'), minArea=areaRange[0], maxArea=areaRange[1]).astype('bool')
						# bImgOpen = myPythonFunc.binary_opening(bImgBig, iterations=2*noiseSize)
						# bImg = myCythonFunc.areaThreshold(bImgOpen.astype('uint8'), minArea=areaRange[0], maxArea=areaRange[1])
						# bImgCircular = myCythonFunc.circularThreshold(bImg.astype('uint8'), minCircular).astype('bool')
						# bImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(bImgCircular))
						
						# labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImgCircular, gImg, centroid=True)
						# bImgTagged = myPythonFunc.normalize(bImgCircular)
						# for j in range(len(dictionary['id'])):
							# cv2.putText(bImgTagged, str(dictionary['id'][j]), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=1, bottomLeftOrigin=False)
						
						# finalImage = numpy.row_stack((numpy.column_stack((gImgInput, myPythonFunc.normalize(allContours), myPythonFunc.normalize(denseContours), myPythonFunc.normalize(bImgFill), myPythonFunc.normalize(bImgInner))), numpy.column_stack((myPythonFunc.normalize(bImgBig), myPythonFunc.normalize(bImgOpen), myPythonFunc.normalize(bImgCircular), bImgTagged, numpy.maximum(gImg, bImgBdry)))))
						# cv2.imwrite(imgDirName+'/'+contourMethod+'/'+rawFilter+'/'+str(i).zfill(4)+'.tiff', finalImage)
			# del gImgBlurStack
##############################################################
##############################################################



##############################################################
##############################################################
#PERFORMAING CONDITIONAL INTERSECTION OF IMAGES USING TWO DIFFERENT TECHNIQUES
##############################################################
##############################################################
# print "PERFORMAING CONDITIONAL INTERSECTION OF IMAGES USING TWO DIFFERENT TECHNIQUES"
# gImgRawStack = numpy.load(inputDir+'/output/data/gImgRawStack.npy')
# [row, col, numFrames] = gImgRawStack.shape
# blank = numpy.zeros([row, col], dtype='uint8')
# if (os.path.exists(inputDir+'/output/images/particleIdentification/conditionalIntersect') == False):
	# os.makedirs(inputDir+'/output/images/particleIdentification/conditionalIntersect')

# dir1 = inputDir+'/output/images/particleIdentification/medianOFgauss/04_060/neighbors/none'
# dir2 = inputDir+'/output/images/particleIdentification/medianOFgauss/04_060/none/none'
# frameList = range(1, numFrames+1)

# for i in frameList:
	# print "Processing frame", i
	# fileName1 = dir1+'/'+str(i).zfill(4)+'.tiff'
	# fileName2 = dir2+'/'+str(i).zfill(4)+'.tiff'
	# img1 = cv2.imread(fileName1, 0); img2 = cv2.imread(fileName2, 0);
	# bImg1 = img1[row:2*row, 2*col:3*col].astype('bool'); bImg2 = img2[row:2*row, 2*col:3*col].astype('bool')
	# bImg = myPythonFunc.conditionalDivision(bImg1, bImg2)
	# bImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(bImg))
	# gImg = gImgRawStack[:,:,i-1]
	
	# labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg1, gImg, centroid=True)
	# bImg1 = myPythonFunc.normalize(bImg1)
	# for j in range(len(dictionary['id'])):
		# cv2.putText(bImg1, str(dictionary['id'][j]), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=1, bottomLeftOrigin=False)
	# labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg2, gImg, centroid=True)
	# bImg2 = myPythonFunc.normalize(bImg2)
	# for j in range(len(dictionary['id'])):
		# cv2.putText(bImg2, str(dictionary['id'][j]), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=1, bottomLeftOrigin=False)
	# labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, centroid=True)
	# bImg = myPythonFunc.normalize(bImg)
	# bImgID = bImg.copy()
	# for j in range(len(dictionary['id'])):
		# cv2.putText(bImgID, str(dictionary['id'][j]), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=1, bottomLeftOrigin=False)
		
	# finalImage = numpy.row_stack((numpy.column_stack((bImg1, bImg2, blank, blank, blank)), numpy.column_stack((blank, blank, bImg, bImgID, numpy.maximum(gImg,bImgBdry)))))
	# cv2.imwrite(inputDir+'/output/images/particleIdentification/conditionalIntersect/'+str(i).zfill(4)+'.tiff', finalImage)
##############################################################
##############################################################



##############################################################
##############################################################
#PERFORMING SEGMENTATION FOR MANUALLY EDITED CONTOUR LINES
##############################################################
##############################################################
# print "PERFORMING SEGMENTATION FOR MANUALLY EDITED CONTOUR LINES"
# gImgStack = numpy.load(inputDir+'/output/data/gImgStack.npy')
# [row, col, numFrames] = gImgStack.shape

# frameList = range(1,114)#[43,49,83,119,132,133,134,135,136,156,160,164,166,174,175]
# areaRange = [22500,300000]
# minCircular = 0.0
# noiseSize = 4

# for i in frameList:
	# fileName = inputDir+'/output/images/particleIdentification/manual/'+str(i).zfill(4)+'.tiff'
	# img = cv2.imread(fileName, 0)
	
	# gImg = gImgStack[:,:,i-1]
	# gImgInput = img[0:row,0:col]
	# allContours = img[0:row,col:2*col]
	# denseContours = img[0:row,2*col:3*col]
	# bImgFill = ndimage.binary_fill_holes(denseContours)
	# bImgInner = bImgFill
	# # bImgInner = (myCythonFunc.removeBoundaryParticles(bImgFill.astype('uint8'))).astype('bool')
	# bImgBig = myCythonFunc.areaThreshold(bImgInner.astype('uint8'), minArea=areaRange[0], maxArea=areaRange[1]).astype('bool')
	# bImgOpen = myPythonFunc.binary_opening(bImgBig, iterations=2*noiseSize)
	# bImg = myCythonFunc.areaThreshold(bImgOpen.astype('uint8'), minArea=areaRange[0], maxArea=areaRange[1])
	# bImgCircular = myCythonFunc.circularThreshold(bImg.astype('uint8'), minCircular).astype('bool')
	# bImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(bImgCircular))
	
	# labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImgCircular, gImg, centroid=True)
	# bImgTagged = myPythonFunc.normalize(bImgCircular)
	# for j in range(len(dictionary['id'])):
		# cv2.putText(bImgTagged, str(dictionary['id'][j]), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=1, bottomLeftOrigin=False)
		
	# finalImage = numpy.row_stack((numpy.column_stack((gImgInput, myPythonFunc.normalize(allContours), myPythonFunc.normalize(denseContours), myPythonFunc.normalize(bImgFill), myPythonFunc.normalize(bImgInner))), numpy.column_stack((myPythonFunc.normalize(bImgBig), myPythonFunc.normalize(bImgOpen), myPythonFunc.normalize(bImgCircular), bImgTagged, numpy.maximum(gImg, bImgBdry)))))
	# cv2.imwrite(fileName, finalImage)
##############################################################
##############################################################



##############################################################
##############################################################
#CREATING BINARY IMAGE STACK FROM FINAL SEGMENTATION
##############################################################
##############################################################
# print 'CREATING bImgStack'
# gImgRawStack = numpy.load(inputDir+'/output/data/gImgRawStack.npy')
# [row, col, numFrames] = gImgRawStack.shape
# bImgStack = numpy.zeros([row, col, numFrames], dtype='bool')

# frameList = range(1, numFrames+1)

# for i in frameList:
	# fileName = inputDir+'/output/images/particleIdentification/final/'+str(i).zfill(4)+'.tiff'
	# img = cv2.imread(fileName, 0)
	# bImg = img[row:2*row, 2*col:3*col].astype('bool')
	# bImgStack[:,:,i-1] = bImg
# numpy.save(inputDir+'/output/data/particleIdentification/bImgStack.npy', bImgStack)
##############################################################
##############################################################



##############################################################
##############################################################
#CREATING SMOOTH BINARY STACK
##############################################################
##############################################################
# print "CREATING SMOOTH BINARY STACK"
# gImgRawStack = numpy.load(inputDir+'/output/data/gImgRawStack.npy')
# bImgStack = (myPythonFunc.normalize(numpy.load(inputDir+'/output/data/particleIdentification/bImgStack.npy'))).astype('double')
# [row, col, numFrames] = gImgRawStack.shape
# if (os.path.exists(inputDir+'/output/images/particleIdentification/smooth') == False):
	# os.makedirs(inputDir+'/output/images/particleIdentification/smooth')

# bImgSmoothStack = numpy.zeros([row, col, numFrames])

# areaRange = [200, 300000]
# frameList = range(1, numFrames+1)
# gaussBlurSize = 4

# # bImgSmoothStack = bImgStack
# # bImgSmoothStack1 = ndimage.gaussian_filter1d(bImgStack[:,:,0:338], sigma=gaussBlurSize, axis=2); bImgSmoothStack1 = myPythonFunc.normalize(bImgSmoothStack1)
# # bImgSmoothStack2 = ndimage.gaussian_filter1d(bImgStack[:,:,338:498], sigma=gaussBlurSize, axis=2); bImgSmoothStack2 = myPythonFunc.normalize(bImgSmoothStack2)
# # bImgSmoothStack3 = ndimage.gaussian_filter1d(bImgStack[:,:,498:numFrames], sigma=gaussBlurSize, axis=2); bImgSmoothStack3 = myPythonFunc.normalize(bImgSmoothStack3)

# # bImgSmoothStack1 = bImgSmoothStack1 >= 100; bImgSmoothStack2 = bImgSmoothStack2 >= 100; bImgSmoothStack3 = bImgSmoothStack3 >= 100
# # bImgSmoothStack[:,:,0:338] = bImgSmoothStack1; bImgSmoothStack[:,:,338:498] = bImgSmoothStack2; bImgSmoothStack[:,:,498:numFrames] = bImgSmoothStack3

# bImgSmoothStack = ndimage.gaussian_filter1d(bImgStack, sigma=gaussBlurSize, axis=2)
# bImgSmoothStack = myPythonFunc.normalize(bImgSmoothStack)
# bImgSmoothStack = bImgSmoothStack >= 150#myPythonFunc.otsuThreshold(bImgSmoothStack, bins=255, range=(1,255))
# for i in frameList:
	# bImg = myCythonFunc.areaThreshold(bImgSmoothStack[:,:,i-1].astype('uint8'), minArea=areaRange[0], maxArea=areaRange[1]).astype('bool')
	# bImg = ndimage.binary_fill_holes(bImg)
	# bimg = myPythonFunc.binary_opening(bImg, iterations=gaussBlurSize)
	# bImgSmoothStack[:,:,i-1] = bImg
	# bImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(bImg))
	# finalImage = numpy.column_stack((myPythonFunc.normalize(bImg), numpy.maximum(gImgRawStack[:,:,i-1],bImgBdry)))
	# cv2.imwrite(inputDir+'/output/images/particleIdentification/smooth/'+str(i).zfill(4)+'.tiff', finalImage)
# numpy.save(inputDir+'/output/data/particleIdentification/bImgSmoothStack.npy', bImgSmoothStack)
##############################################################
##############################################################



##############################################################
##############################################################
#LABELLING PARTICLES
##############################################################
##############################################################
# print "LABELLING PARTICLES"
# gImgRawStack = numpy.load(inputDir+'/output/data/gImgRawStack.npy')
# bImgSmoothStack = numpy.load(inputDir+'/output/data/particleIdentification/bImgSmoothStack.npy')
# [row, col, numFrames] = gImgRawStack.shape

# centerDispTh = 10
# perctAreaChangeTh = 0.2
# missFramesTh = 10
# frameList = range(1, numFrames+1)

# labelStack = numpy.zeros([row, col, numFrames], dtype='int32')

# for i in frameList:
	# if (i == frameList[0]):
		# labelImg_0, numLabel_0, dictionary_0 = myPythonFunc.regionProps(bImgSmoothStack[:,:,i-1], gImgRawStack[:,:,i-1], centroid=True, area=True)
		# maxID = numLabel_0
		# dictionary_0['frame'] = []
		# for j in range(len(dictionary_0['id'])):
			# dictionary_0['frame'].append(i)
		# labelStack[:,:,i-1] = labelImg_0
	# else:
		# labelImg_1, numLabel_1, dictionary_1 = myPythonFunc.regionProps(bImgSmoothStack[:,:,i-1], gImgRawStack[:,:,i-1], centroid=True, area=True)
		# for j in range(len(dictionary_1['id'])):
			# flag = 0
			# bImg_1_LabelN = labelImg_1==dictionary_1['id'][j]
			# center_1 = dictionary_1['centroid'][j]
			# area_1 = dictionary_1['area'][j]
			# frame_1 = i
			# for k in range(len(dictionary_0['id'])):
				# center_0 = dictionary_0['centroid'][k]
				# area_0 = dictionary_0['area'][k]
				# frame_0 = dictionary_0['frame'][k]
				# if ((sqrt((center_1[0]-center_0[0])**2. + (center_1[1]-center_0[1])**2.) <= centerDispTh) and (1.*area_0/area_1 >= (1-perctAreaChangeTh) and 1.*area_0/area_1 <= (1+perctAreaChangeTh)) and (frame_1-frame_0 <= missFramesTh)):
					# labelStack[:,:,i-1] += (bImg_1_LabelN*dictionary_0['id'][k])
					# dictionary_0['centroid'][k] = center_1
					# dictionary_0['area'][k] = area_1
					# dictionary_0['frame'][k] = i
					# flag = 1
					# break
			# if (flag == 0):
				# maxID += 1
				# labelN_1 = bImg_1_LabelN*maxID
				# labelStack[:,:,i-1] += labelN_1
				# dictionary_0['id'].append(maxID)
				# dictionary_0['centroid'].append(center_1)
				# dictionary_0['area'].append(area_1)
				# dictionary_0['frame'].append(i)
# numpy.save(inputDir+'/output/data/particleIdentification/labelStack.npy', labelStack)
##############################################################
##############################################################



##############################################################
##############################################################
#PRINTING ORIGINAL LABELLED PARTICLES
##############################################################
##############################################################
# print "PRINTING ORIGINAL LABELLED PARTICLES"
# bImgSmoothStack = numpy.load(inputDir+'/output/data/particleIdentification/bImgSmoothStack.npy')
# gImgRawStack = numpy.load(inputDir+'/output/data/gImgRawStack.npy')
# labelStack = numpy.load(inputDir+'/output/data/particleIdentification/labelStack.npy')
# [row, col, numFrames] = gImgRawStack.shape
# if (os.path.exists(inputDir+'/output/images/particleIdentification/labels') == False):
	# os.makedirs(inputDir+'/output/images/particleIdentification/labels')
	
# frameList = range(1, numFrames+1)

# for i in frameList:
	# bImg = bImgSmoothStack[:,:,i-1]
	# bImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(bImg))
	# gImg = gImgRawStack[:,:,i-1]
	# labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, centroid=True)
	# bImg = myPythonFunc.normalize(bImg)
	# for j in range(len(dictionary['id'])):
		# bImgLabelN = labelImg == dictionary['id'][j]
		# ID = numpy.max(bImgLabelN*labelStack[:,:,i-1])
		# cv2.putText(bImg, str(ID), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=1, bottomLeftOrigin=False)
	# finalImage = numpy.column_stack((bImg, numpy.maximum(gImg,bImgBdry)))
	# cv2.imwrite(inputDir+'/output/images/particleIdentification/labels/'+str(i).zfill(4)+'.tiff', finalImage)
##############################################################
##############################################################
	
	

##############################################################
##############################################################
#PRINTING NUMBER OF SOLIDS FOR EACH LABEL
##############################################################
##############################################################
# print "PRINTING NUMBER OF SOLIDS FOR EACH LABEL"
# labelStack = numpy.load(inputDir+'/output/data/particleIdentification/labelStack.npy')
# numParticles = labelStack.max()
# for i in range(1,numParticles+1):
	# bImgStackLabelN = labelStack==i
	# [labelImg, numLabel] = ndimage.label(bImgStackLabelN)
	# print "Number of volumes of particle ", i, "=", numLabel
##############################################################
##############################################################



##############################################################
##############################################################
#CALCULATING VOLUME OF EACH PARTICLE
##############################################################
##############################################################
# print "CALCULATING VOLUME OF EACH PARTICLE"
# labelStack = numpy.load(inputDir+'/output/data/particleIdentification/labelStack.npy')
# numParticles = labelStack.max()
# for i in range(1,numParticles+1):
	# bImgStackLabelN = labelStack==i
	# volume = bImgStackLabelN.sum()
	# print "Volume of particle", i, "=", volume
##############################################################
##############################################################



##############################################################
##############################################################
#SMOOTHING STACK BASED ON LABELS
##############################################################
##############################################################
# print "SMOOTHING STACK BASED ON LABELS"
# gImgRawStack = numpy.load(inputDir+'/output/data/gImgRawStack.npy')
# labelStack = numpy.load(inputDir+'/output/data/particleIdentification/labelStack.npy')
# [row, col, numFrames] = gImgRawStack.shape
# bImgSmoothStack = numpy.zeros([row,col,numFrames], dtype='bool')
# if (os.path.exists(inputDir+'/output/images/particleIdentification/smooth') == False):
	# os.makedirs(inputDir+'/output/images/particleIdentification/smooth')

# particleList = [1,2,4,5,6,7,8,9,13,16,18,19,24,26,30,37,40,44,45]
# frameList = range(1,numFrames+1)
# gaussBlurSize = 2
# areaRange = [200, 125000]

# for particle in particleList:
	# print particle
	# bImgStackN = labelStack==particle
	# bImgStackN = myPythonFunc.normalize(bImgStackN).astype('double')
	# [r,c,frames] = numpy.nonzero(bImgStackN)
	# bImgSmoothStackN = ndimage.gaussian_filter1d(bImgStackN, sigma=gaussBlurSize, axis=2)
	# bImgSmoothStackN = myPythonFunc.normalize(bImgSmoothStackN)
	# bImgSmoothStackN = bImgSmoothStackN>=150
	# bImgSmoothStack[:,:,frames.min():frames.max()+1] = numpy.logical_or(bImgSmoothStackN[:,:,frames.min():frames.max()+1], bImgSmoothStack[:,:,frames.min():frames.max()+1])
	
# for frame in frameList:
	# bImg = myCythonFunc.areaThreshold(bImgSmoothStack[:,:,frame-1].astype('uint8'), minArea=areaRange[0], maxArea=areaRange[1]).astype('bool')
	# bImg = ndimage.binary_fill_holes(bImg)
	# bimg = myPythonFunc.binary_opening(bImg, iterations=gaussBlurSize)
	# bImgSmoothStack[:,:,frame-1] = bImg
	# bImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(bImg))
	# finalImage = numpy.column_stack((myPythonFunc.normalize(bImg), numpy.maximum(gImgRawStack[:,:,frame-1],bImgBdry)))
	# cv2.imwrite(inputDir+'/output/images/particleIdentification/smooth/'+str(frame).zfill(4)+'.tiff', finalImage)
# numpy.save(inputDir+'/output/data/particleIdentification/bImgSmoothStack.npy', bImgSmoothStack)
##############################################################
##############################################################



##############################################################
##############################################################
#REMOVING UNWANTED PARTICLES
##############################################################
##############################################################
# print "REMOVING UNWANTED PARTICLES"
# labelStack = numpy.load(inputDir+'/output/data/particleIdentification/labelStack.npy')

# particleList = [3,5,6,7,8,9,10]

# for i in particleList:
	# labelStack[labelStack==i] = 0
# bImgSmoothStack = labelStack.astype('bool')
# numpy.save(inputDir+'/output/data/particleIdentification/bImgSmoothStack.npy', bImgSmoothStack)
# numpy.save(inputDir+'/output/data/particleIdentification/labelStack.npy', labelStack)
##############################################################
##############################################################



##############################################################
##############################################################
#LABEL CORRECTION
##############################################################
##############################################################
# print "LABEL CORRECTION"
# labelStack = numpy.load(inputDir+'/output/data/particleIdentification/labelStack.npy')

# correctionList = [[5,3],[7,3],[9,3],[10,3],[6,4],[8,4],[11,4],[12,5],[13,5],[14,5],[15,5],[16,5]]

# for i in range(len(correctionList)):
	# labelStack[labelStack==correctionList[i][0]] = correctionList[i][1]
# numpy.save(inputDir+'/output/data/particleIdentification/labelStack.npy', labelStack)
##############################################################
##############################################################



##############################################################
##############################################################
#PRINTING UPDATED LABELLED PARTICLES
##############################################################
##############################################################
# print "PRINTING UPDATED LABELLED PARTICLES"
# bImgSmoothStack = numpy.load(inputDir+'/output/data/particleIdentification/bImgSmoothStack.npy')
# gImgRawStack = numpy.load(inputDir+'/output/data/gImgRawStack.npy')
# labelStack = numpy.load(inputDir+'/output/data/particleIdentification/labelStack.npy')
# [row, col, numFrames] = gImgRawStack.shape
# if (os.path.exists(inputDir+'/output/images/particleIdentification/labels') == False):
	# os.makedirs(inputDir+'/output/images/particleIdentification/labels')
	
# frameList = range(1, numFrames+1)

# for i in frameList:
	# bImg = bImgSmoothStack[:,:,i-1]
	# bImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(bImg))
	# gImg = gImgRawStack[:,:,i-1]
	# labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, centroid=True)
	# bImg = myPythonFunc.normalize(bImg)
	# for j in range(len(dictionary['id'])):
		# bImgLabelN = labelImg == dictionary['id'][j]
		# ID = numpy.max(bImgLabelN*labelStack[:,:,i-1])
		# cv2.putText(bImg, str(ID), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=1, bottomLeftOrigin=False)
	# finalImage = numpy.column_stack((bImg, numpy.maximum(gImg,bImgBdry)))
	# cv2.imwrite(inputDir+'/output/images/particleIdentification/labels/'+str(i).zfill(4)+'.tiff', finalImage)
##############################################################
##############################################################



##############################################################
##############################################################
#FINDING MEASURES FOR TRACKED PARTICLES
##############################################################
##############################################################
# print "FINDING MEASURES FOR TRACKED PARTICLES"
# gImgRawStack = numpy.load(inputDir+'/output/data/gImgRawStack.npy')
# labelStack = numpy.load(inputDir+'/output/data/particleIdentification/labelStack.npy')
# outFile = open(inputDir+'/output/data/particleIdentification/imgData.dat', 'wb')
# [row, col, numFrames] = gImgRawStack.shape
# frameList = range(1, numFrames+1)
# particleList = range(1, labelStack.max()+1)

# area=True
# perimeter=True
# circularity=True
# pixelList=False
# bdryPixelList = True
# centroid=True
# intensityList=False
# sumIntensity=False
# effRadius=True
# radius=False
# circumRadius=False
# inRadius=False

# imgDataDict = {}
# for particle in particleList:
	# imgDataDict[particle]={}

# for i in frameList:
	# label = labelStack[:,:,i-1]
	# gImg = gImgRawStack[:,:,i-1]
	# outFile.write("%d," %(i))
	# for j in particleList:
		# bImg = label==j
		# if (bImg.max() == True):
			# labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, centroid=True, area=True, perimeter=True, circularity=True, bdryPixelList=True, effRadius=True)
			# outFile.write("%d,%f,%f,%d,%f,%f,%f," %(j, dictionary['centroid'][0][1], row-dictionary['centroid'][0][0], dictionary['area'][0], dictionary['perimeter'][0], dictionary['circularity'][0], dictionary['effRadius'][0]))
			# imgDataDict[j][i]={}; imgDataDict[j][i]=dictionary
		# else:
			# outFile.write("%d,nan,nan,nan,nan,nan,nan," %(j))
	# outFile.write("\n")
# pickle.dump(imgDataDict, open(inputDir+'/output/data/particleIdentification/imgDataDict.dat', 'wb'))
# outFile.close()
##############################################################
##############################################################



##############################################################
##############################################################
#CALCULATE THE MINIMUM AND MAXIMUM DISTANCE BETWEEN ALL PARTICLES FOR ALL TIME FRAMES
##############################################################
##############################################################
# print "CALCULATE THE MINIMUM AND MAXIMUM DISTANCE BETWEEN ALL PARTICLES FOR ALL TIME FRAMES"
# gImgRawStack = numpy.load(inputDir+'/output/data/gImgRawStack.npy')
# labelStack = numpy.load(inputDir+'/output/data/particleIdentification/labelStack.npy')
# [row,col,numFrames] = gImgRawStack.shape
# imgDataDict = pickle.load(open(inputDir+'/output/data/particleIdentification/imgDataDict.dat','rb'))
# outFile = open(inputDir+'/output/data/particleIdentification/distData.dat', 'wb')

# frameList = range(1,numFrames+1)
# particleList = range(1, labelStack.max()+1)

# dMinArray = numpy.zeros([len(particleList),len(particleList),numFrames]); dMaxArray = numpy.zeros([len(particleList),len(particleList),numFrames])
# dMinArray[:,:,:] = numpy.nan; dMaxArray[:,:,:] = numpy.nan

# for frame in frameList:
	# for i in range(len(particleList)):
		# label_i = labelStack[:,:,frame-1]==particleList[i]
		# if (label_i.max() > 0):
			# r1=numpy.array(imgDataDict[particleList[i]][frame]['bdryPixelList'][0][0]).astype('int32'); c1=numpy.array(imgDataDict[particleList[i]][frame]['bdryPixelList'][0][1]).astype('int32')
			# for j in range(len(particleList)):
				# label_j = labelStack[:,:,frame-1]==particleList[j]
				# if (label_j.max() > 0):
					# r2=numpy.array(imgDataDict[particleList[j]][frame]['bdryPixelList'][0][0]).astype('int32'); c2=numpy.array(imgDataDict[particleList[j]][frame]['bdryPixelList'][0][1]).astype('int32')
					# [dMin, dMax] = myCythonFunc.calculateMinMaxDist(r1,c1,r2,c2)
					# dMinArray[i][j][frame-1]=dMin; dMaxArray[i][j][frame-1]=dMax
					# dMinArray[j][i][frame-1]=dMin; dMaxArray[j][i][frame-1]=dMax
					
# for i in range(len(particleList)-1):
	# for j in range(i+1,len(particleList)):
		# outFile.write("%s," %(str(particleList[i])+'_'+str(particleList[j])))
		# for frame in frameList:
			# outFile.write("%f," %(dMinArray[i][j][frame-1]))
		# outFile.write("\n")
# outFile.close()
##############################################################
##############################################################



##############################################################
##############################################################
#CALCULATING ORDERED PARAMETER FOR THE ENTIRE IMAGE STACK
##############################################################
##############################################################
# print "CALCULATING ORDERED PARAMETER FOR THE ENTIRE IMAGE STACK"
# gImgRawStack = numpy.load(inputDir+'/output/data/gImgRawStack.npy')
# labelStack = numpy.load(inputDir+'/output/data/particleIdentification/labelStack.npy')
# [row, col, numFrames] = gImgRawStack.shape
# numParticles = labelStack.max()

# particleList = range(1,numParticles+1)
# gaussBlurSize = 4
# gamma = 0.4
# ringRange = [1.5, 6]
# step_size = 10
# # pixInAngstrom = 0.4123046875
# pixInAngstrom = 0.3458984375

# mask = ndimage.gaussian_filter((labelStack.astype('bool')).astype('double'), sigma=gaussBlurSize); mask = (mask-mask.min())/(mask.max()-mask.min())
# gImg = (gImgRawStack*mask).astype('uint8')
# [ordParamArr, qArrScale] = myPythonFunc.processAllFrames(gImg)
# if (os.path.exists(inputDir+'/output/kymograph') == False):
	# os.makedirs(inputDir+'/output/kymograph')
# outputDir = inputDir+'/output/kymograph'
# numpy.save(inputDir+'/output/kymograph/ordParamArr.npy', ordParamArr)
# myPythonFunc.plotOrderParam(outputDir, ordParamArr, pixInAngstrom, qArrScale, gamma=gamma, ringRange=ringRange, step_size=step_size)

# for i in particleList:
	# print "Processing Particle", i
	# mask = ndimage.gaussian_filter((labelStack == i).astype('double'), sigma=gaussBlurSize); mask = (mask-mask.min())/(mask.max()-mask.min())
	# gImgLabelN = (gImgRawStack*mask).astype('uint8')
	# [ordParamArr, qArrScale] = myPythonFunc.processAllFrames(gImgLabelN)
	# if (os.path.exists(inputDir+'/output/kymograph/'+str(i).zfill(2)) == False):
		# os.makedirs(inputDir+'/output/kymograph/'+str(i).zfill(2))
	# outputDir = inputDir+'/output/kymograph/'+str(i).zfill(2)
	# numpy.save(inputDir+'/output/kymograph/ordParamArr_'+str(i).zfill(2)+'.npy', ordParamArr)
	# myPythonFunc.plotOrderParam(outputDir, ordParamArr, pixInAngstrom, qArrScale, gamma=gamma, ringRange=ringRange, step_size=step_size)
##############################################################
##############################################################



##############################################################
##############################################################
#IDENTIFICATION OF BRAGG DIFFRACTION PEAKS
##############################################################
##############################################################
# print "IDENTIFICATION OF BRAGG DIFFRACTION PEAKS"
# gImgRawStack = numpy.load(inputDir+'/output/data/gImgRawStack.npy')
# labelStack = numpy.load(inputDir+'/output/data/particleIdentification/labelStack.npy')
# [row, col, numFrames] = gImgRawStack.shape
# numParticles = labelStack.max()

# frameList = range(1,numFrames+1)
# particleList = range(1,numParticles+1)
# gaussBlurSize = 4
# cutoff = 2
# areaRange = [5,200]
# ringRange = [[101,126]]
# circleRadius = 1

# if (os.path.exists(inputDir+'/output/phaseIdentification') == False):
	# os.makedirs(inputDir+'/output/phaseIdentification')
# if (os.path.exists(inputDir+'/output/phaseIdentification/filter') == False):
	# os.makedirs(inputDir+'/output/phaseIdentification/filter')
# for i in particleList:
	# print "Processing particle", i
	# ordParamArr = numpy.load(inputDir+'/output/kymograph/ordParamArr_'+str(i).zfill(2)+'.npy')
	# if (os.path.exists(inputDir+'/output/phaseIdentification/filter/'+str(i).zfill(2)) == False):
		# os.makedirs(inputDir+'/output/phaseIdentification/filter/'+str(i).zfill(2))
	# outputDir = inputDir+'/output/phaseIdentification/filter/'+str(i).zfill(2)
	# [qArr, qArrScale] = myPythonFunc.calculate_qArr(gImgRawStack[:,:,0])
	# [qTcks, rTcks] = myPythonFunc.calculate_qrTcks(ordParamArr, pixInAngstrom, qArrScale)
	# for j in frameList:
		# print "Processing Frame", j
		# mask = (labelStack[:,:,j-1] == i).astype('double')
		# if (mask.max() > 0):
			# mask = ndimage.gaussian_filter(mask, sigma=gaussBlurSize); mask = (mask-mask.min())/(mask.max()-mask.min())
			# gImgLabelN = (gImgRawStack[:,:,j-1]*mask).astype('uint8')
			# finalImage = myPythonFunc.phaseIdentification(gImgLabelN, qArr, rTcks, cutoff, areaRange=areaRange, ringRange=ringRange, circleRadius=circleRadius, rDist=2.40, thetaThresh=5)
			# cv2.imwrite(outputDir+'/'+str(j).zfill(4)+'.tiff', finalImage)
##############################################################
##############################################################



##############################################################
##############################################################
#SELECTING VALID PEAKS
##############################################################
##############################################################
# print "SELECTING VALID PEAKS"
# gImgRawStack = numpy.load(inputDir+'/output/data/gImgRawStack.npy')
# labelStack = numpy.load(inputDir+'/output/data/particleIdentification/labelStack.npy')
# [row, col, numFrames] = gImgRawStack.shape
# numParticles = labelStack.max()

# frameList = range(1,numFrames+1)
# particleList = range(1,numParticles+1)
# rTickList = [2.04,2.356]
# areaRange = [10,200]

# if (os.path.exists(inputDir+'/output/phaseIdentification') == False):
	# os.makedirs(inputDir+'/output/phaseIdentification')
# if (os.path.exists(inputDir+'/output/phaseIdentification/final') == False):
	# os.makedirs(inputDir+'/output/phaseIdentification/final')

# for i in particleList:
	# print "Processing particle", i
	# for j in frameList:
		# fileName = inputDir+'/output/phaseIdentification/filter/'+str(i).zfill(2)+'/'+str(j).zfill(4)+'.tiff'
		# if os.path.isfile(filename):
			# img = cv2.imread(fileName, 0)
			# bImg = img[row:2*row, col:2*col]
			# gImg = img[0:row, col:2*col]
			
			# [qArr, qArrScale] = myPythonFunc.calculate_qArr(gImg)
			# [qTcks, rTcks] = myPythonFunc.calculate_qrTcks(ordParamArr, pixInAngstrom, qArrScale)
			# bImg = myCythonFunc.areaThreshold(bImg, minArea=areaRange[0], maxArea=areaRange[1])
			
			# [labelImg, numLabel, dictionary] = myPythonFunc.regionProps(bImg.astype('bool'), fImg, rTcks, rTick=True)
			# for k in range(1, numLabel+1):
				# flag = 0
				# for rTick in rTickList:
					# if (dictionary[k]['rTick'][0] >= rTick+0.5 and dictionary[k]['rTick'][0] <= rTick-0.5):
						# flag = 1
				# if (flag == 0):
					# labelImg[labelImg==k] = 0
					# bImg[labelImg==k] = 0
			
			# [labelImg, numLabel, dictionary] = myPythonFunc.regionProps(bImg.astype('bool'), fImg, rTcks, theta=True)
			# for k in range(1, numLabel+1):
				# flag = 0
				# for l in range(1, numLabel+1):
					# dTheta = numpy.abs((dictionary[k]['theta'][0]-dictionary[l]['theta'][0]+180)%360-180)
					# if (dTheta >= 180-thetaThreshold):
						# flag = 1
						# break
				# if (flag == 0):
					# labelImg[labelImg==k] = 0
					# bImg[labelImg==k] = 0
			# fileName = inputDir+'/output/phaseIdentification/final/'+str(i).zfill(2)+'/'+str(j).zfill(4)+'.tiff'
			# finalImage = numpy.column_stack((bImg, numpy.maximum(gImg,bImg)))
			# cv2.imwrite(fileName, finalImage)
##############################################################
##############################################################



##############################################################
##############################################################
#GENERATING MEASURES FOR PHASE IDENTIFICATION
##############################################################
##############################################################
print "GENERATING MEASURES FOR PHASE IDENTIFICATION"
gImgRawStack = numpy.load(inputDir+'/output/data/gImgRawStack.npy')
labelStack = numpy.load(inputDir+'/output/data/particleIdentification/labelStack.npy')
[row, col, numFrames] = gImgRawStack.shape	
numParticles = labelStack.max()
phaseDict = {}

frameList = range(1,numFrames+1)
particleList = range(1,numParticles+1)

dirName = inputDir+'/output/phaseIdentification/filter'
outFile = open(inputDir+'/output/phaseIdentification/filter/angles.dat', 'wb')
for i in particleList:
	print "Processing Particle", i
	phaseDict[i] = {}
	ordParamArr = numpy.load(inputDir+'/output/kymograph/ordParamArr_'+str(i).zfill(2)+'.npy')
	for j in frameList:
		fileName = inputDir+'/output/phaseIdentification/filter/'+str(i).zfill(2)+'/'+str(j).zfill(4)+'.tiff'
		if (os.path.exists(fileName) == True):
			finalImage = cv2.imread(fileName, 0)
			gImg = finalImage[0:row, 0:col]
			bImg = finalImage[row:2*row, col:2*col]
			fImg = numpy.abs(numpy.fft.fftshift(numpy.fft.fftn(gImg)))**2
			
			[qArr, qArrScale] = myPythonFunc.calculate_qArr(gImg)
			[qTcks, rTcks] = myPythonFunc.calculate_qrTcks(ordParamArr, pixInAngstrom, qArrScale)
			
			[labelImg, numLabel, dictionary] = myPythonFunc.regionProps(bImg.astype('bool'), fImg, rTcks, area=True, perimeter=False, pixelList=False, centroid=True, intensityList=False, sumIntensity=False, theta=True, rTick=True)
			phaseDict[i][j] = dictionary
			for k in range(len(dictionary['id'])):
				outFile.write("%d %d %f %f %d\n" %(i, j, dictionary['theta'][k], dictionary['rTick'][k], dictionary['area'][k]))
pickle.dump(phaseDict, open(dirName+'/phaseDict.dat', 'wb'))
outFile.close()
##############################################################
##############################################################

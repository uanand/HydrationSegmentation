import sys
import os
import numpy
import matplotlib.pyplot as plt
import cv2
import math
import itertools
from time import time
from scipy import ndimage
from skimage.filter import threshold_otsu, rank
from skimage import data
from skimage.morphology import disk, skeletonize
import matplotlib.patches as patches

import scipy
import cPickle as pickle
import mahotas.polygon
from pylab import *
import matplotlib.cm as cm

sys.path.append(os.path.abspath('../myClasses'))
from EMImaging import dataProcessing
from EMImaging import segmentation

sys.path.append(os.path.abspath('../myFunctions'))
import myCythonFunc
import myPythonFunc


#inputDir = r'D:\Utkarsh\Guanhua\7'
#inputFile = r'D:\Utkarsh\Guanhua\7\7.avi'
#magnification = 250000
#numPixels = 1024 #1024,866,480
#microscopeName = 'JOEL2010' #'JOEL2010','T12'

#pixInAngstrom = 0.5408

#[pixInNM,pixInAngstrom] = myPythonFunc.findPixelWidth(mag=magnification,numPixels=numPixels,microscopeName=microscopeName)
#pixInAngstrom=0.4123046875
#pixInNM = pixInAngstrom/10

##############################################################
##############################################################
#DATA REFINING
##############################################################
##############################################################
#bgSubFlag=False; invertFlag=True; weightFlag=False;
#bgSubMethod='none'; sigmaBackground=20; alpha=0.8; ringRange=[[2,100]]

#methodList = ['none'] #['median','gauss','medianOFgauss','gaussOFmedian','none']
#gaussFilterSizeList = [4,8]
#medianFilterSizeList = [5,10]

#hP = dataProcessing(inputDir, inputFile, magnification, numPixels=numPixels, microscopeName=microscopeName)
#frameList = range(1,hP.numFrames+1)
#hP.createDatasets(bgSubFlag=bgSubFlag, invertFlag=invertFlag, weightFlag=weightFlag,\
				  #bgSubMethod=bgSubMethod, sigmaBackground=sigmaBackground, alpha=alpha, ringRange=ringRange,\
				  #methodList=methodList, gaussFilterSizeList=gaussFilterSizeList, medianFilterSizeList=medianFilterSizeList,\
				  #frameList=frameList)
#del hP



#hP = dataProcessing(inputDir, inputFile, magnification, numPixels=numPixels, microscopeName=microscopeName)
#frameList = range(1,hP.numFrames+1)
#hP.createRawStack()
#hP.subtractBackground(bgSubFlag=bgSubFlag, invertFlag=invertFlag, weightFlag=weightFlag,\
					  #method=bgSubMethod, sigmaBackground=sigmaBackground, alpha=alpha,  ringRange=ringRange,\
					  #frameList=frameList)
#for method in methodList:
	#if (method=='median'):
		#for medianFilterSize in medianFilterSizeList:
			#hP.refineData(method=method, medianFilterSize=medianFilterSize)
	#elif (method=='gauss'):
		#for gaussFilterSize in gaussFilterSizeList:
			#hP.refineData(method=method, gaussFilterSize=gaussFilterSize)
	#elif (method=='medianOFgauss'):
		#for gaussFilterSize in gaussFilterSizeList:
			#for medianFilterSize in medianFilterSizeList:
				#hP.refineData(method='medianOFgauss', gaussFilterSize=gaussFilterSize, medianFilterSize=medianFilterSize)
	#elif (method=='gaussOFmedian'):
		#for gaussFilterSize in gaussFilterSizeList:
			#for medianFilterSize in medianFilterSizeList:
				#hP.refineData(method='gaussOFmedian', gaussFilterSize=gaussFilterSize, medianFilterSize=medianFilterSize)
	#else:
		#print "ERROR:", method, "METHOD IS NOT A VALID OPTION"
#del hP
###############################################################
###############################################################




##############################################################
##############################################################
#PERFORMING SEGMENTATION
##############################################################
##############################################################
#methodList = ['gauss']#['median','gauss','medianOFgauss','gaussOFmedian','none']
#gaussFilterSizeList = [8]#[4,8,12]
#medianFilterSizeList = [5]

#maskFlag=False; maskMethod='none'; maskMethodMode='global'; maskIntensityRange=[0,255]; maskExcludeZero=False; maskStrelSize=0
#gradFlag=True; gradMethod='sobel'; gradMode='hvd'; gradOper=numpy.array([[-1,0,1],[-2,0,2],[-1,0,1]]); gradExclude=numpy.array([0],dtype='uint8')
#gradThreshFlag=True; gradThMethod='mean'; gradThMode='global'; gradThRange=[0,255]; gradThExcludeZero=True; gradThStrelSize=20
#cleanEdgeMethod='neighbors'; minNeighbors=2
#intensityThreshFlag=True; intThMaskFlag=True; intThMethod='otsu'; intThMode='adaptive1'; intThRange=[0,255]; intThExcludeZero=True; intThStrelSize=30
#widthRange=numpy.array([300,850], dtype='float64'); heightRange=numpy.array([150,670], dtype='float64')
#areaRange = numpy.array([25600,133000], dtype='float64')
#circularityRange = numpy.array([0.0,1], dtype='float')
#fontScale = 1

#seg = segmentation(inputDir, methodList, gaussFilterSizeList, medianFilterSizeList)
#frameList = range(1,seg.numFrames+1)
#for key in seg.dict.keys():
	#seg.refineStack = numpy.load(seg.dict[key]['data'])
	#for frame in frameList:
		#print frame
		#gImg = seg.refineStack[:,:,frame-1]
		#bImg = myPythonFunc.intensityThreshold(img=gImg, mask=(gImg>=0).astype('uint8'), flag=intensityThreshFlag, maskFlag=intThMaskFlag, method=intThMethod, mode=intThMode, intensityRange=intThRange, excludeZero=intThExcludeZero, strelSize=intThStrelSize)

		#finalImage = numpy.column_stack((myPythonFunc.normalize(bImg)),seg.gImgRawStack[:,:,frame-1])
		#cv2.imwrite(seg.dict[key]['imgDir']+'/'+str(frame).zfill(4)+'.png', finalImage)


#gImgRefineStack = numpy.load(inputDir+'/output/data/dataProcessing/refineStack/gauss/04/gImgRefineStack.npy')
#seg = segmentation(inputDir, methodList, gaussFilterSizeList, medianFilterSizeList)
#frameList = range(1,seg.numFrames+1)
#for key in seg.dict.keys():
	#seg.refineStack = numpy.load(seg.dict[key]['data'])
	#for frame in frameList:
		#print frame
		#gImg = seg.refineStack[:,:,frame-1]
		#gImgMask = seg.imageMask(img=gImg, flag=maskFlag, method=maskMethod, mode=maskMethodMode, intensityRange=maskIntensityRange, excludeZero=maskExcludeZero, strelSize=maskStrelSize)
		#grad = myPythonFunc.gradient(img=gImgMask, flag=gradFlag, method=gradMethod, mode=gradMode, oper=gradOper, exclude=gradExclude)
		#bGrad = myPythonFunc.intensityThreshold(img=grad, flag=gradThreshFlag, method=gradThMethod, mode=gradThMode, intensityRange=gradThRange, excludeZero=gradThExcludeZero, strelSize=gradThStrelSize)
		#bGradOpen = myPythonFunc.binary_opening(bGrad,iterations=4)
		#bGradOpen = ndimage.binary_dilation(bGradOpen,iterations=1)

		##bGradOpen = myCythonFunc.areaThreshold(bGradOpen.astype('uint8'), areaRange=numpy.array([areaRange[0]/4,areaRange[1]]))

		#bGradInner = myCythonFunc.removeBoundaryParticles(bGradOpen.astype('uint8'))
		##bGradInner = bGradOpen
		##bImg = myPythonFunc.intensityThreshold(img=gImg, mask=bGradInner.astype('uint8'), flag=intensityThreshFlag, maskFlag=intThMaskFlag, method=intThMethod, mode=intThMode, intensityRange=intThRange, excludeZero=intThExcludeZero, strelSize=intThStrelSize)
		#bImg = myPythonFunc.intensityThreshold(img=gImgRefineStack[:,:,frame-1], mask=bGradInner.astype('uint8'), flag=intensityThreshFlag, maskFlag=intThMaskFlag, method=intThMethod, mode=intThMode, intensityRange=intThRange, excludeZero=intThExcludeZero, strelSize=intThStrelSize)

		#bImgFill = ndimage.binary_fill_holes(bImg)
		#bImgOpen = myPythonFunc.binary_opening(bImgFill, iterations=4); bImgOpen = ndimage.binary_dilation(bImgOpen,iterations=1)
		#bImgBig = myCythonFunc.areaThreshold(bImgOpen.astype('uint8'), areaRange=areaRange)
		#bImgCircular = myCythonFunc.circularThreshold(bImgBig.astype('uint8'), circularityRange=circularityRange)

		#bImg = bImgCircular
		#labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, centroid=True)
		#bImg[:]=False
		#for i in range(1,numLabel+1):
			#bImgN=labelImg==i
			#bImgN=ndimage.binary_dilation(bImgN,iterations=15)
			#bImgN=ndimage.binary_fill_holes(bImgN)
			#bImgN=ndimage.binary_erosion(bImgN,iterations=12)
			#bImg=numpy.logical_or(bImg,bImgN)
		#bImg=ndimage.binary_fill_holes(bImg)
		#bImgCircular = bImg
		#bImgBdry = myPythonFunc.boundary(bImg)

		##bImgBdry = myPythonFunc.boundary(bImgCircular)

		#labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImgCircular, gImg, centroid=True)
		#bImgTagged = myPythonFunc.normalize(bImgCircular)
		#for j in range(len(dictionary['id'])):
			#cv2.putText(bImgTagged, str(dictionary['id'][j]), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=1, bottomLeftOrigin=False)

		#bImgCircular=myPythonFunc.normalize(bImgCircular)
		#bImgBdry=myPythonFunc.normalize(bImgBdry)
		#finalImage = numpy.column_stack((bImgCircular,bImgTagged,numpy.maximum(seg.gImgRawStack[:,:,frame-1],bImgBdry)))
		#cv2.imwrite(seg.dict[key]['imgDir']+'/'+str(frame).zfill(4)+'.png', finalImage)


#seg = segmentation(inputDir, methodList, gaussFilterSizeList, medianFilterSizeList)
#frameList = range(1,seg.numFrames+1)
#for frame in frameList:
	#print frame
	#img = cv2.imread(inputDir+'/output/images/segmentation/manual/'+str(frame).zfill(4)+'.png',0)
	#gImg = seg.gImgRawStack[:,:,frame-1]
	#bImg = img[0:seg.row,0:seg.col].astype('bool')
	#bImg = myCythonFunc.removeBoundaryParticles(bImg.astype('uint8'))
	#bImg = myCythonFunc.areaThreshold(bImg.astype('uint8'), areaRange=areaRange, flag=True)
	#bImg = ndimage.binary_fill_holes(bImg)
	#bImg = myCythonFunc.areaThreshold(bImg.astype('uint8'), areaRange=areaRange, flag=True)
	#bImg = myPythonFunc.convexHull(bImg)
	#bImgBdry = myPythonFunc.boundary(bImg)

	#labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, centroid=True)
	#bImgTagged = myPythonFunc.normalize(bImg)
	#for j in range(len(dictionary['id'])):
		#cv2.putText(bImgTagged, str(dictionary['id'][j]), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=1, bottomLeftOrigin=False)

	#bImg=myPythonFunc.normalize(bImg)
	#bImgBdry=myPythonFunc.normalize(bImgBdry)

	#finalImage = numpy.column_stack((bImg,bImgTagged,numpy.maximum(seg.gImgRawStack[:,:,frame-1],bImgBdry)))
	#cv2.imwrite(inputDir+'/output/images/segmentation/manual/'+str(frame).zfill(4)+'.png', finalImage)


#seg = segmentation(inputDir, methodList, gaussFilterSizeList, medianFilterSizeList)
#frameList = range(1,seg.numFrames+1)
#for frame in frameList:
	#print frame
	#img = cv2.imread(inputDir+'/output/images/segmentation/gauss/08/'+str(frame).zfill(4)+'.png',0)
	#gImg = seg.gImgRawStack[:,:,frame-1]
	#bImg = img[0:seg.row,0:seg.col].astype('bool')
	#labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, centroid=True)
	#bImg[:]=False
	#for i in range(1,numLabel+1):
		#bImgN=labelImg==i
		#bImgN=ndimage.binary_dilation(bImgN,iterations=20)
		#bImgN=ndimage.binary_fill_holes(bImgN)
		#bImgN=ndimage.binary_erosion(bImgN,iterations=16)
		#bImg=numpy.logical_or(bImg,bImgN)
	#bImg=ndimage.binary_fill_holes(bImg)
	#bImgBdry = myPythonFunc.boundary(bImg)

	#labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, centroid=True)
	#bImgTagged = myPythonFunc.normalize(bImg)
	#for j in range(len(dictionary['id'])):
		#cv2.putText(bImgTagged, str(dictionary['id'][j]), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=1, bottomLeftOrigin=False)

	#bImg = myPythonFunc.normalize(bImg)
	#bImgBdry = myPythonFunc.normalize(bImgBdry)
	#finalImage = numpy.column_stack((bImg,bImgTagged,numpy.maximum(gImg,bImgBdry)))
	#cv2.imwrite(inputDir+'/output/images/segmentation/gauss/08/'+str(frame).zfill(4)+'.png', finalImage)


#seg = segmentation(inputDir, methodList, gaussFilterSizeList, medianFilterSizeList)
#frameList = range(1,seg.numFrames+1)
##labelStack = numpy.load(inputDir+'/output/data/particleTracking/labelStack.npy').astype('bool')
#for frame in frameList:
	#print frame
	#img1 = cv2.imread(inputDir+'/output/images/segmentation/gauss/04 - Big/'+str(frame).zfill(4)+'.png', 0)
	#img2 = cv2.imread(inputDir+'/output/images/segmentation/gauss/04 - Small/'+str(frame).zfill(4)+'.png', 0)
	#bImg1=img1[0:seg.row,0:seg.col].astype('bool'); bImg2=img2[0:seg.row,0:seg.col].astype('bool')
	#bImg=numpy.logical_or(bImg1,bImg2)
	##bImg=img[0:seg.row,0:seg.col].astype('bool'); bImg=numpy.logical_or(labelStack[:,:,frame-1],bImg); bImg=ndimage.binary_dilation(bImg,iterations=2)
	#bImgBdry = myPythonFunc.boundary(bImg)

	#labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg, seg.gImgRawStack[:,:,frame-1], centroid=True)
	#bImgTagged = myPythonFunc.normalize(bImg)
	#for j in range(len(dictionary['id'])):
		#cv2.putText(bImgTagged, str(dictionary['id'][j]), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=1, bottomLeftOrigin=False)

	#bImg=myPythonFunc.normalize(bImg)
	#bImgBdry=myPythonFunc.normalize(bImgBdry)
	#bImgTagged=myPythonFunc.normalize(bImgTagged)

	#finalImage = numpy.column_stack((bImg,bImgTagged,numpy.maximum(bImgBdry,seg.gImgRawStack[:,:,frame-1])))
	#cv2.imwrite(inputDir+'/output/images/segmentation/gauss/04/'+str(frame).zfill(4)+'.png', finalImage)


#seg.loadData(method='medianOFgauss',gaussFilterSize=8,medianFilterSize=15)

#seg.gImgRaw = seg.gImgRawStack[:,:,5]
#seg.gImgMask = seg.intensityThreshold(gImg=ndimage.gaussian_filter(seg.gImgStack[:,:,5], sigma=2), method='none', partitions=[4,10])
#seg.refineImgMask = seg.intensityThreshold(gImg=seg.gImgRefineStack[:,:,5], method='none', partitions=[4,4])
#seg.createInputImage(gImg=seg.gImgRefineStack[:,:,5], gImgMask=seg.gImgMask, refineImgMask=seg.refineImgMask)
#seg.findContour(gImg=seg.gImgInput, exclude=exclude, gradientRange=gradientRange)
#seg.cleanContour(bImg=seg.bImgAllContour, method='neighbors', minNeighbors=minNeighbors)
#seg.fillHoles(bImg=seg.bImgCleanContour, flag=True)
#seg.removeBoundaryParticles(bImg=seg.bImgFill, flag=False)
#seg.binary_opening(bImg=seg.bImgInner, iterations=5, flag=True)
#seg.areaThreshold(bImg=seg.bImgOpen, areaRange=areaRange, flag=True)
#seg.circularThreshold(bImg=seg.bImgBig, circularityRange=circularityRange, flag=True)
#seg.labelParticles(bImg=seg.bImgCircular, fontScale=fontScale)
#seg.collateResult()
#cv2.imwrite(seg.imgDir+'/'+str(5).zfill(4)+'.png', seg.finalImage)

#plt.figure(), plt.imshow(seg.bImgAllContour), plt.show()
#plt.figure(), plt.imshow(seg.bImgCleanContour), plt.show()
#plt.figure(), plt.imshow(seg.bImgFill), plt.show()
#plt.figure(), plt.imshow(seg.bImgInner), plt.show()
#plt.figure(), plt.imshow(seg.bImgOpen), plt.show()
#plt.figure(), plt.imshow(seg.bImgBig), plt.show()
#plt.figure(), plt.imshow(seg.bImgCircular), plt.show()
#bImgCircular

#plt.figure(), plt.imshow(seg.gImgInput), plt.show()
#plt.figure(), plt.imshow(seg.gImgRefineStack[:,:,seg.frame-1]), plt.show()
#for frame in frameList:
	#seg.gImgMask = seg.intensityThreshold(gImg=ndimage.gaussian_filter(seg.gImgStack[:,:,frame-1], sigma=2), method='median', intensityRange=maskIntensityRange, partitions=maskPartitions)
	#seg.refineImageMask = seg.intensityThreshold(gImg=seg.gImgRefineStack[:,:,frame-1], method='none')
	#seg.createInputImage(frame)
	#seg.findContour(gImg=seg.gImgInput, exclude)
	#seg.cleanContour(method='none', minNeighbors=minNeighbors)
	#seg.fillHoles()
	#seg.removeBoundaryParticles()
	#seg.binary_opening()
	#seg.areaThreshold(areaRange=[])
	#seg.circularThreshold(circularityRange=[])
	#seg.collateResult(frame, cleanContourMethod, gImgMaskMethod)
##############################################################
##############################################################




##############################################################
##############################################################
#CREATING BINARY IMAGE STACK FROM FINAL SEGMENTATION
##############################################################
##############################################################
#print 'CREATING bImgStack'
#gImgRawStack = numpy.load(inputDir+'/output/data/dataProcessing/gImgRawStack/gImgRawStack.npy')
#[row, col, numFrames] = gImgRawStack.shape
#bImgStack = numpy.zeros([row, col, numFrames], dtype='bool')

#frameList = range(1, numFrames+1)

#for frame in frameList:
	#img = cv2.imread(inputDir+'/output/images/segmentation/final/'+str(frame).zfill(4)+'.png', 0)
	#bImg = img[0:row, 0:col].astype('bool')
	#bImgStack[:,:,frame-1] = bImg
#numpy.save(inputDir+'/output/data/segmentation/bImgStack.npy', bImgStack)




##############################################################
##############################################################
#LABELLING PARTICLES
##############################################################
##############################################################
#print "LABELLING PARTICLES"
#myPythonFunc.mkdir(inputDir+'/output/data/particleTracking')
#myPythonFunc.mkdir(inputDir+'/output/images/particleTracking')
#myPythonFunc.mkdir(inputDir+'/output/images/particleTracking/labels')
#gImgRawStack = numpy.load(inputDir+'/output/data/dataProcessing/gImgRawStack/gImgRawStack.npy')
#bImgStack = numpy.load(inputDir+'/output/data/segmentation/bImgStack.npy')
##bImgStack=numpy.load(inputDir+'/output/data/particleTracking/bImgSmoothStack.npy')
#[row, col, numFrames] = gImgRawStack.shape

#centerDispTh = 15
#perctAreaChangeTh = 0.5
#missFramesTh = 10
#frameList = range(1, numFrames+1)
#structure = [[0,1,0],[1,1,1],[0,1,0]]

#labelStack = numpy.zeros([row, col, numFrames], dtype='int32')

#for frame in frameList:
	#if (frame == frameList[0]):
		#labelImg_0, numLabel_0, dictionary_0 = myPythonFunc.regionProps(bImgStack[:,:,frame-1], gImgRawStack[:,:,frame-1], structure=structure, centroid=True, area=True)
		#maxID = numLabel_0
		#dictionary_0['frame'] = []
		#for j in range(len(dictionary_0['id'])):
			#dictionary_0['frame'].append(frame)
		#labelStack[:,:,frame-1] = labelImg_0
	#else:
		#labelImg_1, numLabel_1, dictionary_1 = myPythonFunc.regionProps(bImgStack[:,:,frame-1], gImgRawStack[:,:,frame-1], structure=structure, centroid=True, area=True)
		#for j in range(len(dictionary_1['id'])):
			#flag = 0
			#bImg_1_LabelN = labelImg_1==dictionary_1['id'][j]
			#center_1 = dictionary_1['centroid'][j]
			#area_1 = dictionary_1['area'][j]
			#frame_1 = frame
			#for k in range(len(dictionary_0['id'])-1,-1,-1):
				#center_0 = dictionary_0['centroid'][k]
				#area_0 = dictionary_0['area'][k]
				#frame_0 = dictionary_0['frame'][k]
				#if ((sqrt((center_1[0]-center_0[0])**2. + (center_1[1]-center_0[1])**2.) <= centerDispTh) and (1.*area_0/area_1 >= (1-perctAreaChangeTh) and 1.*area_0/area_1 <= (1+perctAreaChangeTh)) and (frame_1-frame_0 <= missFramesTh)):
					#labelStack[:,:,frame-1] += (bImg_1_LabelN*dictionary_0['id'][k])
					#dictionary_0['centroid'][k] = center_1
					#dictionary_0['area'][k] = area_1
					#dictionary_0['frame'][k] = frame
					#flag = 1
					#break
			#if (flag == 0):
				#maxID += 1
				#labelN_1 = bImg_1_LabelN*maxID
				#labelStack[:,:,frame-1] += labelN_1
				#dictionary_0['id'].append(maxID)
				#dictionary_0['centroid'].append(center_1)
				#dictionary_0['area'].append(area_1)
				#dictionary_0['frame'].append(frame)
#if (labelStack.max() < 256):
	#labelStack = labelStack.astype('uint8')
#elif (labelStack.max()<65536):
	#labelStack = labelStack.astype('uint16')
#numpy.save(inputDir+'/output/data/particleTracking/labelStack.npy', labelStack)


#print "PRINTING ORIGINAL LABELLED PARTICLES"
#for frame in frameList:
	#bImg = bImgStack[:,:,frame-1]
	#bImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(bImg))
	#gImg = gImgRawStack[:,:,frame-1]
	#labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, structure=structure, centroid=True)
	#bImg = myPythonFunc.normalize(bImg)
	#for j in range(len(dictionary['id'])):
		#bImgLabelN = labelImg == dictionary['id'][j]
		#ID = numpy.max(bImgLabelN*labelStack[:,:,frame-1])
		#cv2.putText(bImg, str(ID), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=1, bottomLeftOrigin=False)
	#finalImage = numpy.column_stack((bImg, numpy.maximum(gImg,bImgBdry)))
	#cv2.imwrite(inputDir+'/output/images/particleTracking/labels/'+str(frame).zfill(4)+'.png', finalImage)


#print numpy.unique(labelStack)[1:]




#print "LABELLING PARTICLES"
#myPythonFunc.mkdir(inputDir+'/output/data/particleTracking')
#myPythonFunc.mkdir(inputDir+'/output/images/particleTracking')
#myPythonFunc.mkdir(inputDir+'/output/images/particleTracking/labels')
#gImgRawStack = numpy.load(inputDir+'/output/data/dataProcessing/gImgRawStack/gImgRawStack.npy')
#bImgStack = numpy.load(inputDir+'/output/data/segmentation/bImgStack.npy')
#[row, col, numFrames] = gImgRawStack.shape

#centerDispRange = [10,20]
#perctAreaChangeRange = [0.8,1]
#missFramesTh = 20
#frameList = range(1, numFrames+1)
#structure = [[0,1,0],[1,1,1],[0,1,0]]

#labelStack = numpy.zeros([row, col, numFrames], dtype='int32')

#for frame in frameList:
	#if (frame == frameList[0]):
		#labelImg_0, numLabel_0, dictionary_0 = myPythonFunc.regionProps(bImgStack[:,:,frame-1], gImgRawStack[:,:,frame-1], structure=structure, centroid=True, area=True)
		#maxID = numLabel_0
		#dictionary_0['frame'] = []
		#for j in range(len(dictionary_0['id'])):
			#dictionary_0['frame'].append(frame)
		#labelStack[:,:,frame-1] = labelImg_0
	#else:
		#labelImg_1, numLabel_1, dictionary_1 = myPythonFunc.regionProps(bImgStack[:,:,frame-1], gImgRawStack[:,:,frame-1], structure=structure, centroid=True, area=True)
		#areaMin = min(dictionary_1['area']); areaMax = max(dictionary_1['area'])
		#for j in range(len(dictionary_1['id'])):
			#flag = 0
			#bImg_1_LabelN = labelImg_1==dictionary_1['id'][j]
			#center_1 = dictionary_1['centroid'][j]
			#area_1 = dictionary_1['area'][j]
			#frame_1 = frame
			#perctAreaChangeTh = perctAreaChangeRange[1]+1.0*(perctAreaChangeRange[0]-perctAreaChangeRange[1])/(areaMax-areaMin)*(area_1-areaMin)
			#centerDispTh = 1.0*(perctAreaChangeTh-perctAreaChangeRange[0])/(perctAreaChangeRange[1]-perctAreaChangeRange[0])*(centerDispRange[1]-centerDispRange[0])+centerDispRange[0]
			#for k in range(len(dictionary_0['id'])-1,-1,-1):
				#center_0 = dictionary_0['centroid'][k]
				#area_0 = dictionary_0['area'][k]
				#frame_0 = dictionary_0['frame'][k]
				#if ((sqrt((center_1[0]-center_0[0])**2. + (center_1[1]-center_0[1])**2.) <= centerDispTh) and (1.*area_0/area_1 >= (1-perctAreaChangeTh) and 1.*area_0/area_1 <= (1+perctAreaChangeTh)) and (frame_1-frame_0 <= missFramesTh)):
					#labelStack[:,:,frame-1] += (bImg_1_LabelN*dictionary_0['id'][k])
					#dictionary_0['centroid'][k] = center_1
					#dictionary_0['area'][k] = area_1
					#dictionary_0['frame'][k] = frame
					#flag = 1
					#break
			#if (flag == 0):
				#maxID += 1
				#labelN_1 = bImg_1_LabelN*maxID
				#labelStack[:,:,frame-1] += labelN_1
				#dictionary_0['id'].append(maxID)
				#dictionary_0['centroid'].append(center_1)
				#dictionary_0['area'].append(area_1)
				#dictionary_0['frame'].append(frame)
#if (labelStack.max() < 256):
	#labelStack = labelStack.astype('uint8')
#elif (labelStack.max()<65536):
	#labelStack = labelStack.astype('uint16')
#numpy.save(inputDir+'/output/data/particleTracking/labelStack.npy', labelStack)


#print "PRINTING ORIGINAL LABELLED PARTICLES"
#for frame in frameList:
	#bImg = bImgStack[:,:,frame-1]
	#bImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(bImg))
	#gImg = gImgRawStack[:,:,frame-1]
	#labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, structure=structure, centroid=True)
	#bImg = myPythonFunc.normalize(bImg)
	#for j in range(len(dictionary['id'])):
		#bImgLabelN = labelImg == dictionary['id'][j]
		#ID = numpy.max(bImgLabelN*labelStack[:,:,frame-1])
		#cv2.putText(bImg, str(ID), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=1, bottomLeftOrigin=False)
	#finalImage = numpy.column_stack((bImg, numpy.maximum(gImg,bImgBdry)))
	#cv2.imwrite(inputDir+'/output/images/particleTracking/labels/'+str(frame).zfill(4)+'.png', finalImage)

#print "FIRST AND LAST OCCURENCE OF A PARTICLE"
##labelStack = numpy.load(inputDir+'/output/data/particleTracking/labelStack.npy')
#particleList=numpy.unique(labelStack)[1:]

#for particle in particleList:
	#allFrames = numpy.where(labelStack==particle)[2]
	#print "PARTICLE", particle, " - FIRST FRAME: ", allFrames.min()+1, ", LAST FRAME: ", allFrames.max()+1




##############################################################
##############################################################
#REMOVING UNWANTED PARTICLES
##############################################################
##############################################################
#print "REMOVING UNWANTED PARTICLES"
#labelStack = numpy.load(inputDir+'/output/data/particleTracking/labelStack.npy')
#particleList = numpy.unique(labelStack)[1:]

#keepList = [1,2,3,4,5,6,7,8,9,15,57,62,63,65,66]
#removeList = []
##removeList = [6,24,30,35,36,37,39,40,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,59,60,61,64,65,67,68,69,70,71,72,73,74,75,76,77,78,80,81,82,85,87,88,89,90,91,92,93,94,95,96,97,98,99,100]+range(101,114)+range(115,123)+[124,126,127]+range(129,135)+range(136,140)+[141,142,143]+range(145,162)+[163,164]+range(166,174)+range(175,187)+range(188,194)+[195]+range(197,203)+range(206,211)+[212,214,215,217,218,220,221]+range(223,227)+range(229,243)

#if not removeList:
	#removeList = [s for s in particleList if s not in keepList]

#for i in removeList:
	#labelStack[labelStack==i] = 0
#bImgStack = labelStack.astype('bool')
#numpy.save(inputDir+'/output/data/segmentation/bImgStack.npy', bImgStack)
#if (labelStack.max() < 256):
	#labelStack = labelStack.astype('uint8')
#elif (labelStack.max()<65536):
	#labelStack = labelStack.astype('uint16')
#numpy.save(inputDir+'/output/data/particleTracking/labelStack.npy', labelStack)

#print "PRINTING ORIGINAL LABELLED PARTICLES"
#gImgRawStack = numpy.load(inputDir+'/output/data/dataProcessing/gImgRawStack/gImgRawStack.npy')
#bImgStack = labelStack.astype('bool')
#[row, col, numFrames] = gImgRawStack.shape
#frameList = range(1,numFrames+1)
#structure = [[0,1,0],[1,1,1],[0,1,0]]
#for frame in frameList:
	#bImg = bImgStack[:,:,frame-1]
	#bImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(bImg))
	#gImg = gImgRawStack[:,:,frame-1]
	#labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, structure=structure, centroid=True)
	#bImg = myPythonFunc.normalize(bImg)
	#for j in range(len(dictionary['id'])):
		#bImgLabelN = labelImg == dictionary['id'][j]
		#ID = numpy.max(bImgLabelN*labelStack[:,:,frame-1])
		#cv2.putText(bImg, str(ID), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=1, bottomLeftOrigin=False)
	#finalImage = numpy.column_stack((bImg, numpy.maximum(gImg,bImgBdry)))
	#cv2.imwrite(inputDir+'/output/images/particleTracking/labels/'+str(frame).zfill(4)+'.png', finalImage)
##############################################################
##############################################################



##############################################################
##############################################################
#LABEL CORRECTION
##############################################################
##############################################################
#print "LABEL CORRECTION"
#labelStack = numpy.load(inputDir+'/output/data/particleTracking/labelStack.npy')
#[row, col, numFrames] = labelStack.shape
#frameList = range(1, numFrames+1)
#structure = [[0,1,0],[1,1,1],[0,1,0]]

#correctionList = [[5,4,3,2,1]]
#for i in range(len(correctionList)):
	#for j in range(len(correctionList[i])-1):
		#labelStack[labelStack==correctionList[i][j]] = correctionList[i][-1]
##labelStack[:,:,72:numFrames][labelStack[:,:,72:numFrames]==7]=60
##labelStack[:,:,222:numFrames][labelStack[:,:,222:numFrames]==9]=61
##labelStack[:,:,340:341][labelStack[:,:,340:341]==31]=62

#correctionList=[]
#particlesOld = numpy.unique(labelStack)[1:]; maxLabel = particlesOld.max()+1; counter=0
#newLabels = numpy.zeros([len(particlesOld),3],dtype='int')
#for label in particlesOld:
	#bStack = labelStack==label
	#index = numpy.nonzero(bStack)
	#newLabels[counter][0] = index[2].min(); newLabels[counter][1] = label;
	#counter+=1
#newLabels = newLabels[newLabels[:,0].argsort(kind='mergesort')]
#for i in range(len(particlesOld)):
	#newLabels[i][2]=maxLabel+i
	#correctionList.append([newLabels[i][1],newLabels[i][2]])
#for i in range(len(correctionList)):
	#for j in range(len(correctionList[i])-1):
		#labelStack[labelStack==correctionList[i][j]] = correctionList[i][-1]

#correctionList=[]
#particlesOld = numpy.unique(labelStack)[1:]; particlesNew = range(1,len(particlesOld)+1)
#for i in range(len(particlesOld)):
	#correctionList.append([particlesOld[i],particlesNew[i]])
#for i in range(len(correctionList)):
	#for j in range(len(correctionList[i])-1):
		#labelStack[labelStack==correctionList[i][j]] = correctionList[i][-1]

#if (labelStack.max() < 256):
	#labelStack = labelStack.astype('uint8')
#elif (labelStack.max()<65536):
	#labelStack = labelStack.astype('uint16')
#numpy.save(inputDir+'/output/data/particleTracking/labelStack.npy', labelStack)

#gImgRawStack = numpy.load(inputDir+'/output/data/dataProcessing/gImgRawStack/gImgRawStack.npy')
#bImgStack = labelStack.astype('bool')

#print "PRINTING ORIGINAL LABELLED PARTICLES"
#for frame in frameList:
	#bImg = bImgStack[:,:,frame-1]
	#bImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(bImg))
	#gImg = gImgRawStack[:,:,frame-1]
	#labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, structure=structure, centroid=True)
	#bImg = myPythonFunc.normalize(bImg)
	#for j in range(len(dictionary['id'])):
		#bImgLabelN = labelImg == dictionary['id'][j]
		#ID = numpy.max(bImgLabelN*labelStack[:,:,frame-1])
		#cv2.putText(bImg, str(ID), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=1, bottomLeftOrigin=False)
	#finalImage = numpy.column_stack((bImg, numpy.maximum(gImg,bImgBdry)))
	#cv2.imwrite(inputDir+'/output/images/particleTracking/labels/'+str(frame).zfill(4)+'.png', finalImage)
##############################################################
##############################################################


##############################################################
##############################################################
#SMOOTHING STACK BASED ON LABELS
##############################################################
##############################################################
#print "SMOOTHING STACK BASED ON LABELS"
#gImgRawStack = numpy.load(inputDir+'/output/data/dataProcessing/gImgRawStack/gImgRawStack.npy')
#labelStack = numpy.load(inputDir+'/output/data/particleTracking/labelStack.npy')
#[row, col, numFrames] = gImgRawStack.shape
#myPythonFunc.mkdir(inputDir+'/output/images/particleTracking/smooth')
#bImgSmoothStack = numpy.zeros([row,col,numFrames], dtype='bool')

#particleList = numpy.unique(labelStack)[1:]
#frameList = range(1,numFrames+1)
#gaussBlurSize = 4
##areaRange = numpy.array([10,252000], dtype='float64')

#for particle in particleList:
	#bImgStackN = labelStack==particle
	#bImgStackN = myPythonFunc.normalize(bImgStackN).astype('double')
	#[r,c,frames] = numpy.nonzero(bImgStackN)
	#bImgStackN = bImgStackN[:,:,frames.min():frames.max()+1]
	#bImgSmoothStackN = ndimage.gaussian_filter1d(bImgStackN, sigma=gaussBlurSize, axis=2)
	#bImgSmoothStackN = myPythonFunc.normalize(bImgSmoothStackN)
	#bImgSmoothStackN = bImgSmoothStackN>=150
	#bImgSmoothStack[:,:,frames.min():frames.max()+1] = numpy.logical_or(bImgSmoothStackN, bImgSmoothStack[:,:,frames.min():frames.max()+1])

#for frame in frameList:
	#bImg = myCythonFunc.areaThreshold(bImgSmoothStack[:,:,frame-1].astype('uint8'), areaRange=areaRange)
	#bImg = ndimage.binary_fill_holes(bImg)
	##bImg = myPythonFunc.binary_closing(bImg, iterations=4)#gaussBlurSize)
	##label, nlabel = ndimage.label(bImg, structure=[[1,1,1],[1,1,1],[1,1,1]])
	##bImg[:]=False
	##for i in range(1,nlabel+1):
		##bImgN = label==i
		##bImgN = ndimage.binary_dilation(bImgN, structure=[[1,1,1],[1,1,1],[1,1,1]], iterations=20)
		##bImgN = ndimage.binary_erosion(bImgN, structure=[[1,1,1],[1,1,1],[1,1,1]], iterations=15)
		##bImg = numpy.logical_or(bImg,bImgN)
		##bImgSmoothStack[:,:,frame-1] = numpy.logical_or(bImgSmoothStack[:,:,frame-1],bImgN)
	#bImgSmoothStack[:,:,frame-1] = bImg
	#bImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(bImg))
	#finalImage = numpy.column_stack((myPythonFunc.normalize(bImg), numpy.maximum(gImgRawStack[:,:,frame-1],bImgBdry)))
	#cv2.imwrite(inputDir+'/output/images/particleTracking/smooth/'+str(frame).zfill(4)+'.png', finalImage)
#numpy.save(inputDir+'/output/data/particleTracking/bImgSmoothStack.npy', bImgSmoothStack)


##############################################################
##############################################################
#FINDING MEASURES FOR TRACKED PARTICLES
##############################################################
##############################################################
print "FINDING MEASURES FOR TRACKED PARTICLES"
#gImgRawStack = numpy.load(inputDir+'/output/data/dataProcessing/gImgRawStack/gImgRawStack.npy')
#labelStack = numpy.load(inputDir+'/output/data/particleTracking/labelStack.npy')
#[pixInNM,pixInAngstrom] = myPythonFunc.findPixelWidth(mag=magnification,numPixels=numPixels,microscopeName=microscopeName)

gImgRawStack = numpy.load(r'D:\Utkarsh\Guanhua\14\output\data\gImgRawStack.npy')
labelStack = numpy.load(r'D:\Utkarsh\Guanhua\14\output\data\particleIdentification\labelStack.npy')
outFile2 = open(r'D:\Utkarsh\Guanhua\14\output\data\particleIdentification\imgDataNM.dat','wb')
pixInNM = 10.0/113

#if (labelStack.max() < 256):
	#labelStack = labelStack.astype('uint8')
#elif (labelStack.max()<65536):
	#labelStack = labelStack.astype('uint16')
#numpy.save(inputDir+'/output/data/particleTracking/labelStack.npy', labelStack)

#outFile1 = open(inputDir+'/output/data/particleTracking/imgDataPixels.dat', 'wb')
#outFile2 = open(inputDir+'/output/data/particleTracking/imgDataNM.dat', 'wb')

[row, col, numFrames] = gImgRawStack.shape
structure = [[0,1,0],[1,1,1],[0,1,0]]
frameList = range(1,numFrames+1)
particleList = numpy.unique(labelStack)[1:]


area=True
perimeter=True
circularity=True
pixelList=False
bdryPixelList=True
centroid=True
intensityList=False
sumIntensity=False
effRadius=True
radius=False
circumRadius=False
inRadius=False
radiusOFgyration=False

imgDataDict = {}
for particle in particleList:
	imgDataDict[particle]={}

for i in frameList:
	print i
	label = labelStack[:,:,i-1]
	gImg = gImgRawStack[:,:,i-1]
	#outFile1.write("%d " %(i))
	outFile2.write("%f " %(i/10.))
	for j in particleList:
		bImg = label==j
		if (bImg.max() == True):
			labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, structure=structure, centroid=centroid, area=area, perimeter=perimeter, circularity=circularity, pixelList=pixelList, bdryPixelList=bdryPixelList, effRadius=effRadius, radius=radius, circumRadius=circumRadius, inRadius=inRadius, radiusOFgyration=radiusOFgyration)
			#outFile1.write("%f %f %f %f %f %f " %(dictionary['centroid'][0][1], row-dictionary['centroid'][0][0], dictionary['area'][0], dictionary['perimeter'][0], dictionary['circularity'][0], dictionary['effRadius'][0]))
			outFile2.write("%f %f %f %f %f %f " %(dictionary['centroid'][0][1]*pixInNM, (row-dictionary['centroid'][0][0])*pixInNM, dictionary['area'][0]*pixInNM*pixInNM, dictionary['perimeter'][0]*pixInNM, dictionary['circularity'][0], dictionary['effRadius'][0]*pixInNM))
			imgDataDict[j][i]={}; imgDataDict[j][i]=dictionary
		else:
			pass
			#outFile1.write("nan nan nan nan nan nan ")
			outFile2.write("nan nan nan nan nan nan ")
	#outFile1.write("\n")
	outFile2.write("\n")
#pickle.dump(imgDataDict, open(inputDir+'/output/data/particleTracking/imgDataDict.dat', 'wb'))
#outFile1.close()
outFile2.close()
##############################################################
##############################################################


##############################################################
##############################################################
#CALCULATE THE MINIMUM AND MAXIMUM DISTANCE BETWEEN ALL PARTICLES FOR ALL TIME FRAMES
##############################################################
##############################################################
#print "CALCULATE THE MINIMUM AND MAXIMUM DISTANCE BETWEEN ALL PARTICLES FOR ALL TIME FRAMES"
#gImgRawStack = numpy.load(inputDir+'/output/data/dataProcessing/gImgRawStack/gImgRawStack.npy')
#labelStack = numpy.load(inputDir+'/output/data/particleTracking/labelStack.npy')
#[row,col,numFrames] = labelStack.shape
#imgDataDict = pickle.load(open(inputDir+'/output/data/particleTracking/imgDataDict.dat','rb'))
###outFile1 = open(inputDir+'/output/data/particleTracking/minDistDataPixels.dat', 'wb')
##outFile2 = open(inputDir+'/output/data/particleTracking/minDistDataNM.dat', 'wb')
###outFile3 = open(inputDir+'/output/data/particleTracking/meanDistDataPixels.dat', 'wb')
##outFile4 = open(inputDir+'/output/data/particleTracking/meanDistDataNM.dat', 'wb')

#frameList = [464]#range(1,numFrames+1)
#particleList = numpy.unique(labelStack)[1:]

#bondingList=[]
##tempList = [1,2,3,4,5,6,7,8,9,10,11,12,13]
#tempList = [1,2]
#combos = itertools.combinations(tempList,2)
#for bonds in combos:
	#temp = [bonds[0],bonds[1]]
	#bondingList.append(temp)

##bondingList = [[1,2]]

##distDict = {}
##for bonds in bondingList:
	##distDict[str(bonds[0])+'_'+str(bonds[1])] = {}
	##for frame in frameList:
		##distDict[str(bonds[0])+'_'+str(bonds[1])][frame]=[]

#dMinArray = numpy.zeros([len(particleList),len(particleList),numFrames]); dMaxArray = numpy.zeros([len(particleList),len(particleList),numFrames]); dMeanArray = numpy.zeros([len(particleList),len(particleList),numFrames])
#dMinArray[:,:,:] = numpy.nan; dMaxArray[:,:,:] = numpy.nan; dMeanArray[:,:,:] = numpy.nan

#for frame in frameList:
	#print frame
	#for i in range(len(particleList)):
		#label_i = labelStack[:,:,frame-1]==particleList[i]
		#if (label_i.max() > 0):
			#r1=numpy.array(imgDataDict[particleList[i]][frame]['bdryPixelList'][0][0]).astype('int'); c1=numpy.array(imgDataDict[particleList[i]][frame]['bdryPixelList'][0][1]).astype('int')
			
			#center1 = imgDataDict[particleList[i]][frame]['centroid'][0]
			#periphery1 = numpy.transpose(numpy.asarray(imgDataDict[particleList[i]][frame]['bdryPixelList'][0]))
			#radius1 = imgDataDict[particleList[i]][frame]['effRadius'][0]
			#for j in range(len(particleList)):
				#if ([particleList[i],particleList[j]] in bondingList):
					#label_j = labelStack[:,:,frame-1]==particleList[j]
					#if (label_j.max() > 0):
						#r2=numpy.array(imgDataDict[particleList[j]][frame]['bdryPixelList'][0][0]).astype('int'); c2=numpy.array(imgDataDict[particleList[j]][frame]['bdryPixelList'][0][1]).astype('int')
						
						#center2 = imgDataDict[particleList[j]][frame]['centroid'][0]
						#periphery2 = numpy.transpose(numpy.asarray(imgDataDict[particleList[j]][frame]['bdryPixelList'][0]))
						#radius2 = imgDataDict[particleList[j]][frame]['effRadius'][0]
						
						#[dMin, dMax] = myCythonFunc.calculateMinMaxDist(r1,c1,r2,c2)
						#dList, pointsList = myPythonFunc.distanceRange(center1,periphery1,radius1,center2,periphery2,radius2,theta=numpy.pi/12)
						##distDict[str(particleList[i])+'_'+str(particleList[j])][frame].append(dList)
						
						#dMinArray[i][j][frame-1]=dMin; dMaxArray[i][j][frame-1]=dMax; dMeanArray[i][j][frame-1]=mean(dList)
						#dMinArray[j][i][frame-1]=dMin; dMaxArray[j][i][frame-1]=dMax; dMeanArray[j][i][frame-1]=mean(dList)
						
						#bImg = (labelStack[:,:,frame-1].astype('bool'))*255
						#gImg = gImgRawStack[:,:,frame-1]
						#rgbBimg = myPythonFunc.gray2rgb(bImg)
						#rgbGImg = myPythonFunc.gray2rgb(gImg)
						
						#for points in pointsList:
							#rgbBimg[points[0],points[1],1:] = 0
						
						#img = rgbBimg.copy(); img=img[345:406,375:436,:]
						#R,C,depth = img.shape
						#fig = plt.figure(figsize=(2,2))
						#ax = fig.add_axes([0,0,1,1])
						#ax.imshow(img, extent=[0,C*pixInNM,0,R*pixInNM])
						#ax.set_xticks([])
						#ax.set_yticks([])
						#width = 1
						#color='#00FF00'
						#ax.spines['bottom'].set_color(color)
						#ax.spines['top'].set_color(color)
						#ax.spines['left'].set_color(color)
						#ax.spines['right'].set_color(color)
						#ax.spines['bottom'].set_linewidth(width)
						#ax.spines['top'].set_linewidth(width)
						#ax.spines['left'].set_linewidth(width)
						#ax.spines['right'].set_linewidth(width)
						#plt.savefig('zoom.png',format='png')
						#plt.savefig('zoom.pdf',format='pdf')
						#plt.cla()
						
						#ax.imshow(bImg,extent=[0,col*pixInNM,0,row*pixInNM])
						#ax.add_patch(patches.Rectangle((375,345),60,60,fill=False,edgecolor='#00FF00',linewidth=1))
						##ax.set_xticks([])
						##ax.set_yticks([])
						##width = 0
						##ax.spines['bottom'].set_linewidth(width)
						##ax.spines['top'].set_linewidth(width)
						##ax.spines['left'].set_linewidth(width)
						##ax.spines['right'].set_linewidth(width)
						#plt.savefig('bImg.png',format='png')
						#plt.savefig('bImg.pdf',format='pdf')
						#plt.close()
						
						
						##bgrBImg = myPythonFunc.RGBtoBGR(rgbBimg)
						##bgrGImg = myPythonFunc.RGBtoBGR(rgbGImg)
						
						##cv2.imwrite('bgrGImg.png',bgrGImg)
						##cv2.imwrite('bgrBImg.png',bgrBImg)
						##cv2.imwrite('bImg.png',bImg)
						
#fig = plt.figure(figsize=(3,3))
#ax = fig.add_axes([0,0,1,1])
#ax.set_xlim(0,10), ax.set_ylim(0,10)
#ax.add_patch(patches.Circle((4,5),1,facecolor='#ffd700',edgecolor='#ffd700'))
#ax.add_patch(patches.Circle((8,5),1.5,facecolor='#ffd700',edgecolor='#ffd700'))
#ax.plot((4,8),(5,5),color='k',linestyle='dotted')
#ax.plot((4,4.96),(5,5.26),color='k',linestyle='dotted')
#ax.plot((4,4.96),(5,4.74),color='k',linestyle='dotted')
#ax.add_patch(patches.Arc((4,5),1,1,theta1=-15,theta2=15,edgecolor='k'))
#ax.add_patch(patches.Arc((4,5),2,2,theta1=-15,theta2=15,edgecolor='#FF0000'))
#ax.text(1,1,r'$\frac{\pi}{6}$')
#ax.set_xticks([])
#ax.set_yticks([])
#width = 0
#ax.spines['bottom'].set_linewidth(width)
#ax.spines['top'].set_linewidth(width)
#ax.spines['left'].set_linewidth(width)
#ax.spines['right'].set_linewidth(width)
#plt.savefig('scheme.png',format='png')
#plt.savefig('scheme.pdf',format='pdf')
#plt.show()



#flag=0
#for i in range(len(particleList)-1):
	#for j in range(i+1,len(particleList)):
		#temp = dMeanArray[i,j,:]*pixInNM
		#temp = temp[numpy.isfinite(temp)]
		#if (temp.size>0):
			#if (temp.min()<=6):
				#if (flag==0):
					#flag=1
					#finalArray = dMeanArray[i,j,:]*pixInNM
					#header = str(i+1)+'_'+str(j+1)+' '
				#else:
					#finalArray = numpy.column_stack((finalArray, dMeanArray[i,j,:]*pixInNM))
					#header = header + str(i+1)+'_'+str(j+1)+' '
				
#numpy.savetxt(inputDir+'/output/data/particleTracking/meanDistDataNM.dat',finalArray,fmt='%.6f',header=header)
		
#for i in range(len(particleList)-1):
	#for j in range(i+1,len(particleList)):
		#if ([particleList[i],particleList[j]] in bondingList):
			##outFile1.write("%s " %(str(particleList[i])+'_'+str(particleList[j])))
			#outFile2.write("%s " %(str(particleList[i])+'_'+str(particleList[j])))
			##outFile3.write("%s " %(str(particleList[i])+'_'+str(particleList[j])))
			#outFile4.write("%s " %(str(particleList[i])+'_'+str(particleList[j])))
			#for frame in frameList:
				##outFile1.write("%f " %(dMinArray[i][j][frame-1]))
				#outFile2.write("%f " %(dMinArray[i][j][frame-1]*pixInNM))
				##outFile3.write("%f " %(dMeanArray[i][j][frame-1]))
				#outFile4.write("%f " %(dMeanArray[i][j][frame-1]*pixInNM))
			##outFile1.write("\n")
			#outFile2.write("\n")
			##outFile3.write("\n")
			#outFile4.write("\n")
##outFile1.close()
#outFile2.close()
##outFile3.close()
#outFile4.close()
#pickle.dump(distDict, open(inputDir+'/output/data/particleTracking/distDict.dat', 'wb'))
##############################################################
##############################################################

#outFile = open(inputDir+'/output/data/particleTracking/radDistSpread.dat', 'wb')

#imgDataDict = pickle.load(open(inputDir+'/output/data/particleTracking/imgDataDict.dat','rb'))
#distDict = pickle.load(open(inputDir+'/output/data/particleTracking/distDict.dat', 'rb'))
#labelStack = numpy.load(inputDir+'/output/data/particleTracking/labelStack.npy')

##bondingList = []

#for bonds in bondingList:
	#print bonds
	#allFrames = numpy.where(labelStack==bonds[0])[2]

	#d = numpy.mean(distDict[str(bonds[0])+'_'+str(bonds[1])][allFrames.max()+1][0])*pixInNM
	#r1 = imgDataDict[bonds[0]][allFrames.max()+1]['effRadius'][0]*pixInNM
	#r2 = imgDataDict[bonds[1]][allFrames.max()+1]['effRadius'][0]*pixInNM

	#outFile.write("%s " %(str(bonds[0])+'_'+str(bonds[1])))
	#outFile.write("%f %f %f\n" %(d, r1, r2))

#outFile.close()


import sys
import os
import numpy
import matplotlib.pyplot as plt
import cv2
import math
import itertools
from time import time
from scipy import ndimage
from skimage.filters import threshold_otsu, rank
from skimage import data
from skimage.morphology import disk, skeletonize
import time
import gc
#from mpi4py import MPI

import scipy
import cPickle as pickle
import mahotas.polygon
import matplotlib.cm as cm

sys.path.append(os.path.abspath('../myClasses'))
from EMImaging import dataProcessing
from EMImaging import segmentation

sys.path.append(os.path.abspath('../myFunctions'))
import myCythonFunc
import myPythonFunc
import mpiFunctions

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


###########################################################
inputDir = r'E:\Acads\KOHEtching\2.0%KOH\D08\12-13_20150515'
inputFile = r'E:\Acads\KOHEtching\2.0%KOH\D08\12-13_20150515\12-13_20150515.avi'

magnification = 20000
numPixels = 1024 #1024,866,480
fps = 10
microscopeName = 'JOEL2010' #'JOEL2010','T12'
fontScale = 1
structure = [[0,1,0],[1,1,1],[0,1,0]]
###########################################################

###########################################################
if (rank == 0):
	if (os.path.exists(inputDir+'/output.h5'):
		fp = h5py.File(inputDir+'/output.h5','r')
	else:
		fp = h5py.File(inputDir+'/output.h5', 'w')
###########################################################

###########################################################
#DATA REFINING
###########################################################
bgSubFlag=False; invertFlag=True; weightFlag=False;
bgSubMethod='none'; sigmaBackground=20; alpha=0.8; ringRange=[[3,50]]
sigmaTHT=4; radiusTHT=30

methodList = ['gauss'] #['median','gauss','medianOFgauss','gaussOFmedian','none']
gaussFilterSizeList = [2]
medianFilterSizeList = [5,10]

if (rank==0):
	mpiFunctions.initializeMetaData(fp, inputDir, inputFile, magnification, numPixels, fps, microscopeName)
	gImgRawStack = mpiFunctions.readTIFF(inputFile)
	
	
	hP = dataProcessing(inputDir, inputFile, magnification, numPixels=numPixels, fps=fps, microscopeName=microscopeName)
	hP.readTIFF(inputFile)
	frameList = range(1,hP.numFrames+1)
	hP.createDatasets(bgSubFlag=bgSubFlag, invertFlag=invertFlag, weightFlag=weightFlag,\
					  bgSubMethod=bgSubMethod, sigmaBackground=sigmaBackground, alpha=alpha, ringRange=ringRange,\
					  methodList=methodList, gaussFilterSizeList=gaussFilterSizeList, medianFilterSizeList=medianFilterSizeList,\
					  sigmaTHT=sigmaTHT, radiusTHT=radiusTHT,\
					  frameList=frameList)
	del hP
	gc.collect()
##############################################################
###############################################################




##############################################################
##############################################################
#PERFORMING SEGMENTATION
##############################################################
##############################################################
#methodList = ['gauss']#['median','gauss','medianOFgauss','gaussOFmedian','none']
#gaussFilterSizeList = [2]
#medianFilterSizeList = [5]

#maskFlag=False; maskMethod='none'; maskMethodMode='global'; maskIntensityRange=[0,255]; maskExcludeZero=False; maskStrelSize=0
#gradFlag=True; gradMethod='sobel'; gradMode='hvd'; gradOper=numpy.array([[-1,0,1],[-2,0,2],[-1,0,1]]); gradExclude=numpy.array([0],dtype='uint8')
#gradThreshFlag=True; gradThMethod='mean'; gradThMode='global'; gradThRange=[0,255]; gradThExcludeZero=True; gradThStrelSize=20
#cleanEdgeMethod='neighbors'; minNeighbors=2
#intensityThreshFlag=True; intThMaskFlag=True; intThMethod='otsu'; intThMode='adaptive1'; intThRange=[0,255]; intThExcludeZero=True; intThStrelSize=20
#widthRange=numpy.array([300,850], dtype='float64'); heightRange=numpy.array([150,670], dtype='float64')
#areaRange = numpy.array([400,15000], dtype='float64')
#circularityRange = numpy.array([0.7,1], dtype='float')

#seg = segmentation(inputDir, methodList, gaussFilterSizeList, medianFilterSizeList)
#frameList = metaData['frameList']
#for key in seg.dict.keys():
	#for frame in frameList:
		#if ((frame-1)%size == rank):
			#if (rank == 0):
				#print frame,"/",frameList[-1]
			#gImgRaw = myPythonFunc.normalize(numpy.load(inputDir+'/output/data/dataProcessing/gImgRawStack/'+str(frame)+'.npy'), max=230)
			#gImg = numpy.load(seg.dict[key]['dataDir']+'/'+str(frame)+'.npy')
			
			#bImg = myPythonFunc.intensityThreshold(img=gImg, flag=True, maskFlag=False, method='otsu', mode='global', intensityRange=intThRange, excludeZero=False)
			#bImg = myCythonFunc.removeBoundaryParticles(bImg.astype('uint8'))
			#bImg = myPythonFunc.binary_opening(bImg, iterations=8)
			#bImg = myCythonFunc.areaThreshold(bImg.astype('uint8'), areaRange=areaRange)
			#bImg = ndimage.binary_fill_holes(bImg)
			#bImg = myCythonFunc.circularThreshold(bImg.astype('uint8'), circularityRange=circularityRange)
			
			#labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg)
			#finalBImg = bImg.copy(); finalBImg[:] = False
			#for label in range(1,numLabel+1):
				#bImgN = labelImg==label
				#intensities = gImg[numpy.where(bImgN==True)]
				#bImg2 = gImg >= myPythonFunc.otsuThreshold(intensities, bins=256, range=(0,255))
				#bImg2 = ndimage.binary_fill_holes(bImg2)
				#bImg2 = myCythonFunc.areaThreshold(bImg2.astype('uint8'), areaRange=areaRange)
				#bImg2 = ndimage.binary_erosion(bImg2, iterations=2)
				#bImg2 = myCythonFunc.areaThreshold(bImg2.astype('uint8'), areaRange=areaRange)
				#bImg2 = ndimage.binary_dilation(bImg2, iterations=4)
				#bImg2 = numpy.logical_and(bImgN, bImg2)
				#finalBImg = numpy.logical_or(finalBImg, bImg2)
				
			#bImg = finalBImg.copy()
			
			#labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, circularity=True)
			#for label,circularity in zip(dictionary['id'],dictionary['circularity']):
				#if (circularity<0.85):
					#labelImg[labelImg==label] = 0
			#bImg = labelImg.astype('bool')
			#bImg = myCythonFunc.areaThreshold(bImg.astype('uint8'), areaRange=areaRange)
			
			#bImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(bImg))
	
			#labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, centroid=True)
			#bImgTagged = myPythonFunc.normalize(bImg)
			#for j in range(len(dictionary['id'])):
				#cv2.putText(bImgTagged, str(dictionary['id'][j]), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=2, bottomLeftOrigin=False)
	
			#finalImage = numpy.column_stack((numpy.maximum(gImgRaw,bImgBdry), bImgTagged))
			#cv2.imwrite(seg.dict[key]['imgDir']+'/'+str(frame)+'.png', finalImage)
##############################################################
##############################################################



##############################################################
##############################################################
#CREATING BINARY IMAGE STACK FROM FINAL SEGMENTATION
##############################################################
##############################################################
#if (rank==0):
	#print 'CREATING bImgStack'
#[row,col,numFrames] = [metaData['row'], metaData['col'], metaData['numFrames']]
#frameList = metaData['frameList']
#myPythonFunc.mkdir(inputDir+'/output/data/segmentation/final')

#for frame in metaData['frameList']:
	#if ((frame-1)%size == rank):
		#img = cv2.imread(inputDir+'/output/images/segmentation/final/'+str(frame)+'.png',0)[0:row,0:col]
		#bImg = ndimage.binary_fill_holes(img==255)
		#numpy.save(inputDir+'/output/data/segmentation/final/'+str(frame)+'.npy', bImg)
##############################################################
##############################################################



##############################################################
##############################################################
#LABELLING PARTICLES
##############################################################
##############################################################
#if (rank==0):
	#print "LABELLING PARTICLES"
	#[row,col,numFrames] = [metaData['row'], metaData['col'], metaData['numFrames']]
	
	#centerDispRange = [50,55]
	#perAreaChangeRange = [1,2]
	#missFramesTh = 100
	#frameList = metaData['frameList']
	
	#bImgDir = inputDir+'/output/data/segmentation/final'
	#gImgDir = inputDir+'/output/data/dataProcessing/gImgRawStack'
	#labelDataDir = inputDir+'/output/data/segmentation/tracking'
	#labelImgDir = inputDir+'/output/images/segmentation/tracking'
	
	#myPythonFunc.mkdir(inputDir+'/output/data/segmentation/tracking')
	#myPythonFunc.mkdir(inputDir+'/output/images/segmentation/tracking')
	#maxID = myPythonFunc.labelParticles(bImgDir, gImgDir, labelDataDir, labelImgDir, row, col, numFrames, centerDispRange, perAreaChangeRange, missFramesTh, frameList, structure, fontScale=fontScale)
	#metaData['particleList'] = range(1,maxID+1)
	#pickle.dump(metaData, open(inputDir+'/metaData', 'wb'))
##############################################################
##############################################################



##############################################################
##############################################################
#REMOVING UNWANTED PARTICLES
##############################################################
##############################################################
#if (rank==0):
	#print "REMOVING UNWANTED PARTICLES"
#frameList = metaData['frameList']
#particleList = metaData['particleList']

#keepList = range(1,31)+[32]+range(34,39)
#removeList = []

#if not removeList:
	#removeList = [s for s in particleList if s not in keepList]

#for frame in frameList:
	#if ((frame-1)%size == rank):
		#if (rank==0):
			#print frame,"/",frameList[-1]
		#labelImg = numpy.load(inputDir+'/output/data/segmentation/tracking/'+str(frame)+'.npy')
		#gImg = numpy.load(inputDir+'/output/data/dataProcessing/gImgRawStack/'+str(frame)+'.npy')
		#for r in removeList:
			#labelImg[labelImg==r] = 0
		#bImg = labelImg.astype('bool')
		#bImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(bImg))
		#label, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, structure=structure, centroid=True)
		#bImg = myPythonFunc.normalize(bImg)
		#for j in range(len(dictionary['id'])):
			#bImgLabelN = label == dictionary['id'][j]
			#ID = numpy.max(bImgLabelN*labelImg)
			#cv2.putText(bImg, str(ID), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=2, bottomLeftOrigin=False)
		#finalImage = numpy.column_stack((bImg, numpy.maximum(bImgBdry,gImg)))
		#numpy.save(inputDir+'/output/data/segmentation/tracking/'+str(frame)+'.npy', labelImg)
		#cv2.imwrite(inputDir+'/output/images/segmentation/tracking/'+str(frame)+'.png', finalImage)
		
#for r in removeList:
	#try:
		#metaData['particleList'].remove(r)
	#except:
		#pass
#if (rank==0):
	#pickle.dump(metaData, open(inputDir+'/metaData', 'wb'))
##############################################################
##############################################################



##############################################################
##############################################################
#LABEL CORRECTION
##############################################################
##############################################################
#if (rank==0):
	#print "LABEL CORRECTION"
#[row, col, numFrames] = [metaData['row'], metaData['col'], metaData['numFrames']]
#frameList = metaData['frameList']

#correctionList = []
#frameWiseCorrectionList = [\
#[range(1,63),[1,0]],\
#[range(1,90),[2,0]]\
#]

#for frame in frameList:
	#if ((frame-1)%size == rank):
		#labelImg = numpy.load(inputDir+'/output/data/segmentation/tracking/'+str(frame)+'.npy')
		#for i in range(len(correctionList)):
			#for j in range(len(correctionList[i])-1):
				#labelImg[labelImg==correctionList[i][j]] = correctionList[i][-1]
		#numpy.save(inputDir+'/output/data/segmentation/tracking/'+str(frame)+'.npy', labelImg)
#for i in correctionList:
	#for j in i[:-1]:
		#try:
			#metaData['particleList'].remove(j)
		#except:
			#pass
			
#for frameWiseCorrection in frameWiseCorrectionList:
	#subFrameList, subCorrectionList = frameWiseCorrection[0], frameWiseCorrection[1]
	#for frame in subFrameList:
		#if ((frame-1)%size == rank):
			#labelImg = numpy.load(inputDir+'/output/data/segmentation/tracking/'+str(frame)+'.npy')
			#newLabel = subCorrectionList[-1]
			#for oldLabel in subCorrectionList[:-1]:
				#labelImg[labelImg==oldLabel] = newLabel
			#numpy.save(inputDir+'/output/data/segmentation/tracking/'+str(frame)+'.npy', labelImg)
			
#maxLabel = max(metaData['particleList'])+1; counter=1

#newLabels = {}
#for particle in metaData['particleList']:
	#newLabels[particle]=[]

#for frame in frameList:
	#particlesInFrame = numpy.unique(numpy.load(inputDir+'/output/data/segmentation/tracking/'+str(frame)+'.npy'))[1:]
	#for p in particlesInFrame:
		#if not newLabels[p]:
			#newLabels[p] = [maxLabel, counter]
			#maxLabel+=1; counter+=1

#for frame in frameList:
	#if ((frame-1)%size == rank):
		#labelImg = numpy.load(inputDir+'/output/data/segmentation/tracking/'+str(frame)+'.npy')
		#gImg = numpy.load(inputDir+'/output/data/dataProcessing/gImgRawStack/'+str(frame)+'.npy')
		#for keys in newLabels.keys():
			#labelImg[labelImg==keys] = newLabels[keys][0]
		#for keys in newLabels.keys():
			#labelImg[labelImg==newLabels[keys][0]] = newLabels[keys][1]
		#bImg = labelImg.astype('bool')
		#bImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(bImg))
		#label, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, structure=structure, centroid=True)
		#bImg = myPythonFunc.normalize(bImg)
		#for j in range(len(dictionary['id'])):
			#bImgLabelN = label == dictionary['id'][j]
			#ID = numpy.max(bImgLabelN*labelImg)
			#cv2.putText(bImg, str(ID), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=2, bottomLeftOrigin=False)
		#finalImage = numpy.column_stack((bImg, numpy.maximum(bImgBdry,gImg)))
		#numpy.save(inputDir+'/output/data/segmentation/tracking/'+str(frame)+'.npy', labelImg)
		#cv2.imwrite(inputDir+'/output/images/segmentation/tracking/'+str(frame)+'.png', finalImage)

#metaData['particleList']=[]
#for key in newLabels.keys():
	#metaData['particleList'].append(newLabels[key][1])
#if (rank==0):
	#metaData['particleList'].sort()
	#pickle.dump(metaData, open(inputDir+'/metaData', 'wb'))
##############################################################
##############################################################



##############################################################
##############################################################
#FINDING MEASURES FOR TRACKED PARTICLES
##############################################################
##############################################################
rank = 0
if (rank==0):
	print "FINDING MEASURES FOR TRACKED PARTICLES"
	[row, col, numFrames] = [metaData['row'], metaData['col'], metaData['numFrames']]
	fps = metaData['fps']
	pixInNM = metaData['pixInNM']
	frameList = metaData['frameList']
	particleList = metaData['particleList']
	
	myPythonFunc.mkdir(inputDir+'/output/data/segmentation/measures')
	outFile = open(inputDir+'/output/data/segmentation/measures/imgDataNM.dat', 'wb')
	
	area=True
	perimeter=True
	circularity=True
	pixelList=False
	bdryPixelList=False
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
	
	for frame in frameList:
		print frame,"/",frameList[-1]
		labelImg = numpy.load(inputDir+'/output/data/segmentation/tracking/'+str(frame)+'.npy')
		gImg = numpy.load(inputDir+'/output/data/dataProcessing/gImgRawStack/'+str(frame)+'.npy')
		outFile.write("%f " %(1.0*frame/fps))
		for particle in particleList:
			bImg = labelImg==particle
			if (bImg.max() == True):
				label, numLabel, dictionary = myPythonFunc.regionProps(bImg, gImg, structure=structure, centroid=centroid, area=area, perimeter=perimeter, circularity=circularity, pixelList=pixelList, bdryPixelList=bdryPixelList, effRadius=effRadius, radius=radius, circumRadius=circumRadius, inRadius=inRadius, radiusOFgyration=radiusOFgyration)
				outFile.write("%f %f %f %f %f %f " %(dictionary['centroid'][0][1]*pixInNM, (row-dictionary['centroid'][0][0])*pixInNM, dictionary['area'][0]*pixInNM*pixInNM, dictionary['perimeter'][0]*pixInNM, dictionary['circularity'][0], dictionary['effRadius'][0]*pixInNM))
				imgDataDict[particle][frame]={}; imgDataDict[particle][frame]=dictionary
			else:
				pass
				outFile.write("nan nan nan nan nan nan ")
		outFile.write("\n")
	pickle.dump(imgDataDict, open(inputDir+'/output/data/segmentation/measures/imgDataDict.dat', 'wb'))
	outFile.close()
##############################################################
##############################################################

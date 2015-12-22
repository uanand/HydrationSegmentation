import sys, os
import cv2
import numpy
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
import os
import math
import myCythonFunc
import skimage
from skimage import measure
from itertools import combinations
from scipy import ndimage
from mahotas.polygon import fill_convexhull

sys.path.append(os.path.abspath('../myClasses'))


###########################################################
# MAKE A DIRECTORY AT A CERTAIN LOCATION
###########################################################
def mkdir(dirName):
	if (os.path.exists(dirName) == False):
		os.makedirs(dirName)
###########################################################


###########################################################
# FIND PIXEL SIZE FOR A GIVEN MAGNIFICATION AND MICROSCOPE
###########################################################
def findPixelWidth(mag,numPixels=1024,microscopeName='JOEL2010'):
	if (microscopeName=='JOEL2010'):
		if(mag==1200):
			FOV=8773.20
		elif(mag==4000):
			FOV=2877.90
		elif(mag==5000):
			FOV=2271.20
		elif(mag==6000):
			FOV=1875.69
		elif(mag==8000):
			FOV=1391.21
		elif(mag==10000):
			FOV=1105.63
		elif(mag==12000):
			FOV=917.32
		elif(mag==15000):
			FOV=730.66
		elif(mag==20000):
			FOV=545.62
		elif(mag==25000):
			FOV=435.36
		elif(mag==30000):
			FOV=362.17
		elif(mag==40000):
			FOV=271.04
		elif(mag==50000):
			FOV=216.56
		elif(mag==60000):
			FOV=180.31
		elif(mag==80000):
			FOV=135.09
		elif(mag==100000):
			FOV=108.00
		elif(mag==120000):
			FOV=89.96
		elif(mag==150000):
			FOV=71.94
		elif(mag==200000):
			FOV=53.93
		elif(mag==250000):
			FOV=42.22
		elif(mag==300000):
			FOV=35.42
		elif(mag==400000):
			FOV=26.48
		elif(mag==500000):
			FOV=20.59
		elif(mag==600000):
			FOV=17.45
		else:
			print 'MICROSCOPE ', microscopeName, ' NOT CALIBRATED FOR THIS MAGNIFICATION.'
	elif (microscopeName=='T12'):
		if (mag==15000):
			FOV=568.30
		elif (mag==21000):
			FOV=405.93
		elif (mag==26000):
			FOV=327.87
		elif (mag==30000):
			FOV=284.15
		elif (mag==42000):
			FOV=202.97
		elif (mag==52000):
			FOV=163.93
		else:
			print 'MICROSCOPE ', microscopeName, ' NOT CALIBRATED FOR THIS MAGNIFICATION.'
	pixInNM = FOV/numPixels
	pixInAngstrom = pixInNM*10
	return pixInNM,pixInAngstrom
###########################################################


###########################################################
# COUNT THE NUMBER OF FRAMES IN TIFF STACK
###########################################################
def countFrames(inputFile):
	print 'COUNTING THE NUMBER OF FRAMES IN TIFF STACK'
	numFrames = 0
	stack = TIFF.open(inputFile, mode='r')
	for gImgRaw in stack.iter_images():
		if (numFrames==0):
			if (len(gImgRaw.shape)==3):
				gImgRaw = gImgRaw[:,:,0]
			[row,col] = gImgRaw.shape
		numFrames+=1
	return row, col, numFrames
###########################################################


###########################################################
# CREATING METADATA FOR INPUT FILE
###########################################################
def initializeMetaData(fp, inputDir, inputFile, magnification, numPixels, fps, microscopeName):
	print 'CREATING METADATA FOR INPUT FILE'
	pixInNM,pixInAngstrom = findPixelWidth(mag=magnification,numPixels,microscopeName)
	row, col, numFrames = countFrames(inputFile)
	fp_grp = fp.create_group('metaData')
	fp_grp.attrs['magnification'] = magnification
	fp_grp.attrs['numPixels'] = numPixels
	fp_grp.attrs['fps'] = fps
	fp_grp.attrs['microscopeName'] = microscopeName
	fp_grp.attrs['pixInNM']=pixInNM
	fp_grp.attrs['pixInAngstrom']=pixInAngstrom
	fp_grp.attrs['row']=row
	fp_grp.attrs['col']=col
	fp_grp.attrs['numFrames']=numFrames
	fp_grp.attrs['frameList']=range(1,numFrames+1)
	
	mkdir(inputDir+'/output')
	#mkdir(inputDir+'/output/data')
	mkdir(inputDir+'/output/images')
	#mkdir(inputDir+'/output/data/dataProcessing')
	mkdir(inputDir+'/output/images/dataProcessing')
###########################################################


###########################################################
# READ THE TIFF IMAGE STACK AND RETURN THE NUMPY 3D STACK
###########################################################
def readTIFF(inputFile):
	print "READING THE TIFF STACK"
	stack = TIFF.open(inputFile, mode='r')
	flag=0
	for gImgRaw in stack.iter_images():
		if (len(gImgRaw.shape)==3):
			gImgRaw = gImgRaw[:,:,0]
		if (flag==0):
			gImgRawStack = gImgRaw
			flag=1
		else:
			gImgRawStack = numpy.dstack((gImgRawStack,gImgRaw))
	return gImgRawStack
###########################################################

def otsuThreshold(img, bins=256, range=(0,255)):
	hist, bins = numpy.histogram(img.flatten(), bins=bins, range=range)
	totalPixels = hist.sum()

	currentMax = 0
	threshold = 0
	sumTotal, sumForeground, sumBackground = 0., 0., 0.
	weightBackground, weightForeground = 0., 0.

	for i,t in enumerate(hist):
		sumTotal += i * hist[i]
	for i,t in enumerate(hist):
		weightBackground += hist[i]
		if( weightBackground == 0 ): continue

		weightForeground = totalPixels - weightBackground
		if ( weightForeground == 0 ): break

		sumBackground += i*hist[i]
		sumForeground = sumTotal - sumBackground
		meanB = sumBackground / weightBackground
		meanF = sumForeground / weightForeground
		varBetween = weightBackground*weightForeground
		varBetween *= (meanB-meanF)*(meanB-meanF)
		if(varBetween > currentMax):
			currentMax = varBetween
			threshold = i
	return threshold


def boundary(bImg):
	bImgErode = ndimage.binary_erosion(bImg)
	bImgBdry = (bImg - bImgErode).astype('bool')
	return bImgBdry


def subtractBackground(gImg, sigma_bg, alpha1, weight=True):
	[row, col] = gImg.shape
	background = ndimage.gaussian_filter(gImg, sigma=(row+col)/(2*sigma_bg)).astype('float64')
	if (weight == True):
		alpha2 = (background - background.min())/(background.max() - background.min())*alpha1
		gImgFilter = gImg - (alpha2*background)
	else:
		gImgFilter = gImg - alpha1*background
	gImgFilter = (gImgFilter -gImgFilter.min())/(gImgFilter.max() - gImgFilter.min())*255
	gImgFilter = gImgFilter.astype('uint8')
	return gImgFilter


def finerObjects(imgLabel, numLabels, imgGrad):
	[row, col] = imgLabel.shape
	bImg = numpy.zeros([row, col], dtype='bool')
	for i in range(1, numLabels+1):
		temp1 = imgLabel == i
		temp2 = imgGrad*temp1
		bImg += (temp2 > otsuThreshold(temp2, bins=255, range=(1,255)))
	[imgLabelNew, numLabelsNew] = ndimage.label(bImg)
	return bImg


def createOutputDir(inputDir):
	if (os.path.exists(inputDir+'/output') == False):
		os.makedirs(inputDir+'/output')
	if (os.path.exists(inputDir+'/output/images') == False):
		os.makedirs(inputDir+'/output/images')
	if (os.path.exists(inputDir+'/output/images/particleIdentification') == False):
		os.makedirs(inputDir+'/output/images/particleIdentification')
	if (os.path.exists(inputDir+'/output/data') == False):
		os.makedirs(inputDir+'/output/data')
	if (os.path.exists(inputDir+'/output/data/particleIdentification') == False):
		os.makedirs(inputDir+'/output/data/particleIdentification')


def findSeeds(contours):
	[row, col] = contours.shape
	seeds = numpy.zeros([row, col], dtype='bool')
	[contoursLabel, numLabel] = ndimage.label(contours, structure=[[1,1,1],[1,1,1],[1,1,1]])
	for i in range(1, numLabel+1):
		contoursLabelN = (contoursLabel == i)
		temp1 = numpy.logical_and(contours, numpy.invert(ndimage.binary_fill_holes(contoursLabelN)))
		temp2 = numpy.logical_and(contoursLabel, numpy.invert(contoursLabelN))
		if (numpy.array_equal(temp1, temp2)):
			seeds += ndimage.binary_fill_holes(contoursLabelN)
	return seeds


#def markNuclei(seeds, bImg):
	#seeds, numSeeds = ndimage.label(seeds, structure=[[1,1,1],[1,1,1],[1,1,1]])
	#dist = ndimage.distance_transform_edt(bImg)
	#dist = dist.max() - dist
	#dist -= dist.min()
	#dist = dist/float(dist.ptp()) * 255
	#dist = dist.astype(numpy.uint8)
	#nuclei = pymorph.cwatershed(dist, seeds)
	#return nuclei


#def intensityThreshold(bImg, gImg, seeds, nuclei):
	#bImgFinal = numpy.copy(bImg)
	#gImgObj = gImg*bImg
	#threshold = 1.0*gImg.sum()/gImg.size
	##threshold = 1.0*gImgObj.sum()/numpy.count_nonzero(gImgObj)
	#for i in range(1, nuclei.max()+1):
		#gImgObjN = gImg * (nuclei == i)# * seeds
		#thresholdN = 1.0*gImgObjN.sum()/numpy.count_nonzero(gImgObjN)
		#if (thresholdN <= threshold):
			#bImgFinal *= numpy.invert(nuclei == i)
	#return bImgFinal


def adaptiveThreshold(gImg, method='none', numPartitions = (1, 1), threshold = 127, excludeZero=False, bins=256, intRange=(0,255)):
	[row, col] = gImg.shape
	bImg = numpy.zeros([row, col], dtype='bool')
	[subRow, subCol] = createPartition(gImg, numPartitions)

	if (method == 'mean'):
		for i in range(numPartitions[0]):
			for j in range(numPartitions[1]):
				gImgROI = gImg[subRow[i,0]:subRow[i,1], subCol[j,0]:subCol[j,1]]
				gImgROIFlat = gImgROI.flatten()
				if (excludeZero == True):
					threshold = numpy.mean(gImgROIFlat[gImgROIFlat > 0])
				else:
					threshold = gImgROIFlat.mean()
				bImgROI = gImgROI > threshold
				bImg[subRow[i,0]:subRow[i,1], subCol[j,0]:subCol[j,1]] = bImgROI
	elif (method == 'median'):
		for i in range(numPartitions[0]):
			for j in range(numPartitions[1]):
				pass
				gImgROI = gImg[subRow[i,0]:subRow[i,1], subCol[j,0]:subCol[j,1]]
				gImgROIFlat = gImgROI.flatten()
				if (excludeZero == True):
					threshold = numpy.median(gImgROIFlat[gImgROIFlat > 0])
				else:
					threshold = numpy.median(gImgROIFlat)
				bImgROI = gImgROI >= threshold
				bImg[subRow[i,0]:subRow[i,1], subCol[j,0]:subCol[j,1]] = bImgROI
	elif (method == 'otsu'):
		pass
		for i in range(numPartitions[0]):
			for j in range(numPartitions[1]):
				gImgROI = gImg[subRow[i,0]:subRow[i,1], subCol[j,0]:subCol[j,1]]
				hist, nbins = numpy.histogram(gImgROI.flatten(), bins, range=intRange)
				totalPixels = hist.sum()

				currentMax = 0
				threshold = 0
				sumTotal, sumForeground, sumBackground = 0., 0., 0.
				weightBackground, weightForeground = 0., 0.

				for i,t in enumerate(hist): sumTotal += i * hist[i]
				for i,t in enumerate(hist):
					weightBackground += hist[i]
					if( weightBackground == 0 ): continue

					weightForeground = totalPixels - weightBackground
					if ( weightForeground == 0 ): break

					sumBackground += i*hist[i]
					sumForeground = sumTotal - sumBackground
					meanB = sumBackground / weightBackground
					meanF = sumForeground / weightForeground
					varBetween = weightBackground*weightForeground
					varBetween *= (meanB-meanF)*(meanB-meanF)
					if(varBetween > currentMax):
						currentMax = varBetween
						threshold = i
				bImgROI = gImgROI >= threshold
				bImg[subRow[i,0]:subRow[i,1], subCol[j,0]:subCol[j,1]] = bImgROI
	elif (method == 'none'):
		for i in range(numPartitions[0]):
			for j in range(numPartitions[1]):
				gImgROI = gImg[subRow[i,0]:subRow[i,1], subCol[j,0]:subCol[j,1]]
				if (i == 0 and j == 0):
					thresholdRef = threshold
					meanRef = gImgROI.mean()
				else:
					threshold = thresholdRef + (gImgROI.mean() - meanRef)
				bImgROI = gImgROI > threshold
				bImg[subRow[i,0]:subRow[i,1], subCol[j,0]:subCol[j,1]] = bImgROI
	return bImg


def createPartition(gImg, numPartitions=(1,1), offsetFlag=False, offsetRatio=1):
	[row, col] = gImg.shape
	subRow = numpy.zeros([numPartitions[0], 2], dtype=int)
	subCol = numpy.zeros([numPartitions[1], 2], dtype=int)
	partition = numpy.zeros([numPartitions[0]*numPartitions[1],4], dtype='int')
	for i in range(numPartitions[0]):
		subRow[i,0] = int(i*(1.0*row/numPartitions[0]))
		subRow[i,1] = int((i+1)*(1.0*row/numPartitions[0]))
		if (i == numPartitions[0]-1):
			if (subRow[i,1] < row):
				subRow[i,1] = row
	for i in range(numPartitions[1]):
		subCol[i,0] = int(i*(1.0*col/numPartitions[1]))
		subCol[i,1] = int((i+1)*(1.0*col/numPartitions[1]))
		if (i == numPartitions[1]-1):
			if (subCol[i,1] < col):
				subCol[i,1] = col
				
	counter=0
	for startRow, endRow in subRow:
		for startCol, endCol in subCol:
			partition[counter,:] = [startRow,endRow,startCol,endCol]
			counter+=1
			
	if (offsetFlag==True):
		initialPartition = partition.copy()
		partitionSize = [partition[0,1]-partition[0,0], partition[0,3]-partition[0,2]]
		
		n=1
		shiftR=[0]; shiftC=[0]
		while (n*offsetRatio < 1):
			shiftR.append(n*offsetRatio*partitionSize[0])
			shiftC.append(n*offsetRatio*partitionSize[1])
			n+=1
			
		for startRow,endRow,startCol,endCol in initialPartition:
			for dR in shiftR:
				for dC in shiftC:
					if (dR+dC>0):
						startR=startRow+dR; endR=endRow+dR
						startC=startCol+dC; endC=endCol+dC
						if (endR<row and endC<col):
							partition = numpy.append(partition, [[startR,endR,startC,endC]], axis=0)
							
	#return subRow, subCol
	return partition
	
	
def createPartition2(gImg, partitionSize=200, offsetRatio=1):
	[row, col] = gImg.shape
	partition=[]
	offset = int(partitionSize*offsetRatio)
	for r in range(0,row,offset):
		for c in range(0,col,offset):
			startRow,startCol,endRow,endCol = r,c,r+partitionSize,c+partitionSize
			if (startRow<=row and startCol<=col and endRow<=row and endCol<=col):
				partition.append([startRow,endRow,startCol,endCol])
	partition = numpy.asarray(partition)
	return partition


def normalize(gImg, min=0, max=255):
	if (gImg.max() > gImg.min()):
		gImg = 1.0*(max-min)*(gImg - gImg.min())/(gImg.max() - gImg.min())
		gImg=gImg+min
	elif (gImg.max() > 0):
		gImg[:] = max
	gImg=gImg.astype('uint8')
	return gImg
	
	
#def adjustOutliers(gImg, min=0, max=255):
	#gImgMin, gImgMax = gImg.min(), gImg.max()
	#gImg[gImg<min], gImg[gImg>max] = min, max
	#return gImg
	
	
	

def cleanContour(contour, mode='edges', minNeighbors=0):
	if (mode == 'neighbors'):
		[row, col] = contour.shape
		contourClean = contour.copy()
		if (mode == 'neighbors'):
			for i in range(1,row-1):
				for j in range(1,col-1):
					if (contour[i,j] == True):
						numNeighbors = contour[i,j-1]*1+contour[i,j+1]*1+contour[i-1,j]*1+contour[i+1,j]*1 + contour[i-1,j-1]*1+contour[i-1,j+1]*1+contour[i+1,j+1]*1+contour[i+1,j-1]*1
						if (numNeighbors <= minNeighbors):
							contourClean[i,j] = False
		return contourClean
	elif (mode == 'edges'):
		[row, col] = contour.shape
		for i in range(row):
			if (contour[i, 0] == True):
				contour = removeContour(contour, row, col, i, 0)
			if (contour[i, col-1] == True):
				contour = removeContour(contour, row, col, i, col-1)
		for i in range(col):
			if (contour[0, i] == True):
				contour = removeContour(contour, row, col, 0, i)
			if (contour[row-1, i] == True):
				contour = removeContour(contour, row, col, row-1, i)
		return contour
	else:
		return contour


def removeContour(contour,row,col,i,j):
	contour[i,j] = False
	if (i == 0):
		if (j == 0):
			primaryNeighbors = contour[i,j+1]*1+contour[i+1,j]*1
			secondaryNeighbors = contour[i+1,j+1]*1
			numNeighbors = primaryNeighbors+secondaryNeighbors
			if (numNeighbors <= 2):
				if (primaryNeighbors in [1,2]):
					if (contour[i,j+1] == True):
						contour = removeContour(contour,row,col,i,j+1)
					if (contour[i+1,j] == True):
						contour = removeContour(contour,row,col,i,j+1)
				elif (secondaryNeighbors == 1):
					if (contour[i+1,j+1] == True):
						contour = removeContour(contour,row,col,i,j+1)
		elif (j == col-1):
			primaryNeighbors = contour[i,j-1]*1+contour[i+1,j]*1
			secondaryNeighbors = contour[i+1,j-1]*1
			numNeighbors = primaryNeighbors+secondaryNeighbors
			if (numNeighbors <= 2):
				if (primaryNeighbors in [1,2]):
					if (contour[i,j-1] == True):
						contour = removeContour(contour,row,col,i,j-1)
					if (contour[i+1,j] == True):
						contour = removeContour(contour,row,col,i+1,j)
				elif (secondaryNeighbors == 1):
					if (contour[i+1,j-1] == True):
						contour = removeContour(contour,row,col,i+1,j-1)
		else:
			primaryNeighbors = contour[i,j-1]*1+contour[i,j+1]+contour[i+1,j]*1
			secondaryNeighbors = contour[i+1,j-1]*1+contour[i+1,j+1]*1
			numNeighbors = primaryNeighbors+secondaryNeighbors
			if (numNeighbors <= 2):
				if (primaryNeighbors in [1,2]):
					if (contour[i,j-1] == True):
						contour = removeContour(contour,row,col,i,j-1)
					if (contour[i,j+1] == True):
						contour = removeContour(contour,row,col,i,j+1)
					if (contour[i+1,j] == True):
						contour = removeContour(contour,row,col,i+1,j)
				elif (secondaryNeighbors == 1):
					if (contour[i+1,j-1] == True):
						contour = removeContour(contour,row,col,i+1,j-1)
					if (contour[i+1,j+1] == True):
						contour = removeContour(contour,row,col,i+1,j+1)
	elif (i == row-1):
		if (j == 0):
			primaryNeighbors = contour[i-1,j]*1+contour[i,j+1]*1
			secondaryNeighbors = contour[i-1,j+1]*1
			numNeighbors = primaryNeighbors+secondaryNeighbors
			if (numNeighbors <= 2):
				if (primaryNeighbors in [1,2]):
					if (contour[i-1,j] == True):
						contour = removeContour(contour,row,col,i-1,j)
					if (contour[i,j+1] == True):
						contour = removeContour(contour,row,col,i,j+1)
				elif (secondaryNeighbors == 1):
					if (contour[i-1,j+1] == True):
						contour = removeContour(contour,row,col,i-1,j+1)
		elif (j == col-1):
			primaryNeighbors = contour[i,j-1]*1+contour[i-1,j]*1
			secondaryNeighbors = contour[i-1,j-1]*1
			numNeighbors = primaryNeighbors+secondaryNeighbors
			if (numNeighbors <= 2):
				if (primaryNeighbors in [1,2]):
					if (contour[i,j-1] == True):
						contour = removeContour(contour,row,col,i,j-1)
					if (contour[i-1,j] == True):
						contour = removeContour(contour,row,col,i-1,j)
				elif (secondaryNeighbors == 1):
					if (contour[i-1,j-1] == True):
						contour = removeContour(contour,row,col,i-1,j-1)
		else:
			primaryNeighbors = contour[i,j-1]*1+contour[i,j+1]+contour[i-1,j]*1
			secondaryNeighbors = contour[i-1,j-1]*1+contour[i-1,j+1]*1
			numNeighbors = primaryNeighbors+secondaryNeighbors
			if (numNeighbors <= 2):
				if (primaryNeighbors in [1,2]):
					if (contour[i,j-1] == True):
						contour = removeContour(contour,row,col,i,j-1)
					if (contour[i,j+1] == True):
						contour = removeContour(contour,row,col,i,j+1)
					if (contour[i-1,j] == True):
						contour = removeContour(contour,row,col,i-1,j)
				elif (secondaryNeighbors == 1):
					if (contour[i-1,j-1] == True):
						contour = removeContour(contour,row,col,i-1,j-1)
					if (contour[i-1,j+1] == True):
						contour = removeContour(contour,row,col,i-1,j+1)
	elif (j == 0):
		primaryNeighbors = contour[i-1,j]*1+contour[i+1,j]*1+contour[i,j+1]*1
		secondaryNeighbors = contour[i-1,j+1]*1+contour[i+1,j+1]*1
		numNeighbors = primaryNeighbors+secondaryNeighbors
		if (numNeighbors <= 2):
			if (primaryNeighbors in [1,2]):
				if (contour[i-1,j] == True):
					contour = removeContour(contour,row,col,i-1,j)
				if (contour[i+1,j] == True):
					contour = removeContour(contour,row,col,i+1,j)
				if (contour[i,j+1] == True):
					contour = removeContour(contour,row,col,i,j+1)
			elif (secondaryNeighbors == 1):
				if (contour[i-1,j+1] == True):
					contour = removeContour(contour,row,col,i-1,j+1)
				if (contour[i+1,j+1] == True):
					contour = removeContour(contour,row,col,i+1,j+1)
	elif (j == col-1):
		primaryNeighbors = contour[i-1,j]*1+contour[i+1,j]*1+contour[i,j-1]*1
		secondaryNeighbors = contour[i-1,j-1]*1+contour[i+1,j-1]*1
		numNeighbors = primaryNeighbors+secondaryNeighbors
		if (numNeighbors <= 2):
			if (primaryNeighbors in [1,2]):
				if (contour[i-1,j] == True):
					contour = removeContour(contour,row,col,i-1,j)
				if (contour[i+1,j] == True):
					contour = removeContour(contour,row,col,i+1,j)
				if (contour[i,j-1] == True):
					contour = removeContour(contour,row,col,i,j-1)
			elif (secondaryNeighbors == 1):
				if (contour[i-1,j-1] == True):
					contour = removeContour(contour,row,col,i-1,j-1)
				if (contour[i+1,j-1] == True):
					contour = removeContour(contour,row,col,i+1,j-1)
	else:
		primaryNeighbors = contour[i,j-1]*1+contour[i,j+1]*1+contour[i-1,j]*1+contour[i+1,j]*1
		secondaryNeighbors = contour[i-1,j-1]*1+contour[i-1,j+1]*1+contour[i+1,j+1]*1+contour[i+1,j-1]*1
		numNeighbors = primaryNeighbors+secondaryNeighbors
		if (numNeighbors <= 2):
			if (primaryNeighbors in [1,2]):
				if (contour[i,j-1] == True):
					contour = removeContour(contour,row,col,i,j-1)
				if (contour[i,j+1] == True):
					contour = removeContour(contour,row,col,i,j+1)
				if (contour[i-1,j] == True):
					contour = removeContour(contour,row,col,i-1,j)
				if (contour[i+1,j] == True):
					contour = removeContour(contour,row,col,i+1,j)
			elif (secondaryNeighbors == 1):
				if (contour[i-1,j-1] == True):
					contour = removeContour(contour,row,col,i-1,j-1)
				if (contour[i-1,j+1] == True):
					contour = removeContour(contour,row,col,i-1,j+1)
				if (contour[i+1,j+1] == True):
					contour = removeContour(contour,row,col,i+1,j+1)
				if (contour[i+1,j-1] == True):
					contour = removeContour(contour,row,col,i+1,j-1)
	return contour


def areaThreshold(bImg, minArea=0, maxArea=1E10):
	labelImg, numLabel = ndimage.label(bImg)
	for i in range(1,numLabel+1):
		bImgLabelN = labelImg == i
		if (bImgLabelN.sum() <= minArea or bImgLabelN.sum() >= maxArea):
			bImg -= bImgLabelN
	return bImg


def removeBoundaryParticles(bImg):
	[row, col] = bImg.shape
	labelImg, numLabel = ndimage.label(bImg)
	for i in range(row):
		if (bImg[i,0] == True):
			bImgLabelN = labelImg == labelImg[i,0]
			bImg -= bImgLabelN
		if (bImg[i,col-1] == True):
			bImgLabelN = labelImg == labelImg[i,col-1]
			bImg -= bImgLabelN
	for i in range(1, col-1):
		if (bImg[0,i] == True):
			bImgLabelN = labelImg == labelImg[0,i]
			bImg -= bImgLabelN
		if (bImg[row-1,i] == True):
			bImgLabelN = labelImg == labelImg[row-1,i]
			bImg -= bImgLabelN
	return bImg


def regionProps(bImg, gImg=0, structure=[[1,1,1],[1,1,1],[1,1,1]], area=False, perimeter=False, circularity=False, pixelList=False, bdryPixelList = False, centroid=False, intensityList=False, sumIntensity=False, avgIntensity=False, maxIntensity=False, effRadius=False, radius=False, theta=False, rTick=False, qTick=False, circumRadius=False, inRadius=False, radiusOFgyration=False, rTickMMM=False, thetaMMM=False):
	[labelImg, numLabel] = ndimage.label(bImg, structure=structure)
	[row, col] = bImg.shape
	dictionary = {}
	dictionary['id'] = []
	if (area == True):
		dictionary['area'] = []
	if (perimeter == True):
		dictionary['perimeter'] = []
	if (circularity == True):
		dictionary['circularity'] = []
	if (pixelList == True):
		dictionary['pixelList'] = []
	if (bdryPixelList == True):
		dictionary['bdryPixelList'] = []
	if (centroid == True):
		dictionary['centroid'] = []
	if (intensityList == True):
		dictionary['intensityList'] = []
	if (sumIntensity == True):
		dictionary['sumIntensity'] = []
	if (avgIntensity == True):
		dictionary['avgIntensity'] = []
	if (maxIntensity == True):
		dictionary['maxIntensity'] = []
	if (effRadius == True):
		dictionary['effRadius'] = []
	if (radius == True):
		dictionary['radius'] = []
	if (circumRadius == True):
		dictionary['circumRadius'] = []
	if (inRadius == True):
		dictionary['inRadius'] = []
	if (radiusOFgyration == True):
		dictionary['radiusOFgyration'] = []
	if (rTick == True):
		dictionary['rTick'] = []
	if (qTick == True):
		dictionary['qTick'] = []
	if (rTickMMM == True):
		dictionary['rTickMean'] = []
		dictionary['rTickMin'] = []
		dictionary['rTickMax'] = []
	if (theta == True):
		dictionary['theta'] = []
	if (thetaMMM == True):
		dictionary['thetaMean'] = []
		dictionary['dThetaP'] = []
		dictionary['dThetaM'] = []

	for i in range(1, numLabel+1):
		bImgLabelN = labelImg == i
		dictionary['id'].append(i)
		if (area == True):
			Area = bImgLabelN.sum()
			dictionary['area'].append(Area)
		if (perimeter == True):
			pmeter = measure.perimeter(bImgLabelN)
			dictionary['perimeter'].append(pmeter)
		if (circularity == True):
			Area = bImgLabelN.sum()
			pmeter = measure.perimeter(bImgLabelN)
			circlarity = (4*numpy.pi*Area)/(pmeter**2)
			if (circlarity>1):
				circlarity=1-(circularity-1)
			dictionary['circularity'].append(circlarity)
		if (pixelList == True):
			pixelsRC = numpy.nonzero(bImgLabelN)
			dictionary['pixelList'].append([pixelsRC[0].tolist(),pixelsRC[1].tolist()])
		if (bdryPixelList == True):
			bdry = boundary(bImgLabelN)
			pixelsRC = numpy.nonzero(bdry)
			dictionary['bdryPixelList'].append([pixelsRC[0].tolist(),pixelsRC[1].tolist()])
		if (centroid == True):
			pixelsRC = numpy.nonzero(bImgLabelN)
			centerRC = [numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1])]
			dictionary['centroid'].append(centerRC)
		#if (wightedCentroid == True):
			#pixelsRC = numpy.nonzero(bImgLabelN)
			#centerRCW = [numpy.average(pixelsRC[0],weights=gImg[pixelsRC]), numpy.average(pixelsRC[1],weights=gImg[pixelsRC])]
			#dictionary['weightedCentroid'].append(centerRCW)
		#if (sigma == True):
			#pixelsRC = numpy.nonzero(bImgLabelN)
			#sigma = min(numpy.std(pixelsRC[0]), numpy.std(pixelsRC[1]))
			#dictionary['sigma'].append(sigma)
		#if (weightedSigma == True):
			#pixelsRC = numpy.nonzero(bImgLabelN)
			#averageW = [numpy.average(pixelsRC[0],weights=gImg[pixelsRC]), numpy.average(pixelsRC[1],weights=gImg[pixelsRC])]
			#sigmaW = min(numpy.sqrt(numpy.average((pixelsRC[0]-average[0])**2, weights=gImg[pixelsRC])), numpy.sqrt(numpy.average((pixelsRC[1]-average[1])**2, weights=gImg[pixelsRC])))
			#dictionary['weightedSigma'].append(sigmaW)
		if (intensityList == True):
			pixelsRC = numpy.nonzero(bImgLabelN)
			intensities = gImg[pixelsRC]
			dictionary['intensityList'].append(intensities)
		if (sumIntensity == True):
			pixelsRC = numpy.nonzero(bImgLabelN)
			sumInt = numpy.sum(gImg[pixelsRC])
			dictionary['sumIntensity'].append(sumInt)
		if (avgIntensity == True):
			pixelsRC = numpy.nonzero(bImgLabelN)
			avgInt = numpy.mean(gImg[pixelsRC])
			dictionary['avgIntensity'].append(avgInt)
		if (maxIntensity == True):
			pixelsRC = numpy.nonzero(bImgLabelN)
			maxInt = numpy.max(gImg[pixelsRC])
			dictionary['maxIntensity'].append(maxInt)
		if (radius == True):
			pixelsRC = numpy.nonzero(bImgLabelN)
			centerRC = (numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1]))
			radii = numpy.max(numpy.sqrt((pixelsRC[0]-centerRC[0])**2 + (pixelsRC[1]-centerRC[1])**2))
			dictionary['radius'].append(radii)
		if (effRadius == True):
			Area = bImgLabelN.sum()
			effRadii = numpy.sqrt(Area/numpy.pi)
			dictionary['effRadius'].append(effRadii)
		if (circumRadius == True):
			pixelsRC = numpy.nonzero(bImgLabelN)
			centerRC = (numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1]))
			bdryPixelsRC = numpy.nonzero(boundary(bImgLabelN))
			radii = numpy.max(numpy.sqrt((bdryPixelsRC[0]-centerRC[0])**2 + (bdryPixelsRC[1]-centerRC[1])**2))
			dictionary['circumRadius'].append(radii)
		if (inRadius == True):
			pixelsRC = numpy.nonzero(bImgLabelN)
			centerRC = (numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1]))
			bdryPixelsRC = numpy.nonzero(boundary(bImgLabelN))
			radii = numpy.min(numpy.sqrt((bdryPixelsRC[0]-centerRC[0])**2 + (bdryPixelsRC[1]-centerRC[1])**2))
			dictionary['inRadius'].append(radii)
		if (radiusOFgyration == True):
			pixelsRC = numpy.nonzero(bImgLabelN)
			centerRC = (numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1]))
			gyration = numpy.sqrt(numpy.average((pixelsRC[0]-centerRC[0])**2 + (pixelsRC[1]-centerRC[1])**2))
			dictionary['radiusOFgyration'].append(gyration)
		if (rTick == True):
			pixelsRC = numpy.nonzero(bImgLabelN)
			centerRC = [numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1])]
			center = [row/2,col/2]
			if (row<=col):
				sc = 1.0*col/row
				qArrScale = col
				dist = numpy.sqrt((sc*(centerRC[0]-center[0]))**2 + (centerRC[1]-center[1])**2)
			else:
				sc = 1.0*row/col
				qArrScale = row
				dist = numpy.sqrt((centerRC[0]-center[0])**2 + (sc*(centerRC[1]-center[1]))**2)
			if (dist==0):
				rTck = 0
			else:
				rTck = qArrScale/dist
			dictionary['rTick'].append(rTck)
		if (qTick == True):
			pixelsRC = numpy.nonzero(bImgLabelN)
			centerRC = [numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1])]
			center = [row/2,col/2]
			if (row<=col):
				sc = 1.0*col/row
				qArrScale = col
				qTck = numpy.sqrt((sc*(centerRC[0]-center[0]))**2 + (centerRC[1]-center[1])**2)
			else:
				sc = 1.0*row/col
				qArrScale = row
				qTck = numpy.sqrt((centerRC[0]-center[0])**2 + (sc*(centerRC[1]-center[1]))**2)
			dictionary['qTick'].append(qTck)
		if (rTickMMM == True):
			pixelsRC = numpy.nonzero(bImgLabelN)
			centerRC = [numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1])]
			center = [row/2,col/2]
			if (row<=col):
				sc = 1.0*col/row
				qArrScale = col
				dist = numpy.sqrt((sc*(centerRC[0]-center[0]))**2 + (centerRC[1]-center[1])**2)
				distAll = numpy.sqrt((sc*(pixelsRC[0]-center[0]))**2 + (pixelsRC[1]-center[1])**2)
			else:
				sc = 1.0*row/col
				qArrScale = row
				dist = numpy.sqrt((centerRC[0]-center[0])**2 + (sc*(centerRC[1]-center[1]))**2)
				distAll = numpy.sqrt((pixelsRC[0]-center[0])**2 + (sc*(pixelsRC[1]-center[1]))**2)
			if (dist==0):
				rTck = 0
			else:
				rTck = qArrScale/dist
			rTckAll = qArrScale/distAll
			dictionary['rTickMean'].append(rTck)
			dictionary['rTickMin'].append(rTckAll.min())
			dictionary['rTickMax'].append(rTckAll.max())
		if (theta == True):
			pixelsRC = numpy.nonzero(bImgLabelN)
			centerRC = (numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1]))
			center = [row/2,col/2]
			angle = numpy.arctan2(center[0]-centerRC[0], centerRC[1]-col/2.0)*180/numpy.pi
			if (angle<0):
				angle = 360+angle
			dictionary['theta'].append(angle)
		if (thetaMMM == True):
			pixelsRC = numpy.nonzero(bImgLabelN)
			centerRC = [numpy.average(pixelsRC[0]), numpy.average(pixelsRC[1])]
			center = [row/2,col/2]
			angle = numpy.arctan2(center[0]-centerRC[0], centerRC[1]-center[1])*180/numpy.pi
			angleAll = numpy.arctan2(center[0]-pixelsRC[0], pixelsRC[1]-center[1])*180/numpy.pi
			if (angle<0):
				angle = 360+angle
			for i in range(len(angleAll)):
				if (angleAll[i]<0):
					angleAll[i] = 360+angleAll[i]
			if (numpy.max([angleAll<10])==True and numpy.max(angleAll>350)==True):
				if (angle < 180):
					dThetaP = numpy.max(angleAll[angleAll<180])-angle
					dThetaM = angle-(numpy.min(angleAll[angleAll>180])-360)
				else:
					dThetaP = numpy.max(angleAll[angleAll<180])-(angle-360)
					dThetaM = angle-numpy.min(angleAll[angleAll>180])
			else:
				dThetaP = numpy.max(angleAll)-angle
				dThetaM = angle-numpy.min(angleAll)
			dictionary['thetaMean'].append(angle)
			dictionary['dThetaP'].append(dThetaP)
			dictionary['dThetaM'].append(dThetaM)
		#if (rTickWSigma == True):
			#pass
		#if (thetaWSigma == True):
			#pass
	return labelImg, numLabel, dictionary



def selectObjects(bImg1, bImg2, bImg3):
	[row, col] = bImg1.shape
	bImg = numpy.zeros([row, col], dtype='bool')
	labelImg1, numLabel1 = ndimage.label(bImg1)
	for i in range(1, numLabel1+1):
		bImgLabelN = labelImg1 == i
		if (numpy.max(numpy.logical_and(bImgLabelN, bImg2)) == False):
			bImg1 -= bImgLabelN

	[labelImg1, numLabel1, properties1] = regionProps(bImg1, circularity=True)
	[labelImg3, numLabel3, properties3] = regionProps(bImg3, circularity=True)

	for i in range(1, numLabel1+1):
		bImgLabel1N = labelImg1 == i
		tempImgLabel, tempNumLabel = ndimage.label(numpy.logical_and(bImgLabel1N, bImg3))
		if (tempNumLabel == 0):
			bImg += bImgLabel1N
		elif (tempNumLabel == 1):
			id1 = i
			id3 = numpy.max(labelImg3[numpy.nonzero(numpy.logical_and(bImgLabel1N, bImg3))])
			if (numpy.abs(1 - properties1['circularity'][id1-1]) <= numpy.abs(1 - properties3['circularity'][id3-1])):
				bImg += labelImg1 == id1
			else:
				bImg += labelImg3 == id3

	for i in range(1, numLabel3+1):
		bImgLabel3N = labelImg3 == i
		if (numpy.max(numpy.logical_and(bImgLabel3N, bImg)) == False):
			bImg += bImgLabel3N

	return bImg


def readStack(inputFile, row, col, numFrames):
	gImgStack = numpy.zeros([row, col, numFrames], dtype='uint8')
	stack = cv2.VideoCapture(inputFile)
	if (stack.isOpened() == True):
		for i in range(numFrames):
			flag, frame = stack.read()
			if (len(frame.shape) == 3):
				gImg = frame[:,:,0]
			elif (len(frame.shape) == 2):
				gImg = frame
			gImgStack[:,:,i] = gImg
	return gImgStack


def readStack(inputFile, row, col, numFrames, sigma_bg, alpha, invert=True, backgroundSubtract=True, weight=True):
	gImgStack = numpy.zeros([row, col, numFrames], dtype='uint8')
	stack = cv2.VideoCapture(inputFile)
	if (stack.isOpened() == True):
		print "Background subtraction"
		for i in range(numFrames):
			print "Processing frame", i+1
			flag, frame = stack.read()
			if (len(frame.shape) == 3):
				if (invert == True):
					gImg = 255 - frame[:,:,0]
				else :
					gImg = frame[:,:,0]
			elif (len(frame.shape) == 2):
				if (invert == True):
					gImg = 255 - frame
				else:
					gImg = frame
			if (backgroundSubtract == True):
				gImg = subtractBackground(gImg, sigma_bg, alpha, weight)
			gImgStack[:,:,i] = gImg
	return gImgStack


def processAllFrames(gImgStack,pixInAngstrom=1):
	ordParamList = []
	[row, col, numFrames] = gImgStack.shape
	[qArr, qTcks, rTcks, pixelTicks] = calculate_qArr(gImgStack[:,:,0],pixInAngstrom)

	for i in range(numFrames):
		fImg = (numpy.abs(numpy.fft.fftshift(numpy.fft.fftn(gImgStack[:,:,i])))**2).astype('double')
		[avg, ordParam, ringVals] = myCythonFunc.angAvg2D_C(fImg, qArr)
		ordParamList.append(ordParam)
	ordParamArr = numpy.array(ordParamList)
	return ordParamArr


def calculate_ordParam(gImg, qArr):
	fImg = (numpy.abs(numpy.fft.fftshift(numpy.fft.fftn(gImg)))**2).astype('double')
	[avg, ordParam, ringVals] = myCythonFunc.angAvg2D_C(fImg, qArr)
	return ordParam



def calculate_qrTcks(ordParamArr, pixInAngstrom, qArrScale):
	lenArr = len(ordParamArr[0])
	qTcks = range(1, lenArr)
	rTcks = [numpy.round((1.*pixInAngstrom*qArrScale)/x,2) for x in qTcks]
	return qTcks, rTcks


def calculate_qArr(gImg,pixInAngstrom=1):
	[row, col] = gImg.shape
	[qpr, qMAXr] = [math.floor(row/2), math.floor((row-1)/2)]
	[qpc, qMAXc] = [math.floor(col/2), math.floor((col-1)/2)]
	[x, y] = numpy.mgrid[-qpr:qMAXr+1, -qpc:qMAXc+1]
	if (row <= col):
	    sc = col/(1.*row)
	    qArrScale = col
	    qArr = numpy.round(numpy.sqrt(sc*sc*x*x + y*y)).astype('int32')
	else:
	    sc = row/(1.*col)
	    qArrScale = row
	    qArr = numpy.round(numpy.sqrt(x*x + sc*sc*y*y)).astype('int32')
	qTcks = range(1, qArr.max()+1)
	rTcks = [(1.*pixInAngstrom*qArrScale)/x for x in qTcks]
	pixelTicks = [(1.*qArrScale)/x for x in qTcks]
	return qArr, qTcks, rTcks, pixelTicks


def plotOrderParam(outputDir, ordParamArr, gImg, pixInAngstrom, gamma=0.4, ringRange = [0, 10000], step_size=10, manualXTicks=[], fps=10):
	[row,col] = ordParamArr.shape
	qArr, qTcks, rTcks, pixelTicks = calculate_qArr(gImg,pixInAngstrom)
	temp0 = [abs(x - ringRange[0]) for x in rTcks]
	temp1 = [abs(x - ringRange[1]) for x in rTcks]
	end = temp0.index(min(temp0))
	start = temp1.index(min(temp1))

	fig = plt.figure(figsize=(3.5,3.5))
	ax = fig.add_axes([0,0,1,1])
	ax.imshow(ordParamArr[:,start:end+1]**gamma, cmap='jet', aspect=(1.0*(end-start+1)/(1.0*row/fps)), origin='lower', extent=[0,end-start,1.0/fps,1.0*row/fps])
	if (manualXTicks):
		rTcksROI = manualXTicks
		xTcksROI = []
		for i in rTcksROI:
			temp = [abs(x - i) for x in rTcks]
			xTcksROI.append(temp.index(min(temp)) - start)
		#plt.axvline(x=xTcksROI[0], linewidth=1, color='w', ls='dashed', alpha=0.7)
		plt.xticks(xTcksROI, numpy.round(rTcksROI, 2), rotation=90)
	else:
		xTcksROI = range(end-start+1)
		rTcksROI = rTcks[start:end+1:1]
		plt.xticks(xTcksROI[::step_size], numpy.round(rTcksROI[::step_size], 1), rotation=90)
	plt.xlabel(ur'Spatial Size (\u00c5)')
	plt.ylabel('Time (s)')
	plt.savefig(outputDir+'/kymo1.pdf', format='pdf', bbox_inches='tight')
	plt.savefig(outputDir+'/kymo1.png', format='png', bbox_inches='tight')
	plt.close()
	
	fig = plt.figure(figsize=(3.5,3.5))
	ax = fig.add_axes([0,0,1,1])
	ax.imshow(ordParamArr[:,start:end+1]**gamma, cmap='jet', aspect=(1.0*(end-start+1)/(1.0*row/fps)), origin='lower', extent=[qTcks[start],qTcks[end],1.0/fps,1.0*row/fps])
	#ax.xaxis.set_ticks(range(qTcks[start],qTcks[end],step_size))
	ax.minorticks_on()
	ax.tick_params(axis='y',which='minor',left='off',right='off')
	ax.set_xlabel('Radial Distance (pixel)')
	ax.set_ylabel('Time (s)')
	plt.savefig(outputDir+'/kymo2.pdf', format='pdf', bbox_inches='tight')
	plt.savefig(outputDir+'/kymo2.png', format='png', bbox_inches='tight')
	plt.close()


def binary_opening(bImg, iterations=1, flag=True):
	if (flag==False):
		return bImg
	else:
		bImg = ndimage.binary_erosion(bImg, iterations=iterations)
		bImg = ndimage.binary_dilation(bImg, iterations=iterations)
		return bImg

def binary_closing(bImg, iterations=1, flag=True):
	if (flag==False):
		return bImg
	else:
		bImg = ndimage.binary_dilation(bImg, iterations=iterations)
		bImg = ndimage.binary_erosion(bImg, iterations=iterations)
		return bImg


def mkdir(dirName):
	if (os.path.exists(dirName) == False):
		os.makedirs(dirName)


def findPixelWidth(mag,numPixels=1024,microscopeName='JOEL2010'):
	if (microscopeName=='JOEL2010'):
		if(mag==1200):
			FOV=8773.20
		elif(mag==4000):
			FOV=2877.90
		elif(mag==5000):
			FOV=2271.20
		elif(mag==6000):
			FOV=1875.69
		elif(mag==8000):
			FOV=1391.21
		elif(mag==10000):
			FOV=1105.63
		elif(mag==12000):
			FOV=917.32
		elif(mag==15000):
			FOV=730.66
		elif(mag==20000):
			FOV=545.62
		elif(mag==25000):
			FOV=435.36
		elif(mag==30000):
			FOV=362.17
		elif(mag==40000):
			FOV=271.04
		elif(mag==50000):
			FOV=216.56
		elif(mag==60000):
			FOV=180.31
		elif(mag==80000):
			FOV=135.09
		elif(mag==100000):
			FOV=108.00
		elif(mag==120000):
			FOV=89.96
		elif(mag==150000):
			FOV=71.94
		elif(mag==200000):
			FOV=53.93
		elif(mag==250000):
			FOV=42.22
		elif(mag==300000):
			FOV=35.42
		elif(mag==400000):
			FOV=26.48
		elif(mag==500000):
			FOV=20.59
		elif(mag==600000):
			FOV=17.45
		else:
			print 'MICROSCOPE ', microscopeName, ' NOT CALIBRATED FOR THIS MAGNIFICATION. EXTRAPOLATING LINEARLY INSTEAD.'
			FOV=35.42*300000/mag
	elif (microscopeName=='T12'):
		if (mag==15000):
			FOV=568.30
		elif (mag==21000):
			FOV=405.93
		elif (mag==26000):
			FOV=327.87
		elif (mag==30000):
			FOV=284.15
		elif (mag==42000):
			FOV=202.97
		elif (mag==52000):
			FOV=163.93
		else:
			print 'MICROSCOPE ', microscopeName, ' NOT CALIBRATED FOR THIS MAGNIFICATION. EXTRAPOLATING LINEARLY INSTEAD.'
			FOV=284.15*30000/mag
	pixInNM = FOV/numPixels
	pixInAngstrom = pixInNM*10
	return pixInNM,pixInAngstrom


def gradient(img, flag=True, method='sobel', mode='hvd', oper=numpy.array([[-1,0,1],[-2,0,2],[-1,0,1]]), exclude=numpy.array([],dtype='uint8')):
	[row,col] = img.shape
	if (method=='custom'):
		if (oper.sum()!=0):
			print 'ERROR: INVALID OPERATOR. \'sobel\' OPERATOR WILL BE USED INSTEAD'
			method='sobel'
		if (mode=='hvd'):
			[r,c] = oper.shape
			if (r!=3 or c!=3):
				print 'ERROR: INVALID OPERATOR SIZE FOR \'hvd\' MODE. \'sobel\' OPERATOR WILL BE USED INSTEAD'
				method='sobel'
	if (mode=='hv'):
		if (method=='sobel'):
			operN = numpy.array([[-1,0,1],[-2,0,2],[-1,0,1]]); operW = numpy.rot90(operN)
			gradN = scipy.signal.convolve(img,operN,mode='same'); gradW = scipy.signal.convolve(img,operW,mode='same')
			grad = numpy.sqrt(gradN**2+gradW**2)
		elif (method=='prewitt'):
			operN = numpy.array([[-1,0,1],[-1,0,1],[-1,0,1]]); operW = numpy.rot90(operN)
			gradN = scipy.signal.convolve(img,operN,mode='same'); gradW = scipy.signal.convolve(img,operW,mode='same')
			grad = numpy.sqrt(gradN**2+gradW**2)
		elif (method=='roberts'):
			operN = numpy.array([[1,0],[0,-1]]); operW = numpy.rot90(operN)
			gradN = scipy.signal.convolve(img,operN,mode='same'); gradW = scipy.signal.convolve(img,operW,mode='same')
			grad = numpy.sqrt(gradN**2+gradW**2)
		elif (method=='regular'):
			operN = numpy.array([[1,0],[-1,0]]); operW = numpy.array([[1,-1],[0,0]])
			gradN = scipy.signal.convolve(img,operN,mode='same'); gradW = scipy.signal.convolve(img,operW,mode='same')
			grad = numpy.sqrt(gradN**2+gradW**2)
		elif (method=='custom'):
			operN = oper; operW = numpy.rot90(operN)
			gradN = scipy.signal.convolve(img,operN,mode='same'); gradW = scipy.signal.convolve(img,operW,mode='same')
			grad = numpy.sqrt(gradN**2+gradW**2)
		elif (method=='none'):
			grad=img
		else:
			print "WARNING: NO SUCH EDGE DETECTION METHOD IN \'hv\' MODE AVAILABLE. USING \'sobel\' METHOD INSTEAD."
			operN = numpy.array([[-1,0,1],[-2,0,2],[-1,0,1]]); operW = numpy.rot90(operN)
			gradN = scipy.signal.convolve(img,operN,mode='same'); gradW = scipy.signal.convolve(img,operW,mode='same')
			grad = numpy.sqrt(gradN**2+gradW**2)
	elif (mode=='hvd'):
		if (method=='sobel'):
			operN  = numpy.array([[-1,0,1],[-2,0,2],[-1,0,1]]); gradN  = numpy.abs(scipy.signal.convolve(img,operN ,mode='same'))
			operNW = rot45(operN)                             ; gradNW = numpy.abs(scipy.signal.convolve(img,operNW,mode='same'))
			operW  = rot45(operNW)                            ; gradW  = numpy.abs(scipy.signal.convolve(img,operW ,mode='same'))
			operSW = rot45(operW)                             ; gradSW = numpy.abs(scipy.signal.convolve(img,operSW,mode='same'))
			operS  = rot45(operSW)                            ; gradS  = numpy.abs(scipy.signal.convolve(img,operS ,mode='same'))
			operSE = rot45(operS)                             ; gradSE = numpy.abs(scipy.signal.convolve(img,operSE,mode='same'))
			operE  = rot45(operSE)                            ; gradE  = numpy.abs(scipy.signal.convolve(img,operE ,mode='same'))
			operNE = rot45(operE)                             ; gradNE = numpy.abs(scipy.signal.convolve(img,operNE,mode='same'))
			grad = numpy.maximum(gradN,numpy.maximum(gradNW,numpy.maximum(gradW,numpy.maximum(gradSW,numpy.maximum(gradS,numpy.maximum(gradSE,numpy.maximum(gradE,gradNE)))))))
		elif (method=='prewitt'):
			operN  = numpy.array([[-1,0,1],[-1,0,1],[-1,0,1]]); gradN  = numpy.abs(scipy.signal.convolve(img,operN ,mode='same'))
			operNW = rot45(operN)                             ; gradNW = numpy.abs(scipy.signal.convolve(img,operNW,mode='same'))
			operW  = rot45(operNW)                            ; gradW  = numpy.abs(scipy.signal.convolve(img,operW ,mode='same'))
			operSW = rot45(operW)                             ; gradSW = numpy.abs(scipy.signal.convolve(img,operSW,mode='same'))
			operS  = rot45(operSW)                            ; gradS  = numpy.abs(scipy.signal.convolve(img,operS ,mode='same'))
			operSE = rot45(operS)                             ; gradSE = numpy.abs(scipy.signal.convolve(img,operSE,mode='same'))
			operE  = rot45(operSE)                            ; gradE  = numpy.abs(scipy.signal.convolve(img,operE ,mode='same'))
			operNE = rot45(operE)                             ; gradNE = numpy.abs(scipy.signal.convolve(img,operNE,mode='same'))
			grad = numpy.maximum(gradN,gradNW,gradW,gradSW,gradS,gradSE,gradE,gradNE)
		elif (method=='custom'):
			operN  = oper         ; gradN  = numpy.abs(scipy.signal.convolve(img,operN ,mode='same'))
			operNW = rot45(operN) ; gradNW = numpy.abs(scipy.signal.convolve(img,operNW,mode='same'))
			operW  = rot45(operNW); gradW  = numpy.abs(scipy.signal.convolve(img,operW ,mode='same'))
			operSW = rot45(operW) ; gradSW = numpy.abs(scipy.signal.convolve(img,operSW,mode='same'))
			operS  = rot45(operSW); gradS  = numpy.abs(scipy.signal.convolve(img,operS ,mode='same'))
			operSE = rot45(operS) ; gradSE = numpy.abs(scipy.signal.convolve(img,operSE,mode='same'))
			operE  = rot45(operSE); gradE  = numpy.abs(scipy.signal.convolve(img,operE ,mode='same'))
			operNE = rot45(operE) ; gradNE = numpy.abs(scipy.signal.convolve(img,operNE,mode='same'))
			grad = numpy.maximum(gradN,numpy.maximum(gradNW,numpy.maximum(gradW,numpy.maximum(gradSW,numpy.maximum(gradS,numpy.maximum(gradSE,numpy.maximum(gradE,gradNE)))))))
		elif (method=='none'):
			grad=img
		else:
			print "WARNING: NO SUCH EDGE DETECTION METHOD IN \'hvd\' MODE AVAILABLE. USING \'sobel\' METHOD INSTEAD."
			operN  = numpy.array([[-1,0,1],[-2,0,2],[-1,0,1]]); gradN  = numpy.abs(scipy.signal.convolve(img,operN ,mode='same'))
			operNW = rot45(operN)                             ; gradNW = numpy.abs(scipy.signal.convolve(img,operNW,mode='same'))
			operW  = rot45(operNW)                            ; gradW  = numpy.abs(scipy.signal.convolve(img,operW ,mode='same'))
			operSW = rot45(operW)                             ; gradSW = numpy.abs(scipy.signal.convolve(img,operSW,mode='same'))
			operS  = rot45(operSW)                            ; gradS  = numpy.abs(scipy.signal.convolve(img,operS ,mode='same'))
			operSE = rot45(operS)                             ; gradSE = numpy.abs(scipy.signal.convolve(img,operSE,mode='same'))
			operE  = rot45(operSE)                            ; gradE  = numpy.abs(scipy.signal.convolve(img,operE ,mode='same'))
			operNE = rot45(operE)                             ; gradNE = numpy.abs(scipy.signal.convolve(img,operNE,mode='same'))
			grad = numpy.maximum(gradN,numpy.maximum(gradNW,numpy.maximum(gradW,numpy.maximum(gradSW,numpy.maximum(gradS,numpy.maximum(gradSE,numpy.maximum(gradE,gradNE)))))))
	if (method!='none'):
		temp = grad.min()
		grad[0,:]=temp; grad[row-1,:]=temp;
		grad[:,0]=temp; grad[:,col-1]=temp;
		grad = normalize(grad)
		grad = myCythonFunc.removeEdges(img, grad, exclude=exclude)
	return grad


def rot45(operator):
	newOperator = numpy.zeros([3,3],dtype='int')
	newOperator[0][0]=operator[0][1]
	newOperator[0][1]=operator[0][2]
	newOperator[0][2]=operator[1][2]
	newOperator[1][0]=operator[0][0]
	newOperator[1][1]=operator[1][1]
	newOperator[1][2]=operator[2][2]
	newOperator[2][0]=operator[1][0]
	newOperator[2][1]=operator[2][0]
	newOperator[2][2]=operator[2][1]
	return newOperator


def intensityThreshold(img, mask=1, flag=True, maskFlag=False, method='none', mode='global', intensityRange=[0,255], excludeZero=False, strelSize=0):
	if (flag==True):
		if (mode=='global'):
			if (maskFlag==True):
				img=img*mask
			if (method=='mean'):
				if (excludeZero==True):
					threshold = numpy.mean(img[img>0])
				else:
					threshold = numpy.mean(img)
				return(img>=threshold)
			elif (method=='median'):
				if (excludeZero==True):
					threshold = numpy.median(img[img>0])
				else:
					threshold = numpy.median(img)
				return(img>=threshold)
			elif (method=='otsu'):
				if (excludeZero==True):
					threshold = skimage.filters.threshold_otsu(img[img>0])
				else:
					threshold = skimage.filters.threshold_otsu(img)
				return(img>=threshold)
			elif (method=='custom'):
				bImg = numpy.logical_and(img>=intensityRange[0],img<=intensityRange[1])
			elif (method=='none'):
				return img
			else:
				print 'ERROR', method, 'IS NOT A VALID METHOD. RETURNING THE ORIGINAL IMAGE'
				return img
		elif (mode=='adaptive1'):
			if (maskFlag==False):
				mask=img; mask[:]=1
			return(myCythonFunc.thresholdAdaptive1(img, mask, method=method, excludeZero=excludeZero, strelSize=strelSize))
		elif (mode=='adaptive2'):
			if (maskFlag==False):
				mask=img; mask[:]=1
			return(myCythonFunc.thresholdAdaptive2(img, mask, method=method, excludeZero=excludeZero, strelSize=strelSize))
		else:
			print 'ERROR', mode, 'IS NOT A VALID MODE. RETURNING THE ORIGINAL IMAGE'
			return img
	else:
		return img


def thresholdAdaptive1(img, method='none', excludeZero=False, strelSize=0):
	[row, col] = img.shape
	if (method=='none'):
		return img
	else:
		bImg = numpy.zeros([row,col], dtype='bool')
		padImg = numpy.pad(img,pad_width=strelSize,mode='reflect')
		for r in range(strelSize,row+strelSize):
			for c in range(strelSize,col+strelSize):
				imgROI = padImg[r-strelSize:r+strelSize+1,c-strelSize:c+strelSize+1]
				if (method=='mean'):
					if (excludeZero==True):
						threshold = numpy.mean(imgROI[imgROI>0])
					else:
						threshold = numpy.mean(imgROI)
					bImg[r-strelSize,c-strelSize] = img[r-strelSize,c-strelSize]>=threshold
				elif (method=='median'):
					if (excludeZero==True):
						threshold = numpy.median(imgROI[imgROI>0])
					else:
						threshold = numpy.median(imgROI)
					bImg[r-strelSize,c-strelSize] = img[r-strelSize,c-strelSize]>=threshold
				elif (method=='otsu'):
					if (excludeZero==True):
						threshold = skimage.filters.threshold_otsu(imgROI[imgROI>0])
					else:
						threshold = skimage.filters.threshold_otsu(imgROI)
					bImg[r-strelSize,c-strelSize] = img[r-strelSize,c-strelSize]>=threshold
	return bImg


def thresholdAdaptive2(img, method='none', excludeZero=False, strelSize=0):
	[row, col] = img.shape
	if (method=='none'):
		return img
	else:
		bImg = numpy.zeros([row+2*strelSize,col+2*strelSize], dtype='bool')
		padImg = numpy.pad(img,pad_width=strelSize,mode='reflect')
		for r in range(strelSize,row+strelSize):
			for c in range(strelSize,col+strelSize):
				imgROI = padImg[r-strelSize:r+strelSize+1,c-strelSize:c+strelSize+1]
				if (method=='mean'):
					if (excludeZero==True):
						threshold = numpy.mean(imgROI[imgROI>0])
					else:
						threshold = numpy.mean(imgROI)
					bImg[r-strelSize:r+strelSize+1,c-strelSize:c+strelSize+1]=numpy.logical_or(bImg[r-strelSize:r+strelSize+1,c-strelSize:c+strelSize+1],imgROI>=threshold)
				elif (method=='median'):
					if (excludeZero==True):
						threshold = numpy.median(imgROI[imgROI>0])
					else:
						threshold = numpy.median(imgROI)
					bImg[r-strelSize:r+strelSize+1,c-strelSize:c+strelSize+1]=numpy.logical_or(bImg[r-strelSize:r+strelSize+1,c-strelSize:c+strelSize+1],imgROI>=threshold)
				elif (method=='otsu'):
					if (excludeZero==True):
						threshold = skimage.filters.threshold_otsu(imgROI[imgROI>0])
					else:
						threshold = skimage.filters.threshold_otsu(imgROI)
					bImg[r-strelSize:r+strelSize+1,c-strelSize:c+strelSize+1]=numpy.logical_or(bImg[r-strelSize:r+strelSize+1,c-strelSize:c+strelSize+1],imgROI>=threshold)
	return bImg[strelSize:row+strelSize,strelSize:col+strelSize]


def convexHull(bImg):
	label,numLabel=ndimage.label(bImg)
	bImg[:]=False
	for i in range(1,numLabel+1):
		bImgN=label==i
		bImgN=fill_convexhull(bImgN)
		bImg=numpy.logical_or(bImg,bImgN)
	return bImg


def calculateMSD(x,y):
	MSD = []
	lenX=len(x); lenY=len(y)
	if (lenX!=lenY):
		print "ERROR: INPUT PARAMETERS x AND y ARE NOT OF SAME LENGTH."
		return MSD
	elif (lenX==0):
		print "ERROR: INPUT PARAMETER x IS OF 0 LENGTH"
		return MSD
	elif (lenY==0):
		print "ERROR: INPUT PARAMETER y IS OF 0 LENGTH"
		return MSD
	else:
		for dt in range(0,lenX):
			S=0.; N=0
			for i in range(lenX-dt):
				if (not(math.isnan(x[i+dt]) or math.isnan(x[i]) or math.isnan(y[i+dt]) or math.isnan(y[i]))):
					ds = (x[i+dt]-x[i])**2 + (y[i+dt]-y[i])**2
					if (not math.isnan(ds)):
						S+=ds; N+=1
			if (N>0):
				MSD.append(S/N)
			else:
				MSD.append('nan')
		return MSD


def generateStartingPosition(numParticles,numSteps,diaRange=[],density=19300,solventParticleSize=0,frameSize=1024,pixInM=1e-9,startingPosition=[],particleDia=[],randomInitialPositionFlag=True,randomRadiusFlag=True):
	dictionary={}; particleList = range(1,numParticles+1)
	if (randomInitialPositionFlag==False and randomRadiusFlag==False):
		for particle in particleList:
			dictionary[particle]={}
			dictionary[particle]['center']=numpy.empty([numSteps+1,2]); dictionary[particle]['center'][:]=numpy.nan
			dictionary[particle]['diameter']=particleDia[particle-1]; dictionary[particle]['radius']=particleDia[particle-1]/2.
			dictionary[particle]['volume']=(4./3)*numpy.pi*dictionary[particle]['radius']**3
			dictionary[particle]['mass']=density*dictionary[particle]['volume']
			dictionary[particle]['center'][0,0]=startingPosition[particle-1][0]; dictionary[particle]['center'][0,1]=startingPosition[particle-1][1]
			dictionary[particle]['firstFrame']=0; dictionary[particle]['lastFrame']=0
			dictionary[particle]['parent']=[]; dictionary[particle]['child']=[]
			dictionary[particle]['mergeFlag']=False
	elif (randomInitialPositionFlag==False and randomRadiusFlag==True):
		flagStart=False
		for particle in particleList:
			while(flagStart==False):
				flagStart=True
				if (numParticles<2):
					particleDia = diaRange[0]
				else:
					particleDia=numpy.random.rand(numParticles)
					particleDia=(diaRange[1]-diaRange[0])*(particleDia-particleDia.min())/(particleDia.max()-particleDia.min())
					particleDia=particleDia+diaRange[0]
				for particle in particleList:
					dictionary[particle]={}
					dictionary[particle]['center']=numpy.empty([numSteps+1,2]); dictionary[particle]['center'][:]=numpy.nan
					dictionary[particle]['diameter']=particleDia[particle-1]; dictionary[particle]['radius']=particleDia[particle-1]/2.
					dictionary[particle]['volume']=(4./3)*numpy.pi*dictionary[particle]['radius']**3
					dictionary[particle]['mass']=density*dictionary[particle]['volume']
					dictionary[particle]['center'][0,0]=startingPosition[particle-1][0]; dictionary[particle]['center'][0,1]=startingPosition[particle-1][1]
					dictionary[particle]['firstFrame']=0; dictionary[particle]['lastFrame']=0
					dictionary[particle]['parent']=[]; dictionary[particle]['child']=[]
					dictionary[particle]['mergeFlag']=False
				for i,j in combinations(particleList,2):
						Xi=dictionary[i]['center'][0,0]; Yi=dictionary[i]['center'][0,1]
						Xj=dictionary[j]['center'][0,0]; Yj=dictionary[j]['center'][0,1]
						Ri=dictionary[i]['radius']; Rj=dictionary[j]['radius']
						dist=numpy.sqrt((Xi-Xj)**2 + (Yi-Yj)**2)
						if (dist <= (Ri+Rj+solventParticleSize)):
							flagStart=False
							break
	elif (randomInitialPositionFlag==True and randomRadiusFlag==False):
		flagStart=False
		for particle in particleList:
			while(flagStart==False):
				flagStart=True
				startingPosition = numpy.random.randint(0,frameSize,(numParticles,2))*pixInM
				for particle in particleList:
					dictionary[particle]={}
					dictionary[particle]['center']=numpy.empty([numSteps+1,2]); dictionary[particle]['center'][:]=numpy.nan
					dictionary[particle]['diameter']=particleDia[particle-1]; dictionary[particle]['radius']=particleDia[particle-1]/2.
					dictionary[particle]['volume']=(4./3)*numpy.pi*dictionary[particle]['radius']**3
					dictionary[particle]['mass']=density*dictionary[particle]['volume']
					dictionary[particle]['center'][0,0]=startingPosition[particle-1][0]; dictionary[particle]['center'][0,1]=startingPosition[particle-1][1]
					dictionary[particle]['firstFrame']=0; dictionary[particle]['lastFrame']=0
					dictionary[particle]['parent']=[]; dictionary[particle]['child']=[]
					dictionary[particle]['mergeFlag']=False
				for i,j in combinations(particleList,2):
						Xi=dictionary[i]['center'][0,0]; Yi=dictionary[i]['center'][0,1]
						Xj=dictionary[j]['center'][0,0]; Yj=dictionary[j]['center'][0,1]
						Ri=dictionary[i]['radius']; Rj=dictionary[j]['radius']
						dist=numpy.sqrt((Xi-Xj)**2 + (Yi-Yj)**2)
						if (dist <= (Ri+Rj+solventParticleSize)):
							flagStart=False
							break
	elif (randomInitialPositionFlag==True and randomRadiusFlag==True):
		flagStart=False
		for particle in particleList:
			while(flagStart==False):
				flagStart=True
				if (numParticles<2):
					particleDia = diaRange[0]
				else:
					particleDia=numpy.random.rand(numParticles)
					particleDia=(diaRange[1]-diaRange[0])*(particleDia-particleDia.min())/(particleDia.max()-particleDia.min())
					particleDia=particleDia+diaRange[0]
				startingPosition = numpy.random.randint(0,frameSize,(numParticles,2))*pixInM
				for particle in particleList:
					dictionary[particle]={}
					dictionary[particle]['center']=numpy.empty([numSteps+1,2]); dictionary[particle]['center'][:]=numpy.nan
					dictionary[particle]['diameter']=particleDia[particle-1]; dictionary[particle]['radius']=particleDia[particle-1]/2.
					dictionary[particle]['volume']=(4./3)*numpy.pi*dictionary[particle]['radius']**3
					dictionary[particle]['mass']=density*dictionary[particle]['volume']
					dictionary[particle]['center'][0,0]=startingPosition[particle-1][0]; dictionary[particle]['center'][0,1]=startingPosition[particle-1][1]
					dictionary[particle]['firstFrame']=0; dictionary[particle]['lastFrame']=0
					dictionary[particle]['parent']=[]; dictionary[particle]['child']=[]
					dictionary[particle]['mergeFlag']=False
				for i,j in combinations(particleList,2):
						Xi=dictionary[i]['center'][0,0]; Yi=dictionary[i]['center'][0,1]
						Xj=dictionary[j]['center'][0,0]; Yj=dictionary[j]['center'][0,1]
						Ri=dictionary[i]['radius']; Rj=dictionary[j]['radius']
						dist=numpy.sqrt((Xi-Xj)**2 + (Yi-Yj)**2)
						if (dist <= (Ri+Rj+solventParticleSize)):
							flagStart=False
							break
	return dictionary


def applyForce(testDict,testNum,i,j,n,rho,kB,T,A,viscosity,numSteps,tau,scalingFactor,solventParticleSize):
	e=2.718281828459045

	Xi=testDict[testNum][i]['center'][n,0]; Yi=testDict[testNum][i]['center'][n,1]
	Xj=testDict[testNum][j]['center'][n,0]; Yj=testDict[testNum][j]['center'][n,1]
	Ri=testDict[testNum][i]['radius']; Rj=testDict[testNum][j]['radius']
	Mi=testDict[testNum][i]['mass']; Mj=testDict[testNum][j]['mass']
	dist=numpy.sqrt((Xi-Xj)**2 + (Yi-Yj)**2)
	minDist=dist-(Ri+Rj)
	mu=Mi*Mj/(Mi+Mj)

	c1=(rho*kB*T*A)/mu; c2=(6*numpy.pi*viscosity*Ri)/mu

	counter=0
	while(n<numSteps):
		dTravel=(c1/c2)*(tau*(counter+1)+(1/c2)*(e**(-c2*tau*(counter+1))-1)) - (c1/c2)*(tau*(counter)+(1/c2)*(e**(-c2*tau*(counter))-1))
		#dTravel=((c1/c2)*(tau*(counter+1)+(1/c2)*(e**(-c2*tau*(counter+1))-1)-(c1/c2)*(tau*counter+(1/c2)*(e**(-c2*tau*counter)-1)))
		alpha_i=(numpy.minimum(minDist,dTravel)*Mj)/(Mi+Mj); alpha_j=(numpy.minimum(minDist,dTravel)*Mi)/(Mi+Mj);
		dXi=alpha_i*(Xj-Xi)/dist; dYi=alpha_i*(Yj-Yi)/dist
		dXj=alpha_j*(Xi-Xj)/dist; dYj=alpha_j*(Yi-Yj)/dist
		testDict[testNum][i]['center'][n+1][0]=Xi+dXi; testDict[testNum][i]['center'][n+1][1]=Yi+dYi
		testDict[testNum][j]['center'][n+1][0]=Xj+dXj; testDict[testNum][j]['center'][n+1][1]=Yj+dYj
		testDict[testNum][i]['lastFrame']=n+1; testDict[testNum][i]['lastFrame']=n+1

		print dTravel, minDist

		if (dTravel>=minDist):
			testDict[testNum][i]['mergeFlag']=True; testDict[testNum][j]['mergeFlag']=True
			testDict[testNum][i]['lastFrame']=n; testDict[testNum][j]['lastFrame']=n
			return testDict
			break

		if (testDict[testNum][i]['mergeFlag']==False):
			for particle in [i,j]:
				dR=scalingFactor*numpy.random.randn(1)[0]; theta=2*numpy.pi*numpy.random.rand(1)[0]
				dX=dR*numpy.cos(theta); dY=dR*numpy.sin(theta)
				X=testDict[testNum][particle]['center'][n+1,0]; Y=testDict[testNum][particle]['center'][n+1,1]
				testDict[testNum][particle]['center'][n+1][0]=X+dX; testDict[testNum][particle]['center'][n+1][1]=Y+dY

		n+=1; counter+=1

		Xi=testDict[testNum][i]['center'][n,0]; Yi=testDict[testNum][i]['center'][n,1]
		Xj=testDict[testNum][j]['center'][n,0]; Yj=testDict[testNum][j]['center'][n,1]
		dist=numpy.sqrt((Xi-Xj)**2 + (Yi-Yj)**2)
		minDist=dist-(Ri+Rj)
		if (dist>Ri+Rj+solventParticleSize):
			return testDict
			break
	return testDict


def nonNAN(x):
	x=numpy.asarray(x)
	x=x[~numpy.isnan(x)]
	return x


def average(x):
	x=numpy.asarray(x)
	x=nonNAN(x)
	if (len(x)==0):
		return numpy.nan
	else:
		return x.mean()

def variance(x):
	x=numpy.asarray(x)
	x=nonNAN(x)
	if (len(x)==0):
		return numpy.nan
	else:
		return average(x**2)-average(x)**2


def readCSVasNumpy(fileName, firstObs=2, delimiter=','):
	rows=0; cols=0; flag=0; i=0; j=0; rowNum=1
	strData=csv.reader(open(fileName,"rb"),delimiter=delimiter)
	for row in strData:
		rows+=1
		if (flag==0):
			for col in row:
				cols+=1
			flag=1

	data=numpy.zeros([rows-(firstObs-1),cols],dtype='float64')
	strData=csv.reader(open(fileName,"rb"),delimiter=delimiter)
	for row in strData:
		if (rowNum >= firstObs):
			for col in row:
				try:
					float(col)
				except ValueError:
					col=numpy.nan
				data[i][j]=float(col)
				j+=1
			i+=1; j=0
		rowNum+=1
	return data


def readODS(fileName, sheetName, firstObs=1):
	doc = ODSReader(fileName)
	table = doc.getSheet(sheetName)
	return table


def readODSasNumpy(fileName, sheetName, firstObs=1):
	doc = ODSReader(fileName)
	table = doc.getSheet(sheetName)

	rows = len(table)-firstObs+1
	cols = len(table[firstObs-1])
	data = numpy.zeros([rows,cols])

	for row in range(rows):
		print row
		for col in range(cols):
			value = table[row+firstObs-1][col]
			try:
				float(value)
			except ValueError:
				value = numpy.nan
			data[row][col] = float(value)
	return data


def percentileRange(x,min=0,max=100):
	x=numpy.asarray(x)
	x=nonNAN(x)
	pMin=numpy.percentile(x,min)
	pMax=numpy.percentile(x,max)
	x1=x[x>=pMin]; x1=x1[x1<=pMax]
	if (len(x1)==0):
		return 0
	else:
		return x1



def distanceRange(center1,periphery1,radius1,center2,periphery2,radius2,theta):
	center1=numpy.asarray(center1); periphery1=numpy.asarray(periphery1)
	center2=numpy.asarray(center2); periphery2=numpy.asarray(periphery2)
	dList=[]; pointsList=[]
	if (radius1 <= radius2):
		C1_C2 = center2-center1
		for P in periphery1:
			C1_P = P-center1
			cos_theta = numpy.dot(C1_C2,C1_P)/(numpy.linalg.norm(C1_C2)*numpy.linalg.norm(C1_P))
			if (cos_theta >= numpy.cos(theta)):
				r1 = numpy.array([P[0]]).astype('int'); c1 = numpy.array([P[1]]).astype('int')
				r2 = periphery2[:,0].astype('int'); c2 = periphery2[:,1].astype('int')
				[dMin, dMax] = myCythonFunc.calculateMinMaxDist(r1,c1,r2,c2)
				dList.append(dMin); pointsList.append(P)
	else:
		C2_C1 = center1-center2
		for P in periphery2:
			C2_P = P-center2
			cos_theta = numpy.dot(C2_C1,C2_P)/(numpy.linalg.norm(C2_C1)*numpy.linalg.norm(C2_P))
			if (cos_theta >= numpy.cos(theta)):
				r1 = numpy.array([P[0]]).astype('int'); c1 = numpy.array([P[1]]).astype('int')
				r2 = periphery1[:,0].astype('int'); c2 = periphery1[:,1].astype('int')
				[dMin, dMax] = myCythonFunc.calculateMinMaxDist(r1,c1,r2,c2)
				dList.append(dMin); pointsList.append(P)
	return dList, pointsList


################################################################################
################################################################################
def gray2rgb(gImg):
	[row,col] = gImg.shape
	rgbImg = numpy.zeros([row, col, 3], dtype='uint8')

	rgbImg[:,:,0] = gImg
	rgbImg[:,:,1] = gImg
	rgbImg[:,:,2] = gImg

	return rgbImg
################################################################################
################################################################################


################################################################################
################################################################################
def threshold_kapur(intensities):
	pDist, bin_edges = numpy.histogram(intensities, bins=range(0,257), normed=False)
	totalEntropy = numpy.zeros(len(pDist), dtype='double')
	minInt = intensities.min()
	maxInt = intensities.max()

	for s in range(minInt, maxInt):
		pLeftDist = pDist[minInt:s+1]; pRightDist = pDist[s+1:maxInt]
		pLeftSum = numpy.sum(pLeftDist); pRightSum = numpy.sum(pRightDist)

		#leftEntropy = -numpy.sum(pLeftDist/pLeftSum * numpy.log(pLeftDist/pLeftSum)); rightEntropy = -numpy.sum(pRightDist/pRightSum * numpy.log(pRightDist/pRightSum))
		#totalEntropy[s] = leftEntropy+rightEntropy
################################################################################
################################################################################


################################################################################
################################################################################
def labelBraggPeaks(bImgDir, gImgDir, labelDataDir, labelImgDirrow, col, numFrames, thetaTh=5, rTickTh=0.1, missFramesTh=50, frameList=[1], structure=[[0,1,0],[1,1,1],[0,1,0]]):
	'''
	INPUT PARAMETERS
	bimgDir      - path of the directory containing the binary image data
	gImgDir      - path of the directory containing the raw image data
	labelDataDir - path of the directory where the labelled data stack will be stored
	labelmgDir   - path of the directory where the labelled images will be stored
	row          - number of rows in the input image
	col          - number of columns in the input image
	numFrames    - number of frames in the input image stack
	thetaTh      - threshold on the rotation of a Bragg spot in some time interval
	rTickTh      - threshold on the resolution value of a Bragg spot in some time interval
	missFramesTh - maximum number of frames a Bragg spot could go missing
	frameList    - list of frames for which labelling is required
	structure    - structuring element to find the connected objects

	OUTPUT PARAMETERS
	maxID        - total number of particles found in the stack
	'''
	braggLabelStack = numpy.zeros([row, col, numFrames], dtype='uint16')
	for frame in frameList:
		bImg = numpy.load(bImgDir+'/'+str(frame)+'.npy')
		gImg = numpy.abs(numpy.load(gImgDir+'/'+str(frame)+'.npy'))

		if (frame == frameList[0]):
			labelImg_0, numLabel_0, dictionary_0 = regionProps(bImg, gImg, structure=structure, theta=True, rTick=True)
			maxID = numLabel_0
			dictionary_0['frame'] = []
			for j in range(len(dictionary_0['id'])):
				dictionary_0['frame'].append(frame)
			braggLabelStack[:,:,frame-1] = labelImg_0
		else:
			labelImg_1, numLabel_1, dictionary_1 = regionProps(bImg, gImg, structure=structure, theta=True, rTick=True)
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
		gImg = numpy.abs(numpy.load(gImgDir+'/'+str(frame)+'.npy'))
		bImgBdry = normalize(boundary(bImg))
		labelImg, numLabel, dictionary = regionProps(bImg, gImg, structure=structure, centroid=True)
		bImg = normalize(bImg)
		for j in range(len(dictionary['id'])):
			bImgLabelN = labelImg == dictionary['id'][j]
			ID = numpy.max(bImgLabelN*braggLabelStack[:,:,frame-1])
			cv2.putText(bImg, str(ID), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=127, thickness=2, bottomLeftOrigin=False)
		finalImage = numpy.column_stack((bImg, normalize(numpy.log10(gImg+1))))
		numpy.save(labelDataDir+'/'+str(frame)+'.npy', braggLabelStack[:,:,frame-1])
		cv2.imwrite(labelImgDir+'/'+str(frame)+'.png', finalImage)
	#numpy.save(labelDataDir+'/braggLabelStack.npy', braggLabelStack)
	del braggLabelStack
	return maxID
################################################################################
################################################################################


##############################################################
##############################################################
def labelParticles(bImgDir, gImgDir, labelDataDir, labelImgDir, row, col, numFrames, centerDispRange=[5,10], perAreaChangeRange=[0.1,0.2], missFramesTh=50, frameList=[1], structure=[[0,1,0],[1,1,1],[0,1,0]], fontScale=1):
	labelStack = numpy.zeros([row,col,numFrames], dtype='uint16')
	for frame in frameList:
		print frame,"/",frameList[-1]
		
		bImg = numpy.load(bImgDir+'/'+str(frame)+'.npy')
		gImg = numpy.load(gImgDir+'/'+str(frame)+'.npy')

		if (frame==frameList[0]):
			labelImg_0, numLabel_0, dictionary_0 = regionProps(bImg, gImg, structure=structure, centroid=True, area=True)
			maxID = numLabel_0
			dictionary_0['frame'] = []
			for i in range(len(dictionary_0['id'])):
				dictionary_0['frame'].append(frame)
			labelStack[:,:,frame-1] = labelImg_0
		else:
			labelImg_1, numLabel_1, dictionary_1 = regionProps(bImg, gImg, structure=structure, centroid=True, area=True)
			if (numLabel_1>0):
				areaMin = min(dictionary_1['area']); areaMax = max(dictionary_1['area'])
			for i in range(len(dictionary_1['id'])):
				flag = 0
				bImg_1_LabelN = labelImg_1==dictionary_1['id'][i]
				center_1 = dictionary_1['centroid'][i]
				area_1 = dictionary_1['area'][i]
				frame_1 = frame
				if (areaMax-areaMin>0):
					factor = 1.0*(area_1-areaMin)/(areaMax-areaMin)
					perAreaChangeTh = perAreaChangeRange[1] - factor*(perAreaChangeRange[1]-perAreaChangeRange[0])
					centerDispTh = centerDispRange[1] - factor*(centerDispRange[1]-centerDispRange[0])
				else:
					perAreaChangeTh = perAreaChangeRange[1]
					centerDispTh = centerDispRange[1]
				for j in range(len(dictionary_0['id'])-1,-1,-1):
					center_0 = dictionary_0['centroid'][j]
					area_0 = dictionary_0['area'][j]
					frame_0 = dictionary_0['frame'][j]
					if ((numpy.sqrt((center_1[0]-center_0[0])**2. + (center_1[1]-center_0[1])**2.) <= centerDispTh) and (1.*area_0/area_1 >= (1-perAreaChangeTh) and 1.*area_0/area_1 <= (1+perAreaChangeTh)) and (frame_1-frame_0 <= missFramesTh)):
						labelStack[:,:,frame-1] += (bImg_1_LabelN*dictionary_0['id'][j])
						dictionary_0['centroid'][j] = center_1
						dictionary_0['area'][j] = area_1
						dictionary_0['frame'][j] = frame
						flag = 1
						break
				if (flag == 0):
					maxID += 1
					labelN_1 = bImg_1_LabelN*maxID
					labelStack[:,:,frame-1] += labelN_1
					dictionary_0['id'].append(maxID)
					dictionary_0['centroid'].append(center_1)
					dictionary_0['area'].append(area_1)
					dictionary_0['frame'].append(frame)

	if (labelStack.max() < 256):
		labelStack = labelStack.astype('uint8')
	elif (labelStack.max()<65536):
		labelStack = labelStack.astype('uint16')

	for frame in frameList:
		bImg = numpy.load(bImgDir+'/'+str(frame)+'.npy')
		gImg = numpy.load(gImgDir+'/'+str(frame)+'.npy')
		bImgBdry = normalize(boundary(bImg))
		labelImg, numLabel, dictionary = regionProps(bImg, gImg, structure=structure, centroid=True)
		bImg = normalize(bImg)
		for j in range(len(dictionary['id'])):
			bImgLabelN = labelImg == dictionary['id'][j]
			ID = numpy.max(bImgLabelN*labelStack[:,:,frame-1])
			cv2.putText(bImg, str(ID), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=2, bottomLeftOrigin=False)
		finalImage = numpy.column_stack((bImg, numpy.maximum(bImgBdry,gImg)))
		numpy.save(labelDataDir+'/'+str(frame)+'.npy', labelStack[:,:,frame-1])
		cv2.imwrite(labelImgDir+'/'+str(frame)+'.png', finalImage)
	#numpy.save(labelDataDir+'/labelStack.npy', labelStack)
	del labelStack
	return maxID
##############################################################
##############################################################


def inverseFFT(fImg, mask=1):
	gImg = numpy.abs(numpy.fft.ifftn(numpy.fft.ifftshift(fImg*mask)))
	return gImg


def rotateTheta(pointsX, pointsY, rotPointX, rotPointY, theta):
	pointsX = numpy.asarray(pointsX).astype('double')
	pointsY = numpy.asarray(pointsY).astype('double')
	theta = numpy.deg2rad(theta)
	
	pointsXOffset = pointsX-rotPointX
	pointsYOffset = pointsY-rotPointY
	
	newPointsX = pointsXOffset*numpy.cos(theta) - pointsYOffset*numpy.sin(theta) + rotPointX
	newPointsY = pointsXOffset*numpy.sin(theta) + pointsYOffset*numpy.cos(theta) + rotPointY
	
	return newPointsX, newPointsY
	
	
def RGBtoBGR(RGBImage):
	BGRImage = RGBImage.copy()
	BGRImage[:,:,0] = RGBImage[:,:,2]
	BGRImage[:,:,2] = RGBImage[:,:,0]
	return BGRImage
	
	
def RGB2Gray(RGBImage, method='average', r=0.21, g=0.72, b=0.07):
	if (method=='lightness'):
		gImg = (numpy.max(RGBImage,axis=2)+numpy.min(RGBImage,axis=2))/2
	elif (method=='average'):
		gImg = numpy.mean(RGBImage,axis=2).astype('uint8')
	elif (method=='luminosity'):
		gImg = (0.21*RGBImage[:,:,0] + 0.72*RGBImage[:,:,1] + 0.07*RGBImage[:,:,2]).astype('uint8')
	elif (method=='custom'):
		gImg = (r*RGBImage[:,:,0] + g*RGBImage[:,:,1] + b*RGBImage[:,:,2]).astype('uint8')
	else:
		print method, 'IS NOT A VALID OPTION'
	return gImg

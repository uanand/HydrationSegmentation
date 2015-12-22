import cv2
import numpy
import os.path, os
import sys
from scipy import ndimage
import skimage.filters
import cPickle as pickle
from skimage.morphology import disk, white_tophat

sys.path.append(os.path.abspath('../myFunctions'))
sys.path.append(os.path.abspath('../myClasses'))
import myPythonFunc
import myCythonFunc

import DM3lib as dm3

##############################################################
##############################################################
#CLASS DEFINITIONS FOR READING IMAGE STACK, PERFORMING BACKGROUND
#SUBTRACTION, AND PREPARING DATASETS FOR SEGMENTATION
##############################################################
##############################################################
class dataIO(object):
	##############################################################
	def __init__(self,inputDir,nProcs):
		self.inputDir = inputDir
		self.nProcs = nProcs
		self.metaData = {}
		self.metaData['inputDir'] = self.inputDir
		
		myPythonFunc.mkdir(self.inputDir+'/output')
		myPythonFunc.mkdir(self.inputDir+'/output/data')
		myPythonFunc.mkdir(self.inputDir+'/output/images')
		myPythonFunc.mkdir(self.inputDir+'/output/data/dataProcessing')
		myPythonFunc.mkdir(self.inputDir+'/output/images/dataProcessing')
	##############################################################
	
	##############################################################
	def countFrames(self):
		self.numFrames = 0
		stack = cv2.VideoCapture(self.inputFile)
		if (stack.isOpened() == True):
			while(1):
				flag, frame = stack.read()
				if (self.numFrames == 0):
					if (len(frame.shape) == 3):
						[self.row, self.col] = frame[:,:,0].shape
					elif (len(frame.shape) == 2):
						[self.row, self.col] = frame.shape
				if (flag == True):
					self.numFrames += 1
				else:
					break
		else:
			print "ERROR: COULD NOT OPEN", self.inputFile
			sys.exit()
	##############################################################
	
	
	##############################################################
	def readAVI(self. inputFile):
		self.inputFile = inputFile
		self.countFrames()
		myPythonFunc.mkdir(self.inputDir+'/output/data/dataProcessing/gImgRawStack')
		myPythonFunc.mkdir(self.inputDir+'/output/images/dataProcessing/gImgRawStack')
		self.frameList=range(1,self.numFrames+1)
		stack = cv2.VideoCapture(self.inputFile)
		if (stack.isOpened() == True):
			for i in range(self.numFrames):
				flag, frame = stack.read()
				if (len(frame.shape) == 3):
					gImgRaw = frame[:,:,0]
				elif (len(frame.shape) == 2):
					gImgRaw = frame
				numpy.save(self.inputDir+'/output/data/dataProcessing/gImgRawStack/'+str(i+1)+'.npy', gImgRaw)
				cv2.imwrite(self.inputDir+'/output/images/dataProcessing/gImgRawStack/'+str(i+1)+'.png', gImgRaw)
				
		self.metaData['row']=self.row
		self.metaData['col']=self.col
		self.metaData['numFrames']=self.numFrames
		self.metaData['frameList']=self.frameList
		self.metaData['inputFile'] = self.inputFile
	##############################################################
	
	
	def readImageSequence(self, frameList=[1], fmt='png', lenFileName=1):
		myPythonFunc.mkdir(self.inputDir+'/output/data/dataProcessing/gImgRawStack')
		myPythonFunc.mkdir(self.inputDir+'/output/images/dataProcessing/gImgRawStack')
		self.frameList = frameList
		self.numFrames = len(frameList)
		for frame in self.frameList:
			
	
	
	##############################################################
	def readImageSequence(self, frameList = [1], ext='.png'):
		print "CREATING THE RAW IMAGE NUMPY STACK"
		myPythonFunc.mkdir(self.inputDir+'/output/data/dataProcessing/gImgRawStack')
		myPythonFunc.mkdir(self.inputDir+'/output/images/dataProcessing/gImgRawStack')
		self.frameList = frameList
		self.numFrames = len(self.frameList)
		for frame in frameList:
			if (ext=='.dm3'):
				dm3f = dm3.DM3(self.inputDir+'/'+fileName)
				gImgRaw = dm3f.imagedata.astype('uint16')
			else:
				gImgRaw = cv2.imread(self.inputDir+'/'+str(frame).zfill(4)+ext, 0)
			if (frame==frameList[0]):
				[self.row, self.col] = gImgRaw.shape
			numpy.save(self.inputDir+'/output/data/dataProcessing/gImgRawStack/'+str(frame)+'.npy', gImgRaw)
			if (ext=='.dm3'):
				cv2.imwrite(self.inputDir+'/output/images/dataProcessing/gImgRawStack/'+str(frame)+'.png', myPythonFunc.normalize(gImgRaw))
			else:
				cv2.imwrite(self.inputDir+'/output/images/dataProcessing/gImgRawStack/'+str(frame)+'.png', gImgRaw)
		self.metaData['row']=self.row
		self.metaData['col']=self.col
		self.metaData['numFrames']=self.numFrames
		self.metaData['frameList']=self.frameList
		pickle.dump(self.metaData, open(self.inputDir+'/metaData', 'wb'))
	##############################################################
	
	
	##############################################################
	def readDM3Sequence(self, frameList = []):
		print "CREATING THE RAW IMAGE NUMPY STACK"
		myPythonFunc.mkdir(self.inputDir+'/output/data/dataProcessing/gImgRawStack')
		myPythonFunc.mkdir(self.inputDir+'/output/images/dataProcessing/gImgRawStack')
		tempFileNameList = os.listdir(self.inputDir); fileNameList=[]
		for fileName in tempFileNameList:
			if ('.dm3' in fileName):
				fileNameList.append(fileName)
				
		self.numFrames = len(fileNameList)
		self.frameList = range(1,self.numFrames+1)
		
		for fileName, i in zip(fileNameList, self.frameList):
			dm3f = dm3.DM3(self.inputDir+'/'+fileName)
			gImgRaw = dm3f.imagedata.astype('uint16')
			if (fileName==fileNameList[0]):
				[self.row, self.col] = gImgRaw.shape
			numpy.save(self.inputDir+'/output/data/dataProcessing/gImgRawStack/'+str(i)+'.npy', gImgRaw)
			cv2.imwrite(self.inputDir+'/output/images/dataProcessing/gImgRawStack/'+str(i)+'.png', myPythonFunc.normalize(gImgRaw))
		self.metaData['row']=self.row
		self.metaData['col']=self.col
		self.metaData['numFrames']=self.numFrames
		self.metaData['frameList']=self.frameList
		pickle.dump(self.metaData, open(self.inputDir+'/metaData', 'wb'))
	##############################################################


	##############################################################
	def createRawStack(self):
		"""
		Creates and stores the raw image stack
		Usage           --- self.createRawStack()
		Input arguments --- NULL
		Returns         --- NULL
		Initializes     --- self.gImgRawStack
		"""
		print "CREATING THE RAW IMAGE NUMPY STACK"
		myPythonFunc.mkdir(self.inputDir+'/output/data/dataProcessing/gImgRawStack')
		myPythonFunc.mkdir(self.inputDir+'/output/images/dataProcessing/gImgRawStack')
		self.frameList=range(1,self.numFrames+1)
		stack = cv2.VideoCapture(self.inputFile)
		if (stack.isOpened() == True):
			for i in range(self.numFrames):
				flag, frame = stack.read()
				if (len(frame.shape) == 3):
					gImgRaw = frame[:,:,0]
				elif (len(frame.shape) == 2):
					gImgRaw = frame
				numpy.save(self.inputDir+'/output/data/dataProcessing/gImgRawStack/'+str(i+1)+'.npy', gImgRaw)
				cv2.imwrite(self.inputDir+'/output/images/dataProcessing/gImgRawStack/'+str(i+1)+'.png', gImgRaw)
	##############################################################


	##############################################################
	def subtractBackground(self, bgSubFlag=False, invertFlag=False, weightFlag=False,\
						   method='none', sigmaBackground=20, alpha=0.8, ringRange=[[0,1E8]],\
						   sigmaTHT=1, radiusTHT=5,\
						   frameList=[1]):
		"""
		Performs 'gauss', 'fourier', 'THT', or 'none' background subtraction. Default background subtraction method is 'none'.
		Usage           ---
			self.subtractBackground(bgSubFlag=False, invertFlag=True, frameList=range(1,self.numFrames+1)) --- inverts the raw image stack for frameList
			self.subtractBackground(bgSubFlag=True, method='none', frameList=range(1,self.numFrames+1)) --- keeps the raw image stack as it is
			self.subtractBackground(bgSubFlag=True, invertFlag=True, weightFlag=True, method='gauss', sigmaBackground=20, alpha=0.8) --- inverted gauss weighted background subtraction
			self.subtractBackground(bgSubFlag=True, invertFlag=True, weightFlag=True, method='gauss', sigmaBackground=20, alpha=0.8) --- inverted gauss weighted background subtraction
			self.subtractBackground(bgSubFlag=True, invertFlag=True, method='fourier', ringRange=[[3,30]]) --- inverted Fourier background subtraction with bandpass filter mask between 3 and 30 according to qArr
		Input arguments ---
			bgSubFlag: set to True or False depending upon whether background subtraction is required. Default: False.
			invertFlag: set to True or False depending upon whether image inversion is required. Default: False.
			weightFlag: set to True or False depending upon whether weighted background subtraction is required. Only applicable to 'gauss' background subtraction method. Default: False.
			method: choose one of 'fourier'/'gauss'/'none'. Choosing 'none' will not perform any background subtraction. Default: 'none'.
			sigmaBackground: sigma for gaussian filter used to extract background. Only applicable to 'gauss' background subtraction method. Default: 20.
			alpha: highest weight of the background to be subtracted. newImg = origImg-alpha*background. Only applicable to 'gauss' background subtraction method. Default: 0.8.
			ringRange: bandpass filter range for 'fourier' background subtraction. eg.- ringRange=[[2,5],[20,50]]. Default: [[0,1E8]]
			frameList: list of frames on which background subtraction is required. Default: range(1,int(1E8))
		Returns         --- NULL
			Creates and saves images after background subtraction.
		Initializes     --- self.gImgStack
		"""
		print "SUBTRACTING BACKGROUND"
		if (method not in ['gauss', 'fourier', 'THT', 'none']):
			print method, 'IS NOT A VALID OPTION FOR BACKGROUND SUBTRACTION. CHANGING TO DEFAULT VALUE \'none\'.'
			method='none'

		myPythonFunc.mkdir(self.inputDir+'/output/data/dataProcessing/bgSubStack')
		myPythonFunc.mkdir(self.inputDir+'/output/images/dataProcessing/bgSubStack')
		if (max(frameList)>self.numFrames):
			self.frameList=range(1,self.numFrames+1)
		else:
			self.frameList=frameList
		ringRange = numpy.asarray(ringRange).astype('int32')
		for frame in self.frameList:
			print frame
			gImg = myPythonFunc.normalize(numpy.load(self.inputDir+'/output/data/dataProcessing/gImgRawStack/'+str(frame)+'.npy'))
			if (invertFlag==True):
				gImg = 255-gImg
			if (bgSubFlag==True):
				if (method=='gauss'):
					background = ndimage.gaussian_filter(gImg, sigma=(self.row+self.col)/(2*sigmaBackground)).astype('float64')
					if (weightFlag == True):
						alpha2 = (background - background.min())/(background.max() - background.min())*alpha
						gImg = gImg - (alpha2*background)
					else:
						gImg = gImg - alpha*background
					gImg = myPythonFunc.normalize(gImg)
				elif (method=='fourier'):
					if (frame==self.frameList[0]):
						[pixInNM, pixInAngstrom] = myPythonFunc.findPixelWidth(mag=self.magnification, numPixels=self.numPixels, microscopeName=self.microscopeName)
						[qArr,qTcks,rTcks,pixelTicks] = myPythonFunc.calculate_qArr(gImg, pixInAngstrom=pixInAngstrom)
					fImg = numpy.fft.fftshift(numpy.fft.fftn(gImg))
					mask = ndimage.gaussian_filter(myCythonFunc.createMask(qArr, ringRange).astype('float64'),sigma=1)
					mask[numpy.unravel_index(qArr.argmin(), qArr.shape)] = 1.0
					gImg = myPythonFunc.normalize(numpy.abs(numpy.fft.ifftn(numpy.fft.ifftshift(fImg*mask))))
				elif (method=='THT'):
					gImgBlur = ndimage.gaussian_filter(gImg, sigma=sigmaTHT)
					gImgTHT=white_tophat(gImgBlur, selem=disk(radiusTHT))
					gImg = myPythonFunc.normalize(gImgTHT)
			cv2.imwrite(self.inputDir+'/output/images/dataProcessing/bgSubStack/'+str(frame)+'.png', gImg)
			numpy.save(self.inputDir+'/output/data/dataProcessing/bgSubStack/'+str(frame)+'.npy', gImg)
	##############################################################


	##############################################################
	def refineData(self, method='none', gaussFilterSize=0, medianFilterSize=0):
		"""
		Refines the background subtracted data. Default refinement method is 'none'.
		Usage           ---
			self.refineData(method='none') --- does nothing to the
			self.subtractBackground(bgSubFlag=True, method='none', frameList=range(1,self.numFrames+1)) --- keeps the raw image stack as it is
			self.subtractBackground(bgSubFlag=True, invertFlag=True, weightFlag=True, method='gauss', sigmaBackground=20, alpha=0.8) --- inverted gauss weighted background subtraction
			self.subtractBackground(bgSubFlag=True, invertFlag=True, weightFlag=True, method='gauss', sigmaBackground=20, alpha=0.8) --- inverted gauss weighted background subtraction
			self.subtractBackground(bgSubFlag=True, invertFlag=True, method='fourier', ringRange=[[3,30]]) --- inverted Fourier background subtraction with bandpass filter mask between 3 and 30 according to qArr
		Input arguments ---
			bgSubFlag: set to True or False depending upon whether background subtraction is required. Default: False.
			invertFlag: set to True or False depending upon whether image inversion is required. Default: False.
			weightFlag: set to True or False depending upon whether weighted background subtraction is required. Only applicable to 'gauss' background subtraction method. Default: False.
			method: choose one of 'fourier'/'gauss'/'none'. Choosing 'none' will not perform any background subtraction. Default: 'none'.
			sigmaBackground: sigma for gaussian filter used to extract background. Only applicable to 'gauss' background subtraction method. Default: 20.
			alpha: highest weight of the background to be subtracted. newImg = origImg-alpha*background. Only applicable to 'gauss' background subtraction method. Default: 0.8.
			ringRange: bandpass filter range for 'fourier' background subtraction. eg.- ringRange=[[2,5],[20,50]]. Default: [[0,1E8]]
			frameList: list of frames on which background subtraction is required. Default: range(1,2)
		Returns         --- NULL
			Creates and saves images after background subtraction.
		Initializes     --- self.gImgStack
		"""
		print 'REFINING IMAGE'
		if (method not in ['gauss','median','medianOFgauss','gaussOFmedian','none']):
			print method, 'IS NOT A VALID OPTION FOR IMAGE REFINEMENT. CHANGING TO DEFAULT OPTION \'none\'.'
			method='none'

		myPythonFunc.mkdir(self.inputDir+'/output/data/dataProcessing/refineStack')
		myPythonFunc.mkdir(self.inputDir+'/output/images/dataProcessing/refineStack')
		myPythonFunc.mkdir(self.inputDir+'/output/data/dataProcessing/refineStack/'+method)
		myPythonFunc.mkdir(self.inputDir+'/output/images/dataProcessing/refineStack/'+method)
		if (method == 'gauss'):
			myPythonFunc.mkdir(self.inputDir+'/output/data/dataProcessing/refineStack/'+method+'/'+str(gaussFilterSize))
			myPythonFunc.mkdir(self.inputDir+'/output/images/dataProcessing/refineStack/'+method+'/'+str(gaussFilterSize))
			for frame in self.frameList:
				str1 = 'method='+method+', gaussFilterSize='+str(gaussFilterSize)+', frame='+str(frame); str2 = '\r'+' '*len(str1)+'\r'
				sys.stdout.write(str1); sys.stdout.flush(); sys.stdout.write(str2)
				gImg = numpy.load(self.inputDir+'/output/data/dataProcessing/bgSubStack/'+str(frame)+'.npy')
				gImgRefine = ndimage.gaussian_filter(gImg, sigma=gaussFilterSize)
				numpy.save(self.inputDir+'/output/data/dataProcessing/refineStack/'+method+'/'+str(gaussFilterSize)+'/'+str(frame)+'.npy', gImgRefine)
				cv2.imwrite(self.inputDir+'/output/images/dataProcessing/refineStack/'+method+'/'+str(gaussFilterSize)+'/'+str(frame)+'.png', gImgRefine)
			sys.stdout.flush()
		elif (method == 'median'):
			myPythonFunc.mkdir(self.inputDir+'/output/data/dataProcessing/refineStack/'+method+'/'+str(medianFilterSize))
			myPythonFunc.mkdir(self.inputDir+'/output/images/dataProcessing/refineStack/'+method+'/'+str(medianFilterSize))
			for frame in self.frameList:
				str1 = 'method='+method+', medianFilterSize='+str(medianFilterSize)+', frame='+str(frame); str2 = '\r'+' '*len(str1)+'\r'
				sys.stdout.write(str1); sys.stdout.flush(); sys.stdout.write(str2)
				gImg = numpy.load(self.inputDir+'/output/data/dataProcessing/bgSubStack/'+str(frame)+'.npy')
				gImgRefine = myCythonFunc.medianFilter2D_C(gImg, size=[medianFilterSize,medianFilterSize])
				numpy.save(self.inputDir+'/output/data/dataProcessing/refineStack/'+method+'/'+str(medianFilterSize)+'/'+str(frame)+'.npy', gImgRefine)
				cv2.imwrite(self.inputDir+'/output/images/dataProcessing/refineStack/'+method+'/'+str(medianFilterSize)+'/'+str(frame)+'.png', gImgRefine)
			sys.stdout.flush()
		elif (method == 'medianOFgauss'):
			myPythonFunc.mkdir(self.inputDir+'/output/data/dataProcessing/refineStack/'+method+'/'+str(gaussFilterSize)+'_'+str(medianFilterSize))
			myPythonFunc.mkdir(self.inputDir+'/output/images/dataProcessing/refineStack/'+method+'/'+str(gaussFilterSize)+'_'+str(medianFilterSize))
			for frame in self.frameList:
				str1 = 'method='+method+', gaussFilterSize='+str(gaussFilterSize)+', medianFilterSize='+str(medianFilterSize)+', frame='+str(frame); str2 = '\r'+' '*len(str1)+'\r'
				sys.stdout.write(str1); sys.stdout.flush(); sys.stdout.write(str2)
				gImg = numpy.load(self.inputDir+'/output/data/dataProcessing/bgSubStack/'+str(frame)+'.npy')
				gImgRefine = ndimage.gaussian_filter(gImg, sigma=gaussFilterSize)
				gImgRefine = myCythonFunc.medianFilter2D_C(gImgRefine, size=[medianFilterSize,medianFilterSize])
				numpy.save(self.inputDir+'/output/data/dataProcessing/refineStack/'+method+'/'+str(gaussFilterSize)+'_'+str(medianFilterSize)+'/'+str(frame)+'.npy', gImgRefine)
				cv2.imwrite(self.inputDir+'/output/images/dataProcessing/refineStack/'+method+'/'+str(gaussFilterSize)+'_'+str(medianFilterSize)+'/'+str(frame)+'.png', gImgRefine)
			sys.stdout.flush()
		elif (method == 'gaussOFmedian'):
			myPythonFunc.mkdir(self.inputDir+'/output/data/dataProcessing/refineStack/'+method+'/'+str(gaussFilterSize)+'_'+str(medianFilterSize))
			myPythonFunc.mkdir(self.inputDir+'/output/images/dataProcessing/refineStack/'+method+'/'+str(gaussFilterSize)+'_'+str(medianFilterSize))
			for frame in self.frameList:
				str1 = 'method='+method+', gaussFilterSize='+str(gaussFilterSize)+', medianFilterSize='+str(medianFilterSize)+', frame='+str(frame); str2 = '\r'+' '*len(str1)+'\r'
				sys.stdout.write(str1); sys.stdout.flush(); sys.stdout.write(str2)
				gImg = numpy.load(self.inputDir+'/output/data/dataProcessing/bgSubStack/'+str(frame)+'.npy')
				gImgRefine = myCythonFunc.medianFilter2D_C(gImg, size=[medianFilterSize,medianFilterSize])
				gImgRefine = ndimage.gaussian_filter(gImgRefine, sigma=gaussFilterSize)
				numpy.save(self.inputDir+'/output/data/dataProcessing/refineStack/'+method+'/'+str(gaussFilterSize)+'_'+str(medianFilterSize)+'/'+str(frame)+'.npy', gImgRefine)
				cv2.imwrite(self.inputDir+'/output/images/dataProcessing/refineStack/'+method+'/'+str(gaussFilterSize)+'_'+str(medianFilterSize)+'/'+str(frame)+'.png', gImgRefine)
			sys.stdout.flush()
		elif (method=='none'):
			myPythonFunc.mkdir(self.inputDir+'/output/data/dataProcessing/refineStack/'+method)
			myPythonFunc.mkdir(self.inputDir+'/output/images/dataProcessing/refineStack/'+method)
			for frame in self.frameList:
				gImg = numpy.load(self.inputDir+'/output/data/dataProcessing/bgSubStack/'+str(frame)+'.npy')
				numpy.save(self.inputDir+'/output/data/dataProcessing/refineStack/'+method+'/'+str(frame)+'.npy', gImg)
				cv2.imwrite(self.inputDir+'/output/images/dataProcessing/refineStack/'+method+'/'+str(frame)+'.png', gImg)
	##############################################################


	##############################################################
	def createDatasets(self, bgSubFlag=False, invertFlag=False, weightFlag=False,\
					   bgSubMethod='none', sigmaBackground=20, alpha=0.8, ringRange=[[0,1E8]],\
					   methodList=['none'], gaussFilterSizeList=[], medianFilterSizeList=[],\
					   sigmaTHT=1, radiusTHT=5,\
					   frameList=[1]):
		"""
		Creates raw image, background subtracted, and refined image stacks based on user input parameters.
		Usage           ---
			self.createDatasets(bgSubFlag=True, invertFlag=False, weightFlag=True,\
								bgSubMethod='gauss', sigmaBackground=20, alpha=0.8, ringRange=[[0,1E8]],\
								methodList=['gauss', 'median', 'medianOFgauss'], gaussFilterSizeList=[4,6], medianFilterSizeList=[5,10],\
								frameList=range(1,101))
		Input arguments ---
			bgSubFlag: set to True or False depending upon whether background subtraction is required. Default: False.
			invertFlag: set to True or False depending upon whether image inversion is required. Default: False.
			weightFlag: set to True or False depending upon whether weighted background subtraction is required. Only applicable to 'gauss' background subtraction method. Default: False.
			bgSubMethod: choose one of 'fourier'/'gauss'/'none'. Choosing 'none' will not perform any background subtraction. Default: 'none'.
			sigmaBackground: sigma for gaussian filter used to extract background. Only applicable to 'gauss' background subtraction method. Default: 20.
			alpha: highest weight of the background to be subtracted. newImg = origImg-alpha*background. Only applicable to 'gauss' background subtraction method. Default: 0.8.
			ringRange: bandpass filter range for 'fourier' background subtraction. eg.- ringRange=[[2,5],[20,50]]. Default: [[0,1E8]]
			methodList: the list of methods using which data is to be refined. Choose any from ['gauss','median','medianOFgauss','gausOFmedian','none']. The input should be in the form of a list. Default: ['none']
			gaussFilterSizeList: sigma list for image refinement using gaussian blurring. Default: []
			medianFilterSizeList: median kernel size list for image refinement using median filtering. Default: []
			frameList: list of frames on which background subtraction is required. Default: range(1,2)
		Returns         --- NULL
			Creates and saves images after background subtraction.
		Initializes     --- self.gImgStack
		"""
		self.createRawStack()
		if (max(frameList)>self.numFrames):
			self.frameList=range(1,self.numFrames+1)
		else:
			self.frameList = frameList
		self.subtractBackground(bgSubFlag=bgSubFlag, invertFlag=invertFlag, weightFlag=weightFlag,\
							    method=bgSubMethod, sigmaBackground=sigmaBackground, alpha=alpha,  ringRange=ringRange,\
							    sigmaTHT=sigmaTHT, radiusTHT=radiusTHT,\
							    frameList=self.frameList)
		for method in methodList:
			if (method=='median'):
				for medianFilterSize in medianFilterSizeList:
					self.refineData(method=method, medianFilterSize=medianFilterSize)
			elif (method=='gauss'):
				for gaussFilterSize in gaussFilterSizeList:
					self.refineData(method=method, gaussFilterSize=gaussFilterSize)
			elif (method=='medianOFgauss'):
				for gaussFilterSize in gaussFilterSizeList:
					for medianFilterSize in medianFilterSizeList:
						self.refineData(method=method, gaussFilterSize=gaussFilterSize, medianFilterSize=medianFilterSize)
			elif (method=='gaussOFmedian'):
				for gaussFilterSize in gaussFilterSizeList:
					for medianFilterSize in medianFilterSizeList:
						self.refineData(method=method, gaussFilterSize=gaussFilterSize, medianFilterSize=medianFilterSize)
			elif (method=='none'):
				self.refineData(method=method)
			else:
				print 'WARNING:', method, 'IS NOT A VALID OPTION. THE VALID OPTIONS ARE - [\'gauss\',\'median\',\'medianOFgauss\',\'gaussOFmedian\',\'none\']. USING DEFAULT OPTION \'none\''
				self.refineData(method='none')
	##############################################################
##############################################################
##############################################################



##############################################################
##############################################################
#
##############################################################
##############################################################
class segmentation(object):
	def __init__(self, inputDir, methodList, gaussFilterSizeList, medianFilterSizeList):
		self.inputDir = inputDir
		self.createDictionary(methodList=methodList,gaussFilterSizeList=gaussFilterSizeList,medianFilterSizeList=medianFilterSizeList)
		#self.gImgRawStack = numpy.load(self.inputDir+'/output/data/dataProcessing/gImgRawStack/gImgRawStack.npy')
		#[self.row, self.col, self.numFrames] = self.gImgRawStack.shape


	def createDictionary(self,methodList=['none'],gaussFilterSizeList=[],medianFilterSizeList=[]):
		myPythonFunc.mkdir(self.inputDir+'/output/data/segmentation')
		myPythonFunc.mkdir(self.inputDir+'/output/images/segmentation')
		self.dict = {}; counter=1
		for method in methodList:
			myPythonFunc.mkdir(self.inputDir+'/output/images/segmentation/'+method)
			if (method=='median'):
				for medianFilterSize in medianFilterSizeList:
					myPythonFunc.mkdir(self.inputDir+'/output/images/segmentation/'+method+'/'+str(medianFilterSize))
					self.dict[counter]={}
					self.dict[counter]['method']='median'
					self.dict[counter]['medianFilterSize']=medianFilterSize
					self.dict[counter]['dataDir']=self.inputDir+'/output/data/dataProcessing/refineStack/'+method+'/'+str(medianFilterSize)
					self.dict[counter]['imgDir']=self.inputDir+'/output/images/segmentation/'+method+'/'+str(medianFilterSize)
					counter+=1
			elif (method=='gauss'):
				for gaussFilterSize in gaussFilterSizeList:
					myPythonFunc.mkdir(self.inputDir+'/output/images/segmentation/'+method+'/'+str(gaussFilterSize))
					self.dict[counter]={}
					self.dict[counter]['method']='gauss'
					self.dict[counter]['gaussFilterSize']=gaussFilterSize
					self.dict[counter]['dataDir']=self.inputDir+'/output/data/dataProcessing/refineStack/'+method+'/'+str(gaussFilterSize)
					self.dict[counter]['imgDir']=self.inputDir+'/output/images/segmentation/'+method+'/'+str(gaussFilterSize)
					counter+=1
			elif (method=='medianOFgauss'):
				for gaussFilterSize in gaussFilterSizeList:
					for medianFilterSize in medianFilterSizeList:
						myPythonFunc.mkdir(self.inputDir+'/output/images/segmentation/'+method+'/'+str(gaussFilterSize)+'_'+str(medianFilterSize))
						self.dict[counter]={}
						self.dict[counter]['method']='medianOFgauss'
						self.dict[counter]['gaussFilterSize']=gaussFilterSize
						self.dict[counter]['medianFilterSize']=medianFilterSize
						self.dict[counter]['dataDir']=self.inputDir+'/output/data/dataProcessing/refineStack/'+method+'/'+str(gaussFilterSize)+'_'+str(medianFilterSize)
						self.dict[counter]['imgDir']=self.inputDir+'/output/images/segmentation/'+method+'/'+str(gaussFilterSize)+'_'+str(medianFilterSize)
						counter+=1
			elif (method=='gaussOFmedian'):
				for gaussFilterSize in gaussFilterSizeList:
					for medianFilterSize in medianFilterSizeList:
						myPythonFunc.mkdir(self.inputDir+'/output/images/segmentation/'+method+'/'+str(gaussFilterSize)+'_'+str(medianFilterSize))
						self.dict[counter]={}
						self.dict[counter]['method']='medianOFgauss'
						self.dict[counter]['gaussFilterSize']=gaussFilterSize
						self.dict[counter]['medianFilterSize']=medianFilterSize
						self.dict[counter]['dataDir']=self.inputDir+'/output/data/dataProcessing/refineStack/'+method+'/'+str(gaussFilterSize)+'_'+str(medianFilterSize)
						self.dict[counter]['imgDir']=self.inputDir+'/output/images/segmentation/'+method+'/'+str(gaussFilterSize)+'_'+str(medianFilterSize)
						counter+=1
			elif (method=='none'):
				myPythonFunc.mkdir(self.inputDir+'/output/images/segmentation/'+method)
				self.dict[counter]={}
				self.dict[counter]['method']='none'
				self.dict[counter]['dataDir']=self.inputDir+'/output/data/dataProcessing/refineStack/'+method
				self.dict[counter]['imgDir']=self.inputDir+'/output/images/segmentation/'+method
				counter+=1
			else:
				print 'ERROR: ',method,' IS NOT A VALID OPTION'


	def imageMask(self, img, flag=False, method='none', mode='global', intensityRange=[0,255], excludeZero=False, strelSize=0):
		if (flag==False or method=='none'):
			gImgMask = img
		else:
			gImgMask = img*myPythonFunc.intensityThreshold(img=img, flag=flag, method=method, mode=mode, intensityRange=intensityRange, excludeZero=excludeZero, strelSize=strelSize)
		return gImgMask


	def intensityThreshold(self, img, flag=False, method='none', mode='global', intensityRange=[0,255], excludeZero=False, strelSize=0):
		bImg = myPythonFunc.intensityThreshold(img=img, flag=flag, method=method, mode=mode, intensityRange=intensityRange, excludeZero=excludeZero, strelSize=strelSize)
		return bImg


	def intensityThreshold(self, gImg, method='none', intensityRange=[0,255], partitions=[1,1], excludeZero=False):
		bImg = numpy.zeros([self.row,self.col], dtype='bool')
		[subRow, subCol] = myPythonFunc.createPartition(gImg, partitions)

		if (method == 'mean'):
			for i in range(partitions[0]):
				for j in range(partitions[1]):
					gImgROI = gImg[subRow[i,0]:subRow[i,1], subCol[j,0]:subCol[j,1]]
					if (excludeZero == True):
						threshold = numpy.mean(gImgROI[gImgROI>0])
					else:
						threshold = gImgROI.mean()
					bImgROI = gImgROI>=threshold
					bImg[subRow[i,0]:subRow[i,1], subCol[j,0]:subCol[j,1]] = bImgROI
		elif (method == 'median'):
			for i in range(partitions[0]):
				for j in range(partitions[1]):
					gImgROI = gImg[subRow[i,0]:subRow[i,1], subCol[j,0]:subCol[j,1]]
					if (excludeZero == True):
						threshold = numpy.median(gImgROI[gImgROI>0])
					else:
						threshold = numpy.median(gImgROI)
					bImgROI = gImgROI>=threshold
					bImg[subRow[i,0]:subRow[i,1], subCol[j,0]:subCol[j,1]] = bImgROI
		elif (method == 'otsu'):
			for i in range(partitions[0]):
				for j in range(partitions[1]):
					gImgROI = gImg[subRow[i,0]:subRow[i,1], subCol[j,0]:subCol[j,1]]
					if (excludeZero == True):
						threshold = skimage.filter.threshold_otsu(gImgROI[gImgROI>0])
					else:
						threshold = skimage.filter.threshold_otsu(gImgROI)
					bImgROI = gImgROI>=threshold
					bImg[subRow[i,0]:subRow[i,1], subCol[j,0]:subCol[j,1]] = bImgROI
		elif (method == 'custom'):
			bImg = numpy.logical_and(gImg>=intensityRange[0],gImg<=intensityRange[1])
		elif (method == 'none'):
			return gImg
		else:
			print 'WARNING:', method, 'IS NOT A VALID OPTION FOR intensityThreshold(). USING METHOD \'none\' INSTEAD'
			return gImg
		return bImg


	def createInputImage(self, gImg, gImgMask, refineImgMask):
		self.gImgInput = numpy.logical_and(gImgMask,refineImgMask)*gImg


	def findContour(self, gImg, exclude=numpy.array([],dtype='uint8'), gradientRange=[1,1E10]):
		allContour = myCythonFunc.gradient(gImg.astype('double'), exclude)
		self.bImgAllContour = numpy.logical_and(allContour>=gradientRange[0], allContour<=gradientRange[1])


	def cleanContour(self, bImg, method='none', minNeighbors=0):
		self.cleanContourMethod = method
		if (method in ['edges','neighbors']):
			self.bImgCleanContour = myCythonFunc.cleanContour(bImg.astype('uint8'), method, minNeighbors).astype('bool')
		elif (method == 'none'):
			self.bImgCleanContour = bImg
		else:
			print 'WARNING:', method, 'IS NOT A VALID OPTION FOR cleanContour(). USING METHOD \'none\' INSTEAD'
			self.bImgCleanContour = bImg


	def fillHoles(self, bImg, flag=True):
		self.bImgFill = ndimage.binary_fill_holes(bImg)


	def removeBoundaryParticles(self, bImg, flag=True):
		if (flag == True):
			self.bImgInner = (myCythonFunc.removeBoundaryParticles(bImg.astype('uint8'))).astype('bool')
		else:
			self.bImgInner = bImg


	def binary_opening(self, bImg, iterations=0, flag=True):
		if (flag == True):
			bImg = ndimage.binary_erosion(bImg, iterations=iterations)
			bImg = ndimage.binary_dilation(bImg, iterations=iterations)
			self.bImgOpen = bImg
		else:
			self.bImgOpen = bImg


	def areaThreshold(self, bImg, areaRange=numpy.array([0,1E10]), flag=True):
		if (flag == True):
			self.bImgBig = myCythonFunc.areaThreshold(bImg.astype('uint8'), areaRange).astype('bool')
		else:
			self.bImgBig = bImg


	def circularThreshold(self, bImg, circularityRange=numpy.array([0.0,1.0]), flag=True):
		if (flag == True):
			self.bImgCircular = myCythonFunc.circularThreshold(bImg.astype('uint8'), circularityRange).astype('bool')
		else:
			self.bImgCircular = bImg


	def boundary(self, bImg):
		self.bImgBdry = myPythonFunc.boundary(bImg)


	def labelParticles(self, bImg, fontScale=1):
		labelImg, numLabel, dictionary = myPythonFunc.regionProps(bImg, centroid=True)
		self.gImgTagged = myPythonFunc.normalize(bImg)
		for j in range(len(dictionary['id'])):
			cv2.putText(self.gImgTagged, str(dictionary['id'][j]), (int(dictionary['centroid'][j][1]),int(dictionary['centroid'][j][0])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=127, thickness=1, bottomLeftOrigin=False)


	def collateResult(self):
		gImgInput = self.gImgInput
		gImgAllContour = myPythonFunc.normalize(self.bImgAllContour)
		gImgCleanContour = myPythonFunc.normalize(self.bImgCleanContour)
		gImgFill = myPythonFunc.normalize(self.bImgFill)
		gImgInner = myPythonFunc.normalize(self.bImgInner)
		gImgOpen = myPythonFunc.normalize(self.bImgOpen)
		gImgBig = myPythonFunc.normalize(self.bImgBig)
		gImgCircular = myPythonFunc.normalize(self.bImgCircular)
		gImgTagged = self.gImgTagged
		gImgBdry = myPythonFunc.normalize(myPythonFunc.boundary(self.bImgCircular))

		self.finalImage = numpy.row_stack((numpy.column_stack((gImgInput,gImgAllContour,gImgCleanContour,gImgFill,gImgInner)),\
									  numpy.column_stack((gImgOpen,gImgBig,gImgCircular,gImgTagged,numpy.maximum(gImgBdry,self.gImgRaw)))))

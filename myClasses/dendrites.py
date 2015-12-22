import os, sys
import cv2
import numpy
from scipy import ndimage
import cPickle as pickle
from skimage.morphology import skeletonize
from itertools import combinations

######################################################################
def mkdir(dirName):
	if (os.path.exists(dirName) == False):
		os.makedirs(dirName)
######################################################################

######################################################################
def binary_opening(bImg, iterations=1):
	bImg = ndimage.binary_erosion(bImg, iterations=iterations)
	bImg = ndimage.binary_dilation(bImg, iterations=iterations)
	return bImg
######################################################################

######################################################################
def binary_closing(bImg, iterations=1):
	bImg = ndimage.binary_dilation(bImg, iterations=iterations)
	bImg = ndimage.binary_erosion(bImg, iterations=iterations)
	return bImg
######################################################################

######################################################################
def boundary(bImg):
	bImgErode = ndimage.binary_erosion(bImg)
	bImgBdry = (bImg - bImgErode).astype('bool')
	return bImgBdry
######################################################################

######################################################################
def normalize(gImg, min=0, max=255):
	if (gImg.max() > gImg.min()):
		gImg = 1.0*(max-min)*(gImg - gImg.min())/(gImg.max() - gImg.min())
		gImg=gImg+min
	elif (gImg.max() > 0):
		gImg[:] = max
	gImg=gImg.astype('uint8')
	return gImg
######################################################################


class dendrites(object):
	######################################################################
	def __init__(self,inputFile,outputDir,minArea=2,structure = [[1,1,1],[1,1,1],[1,1,1]]):
		print "LOADING THE RAW IMAGE AND SKELETONIZING"
		if (minArea<2):
			minArea=2
		
		self.inputFile = inputFile
		self.outputDir = outputDir
		self.minArea = minArea
		self.structure = structure
		
		self.gImgRaw = cv2.imread(self.inputFile,0)
		self.bImgRaw = self.gImgRaw==255
		self.bImg = skeletonize(self.bImgRaw)
		[self.row, self.col] = self.bImg.shape
		
		numpy.save(self.outputDir+'/gImgRaw.npy', self.gImgRaw)
		numpy.save(self.outputDir+'/bImg.npy', self.bImg)
		cv2.imwrite(outputDir+'/bImg.png', normalize(self.bImg.astype('uint8')))
	######################################################################
	
	######################################################################
	def loadData(self):
		print "LOADING DATA"
		self.nodes = numpy.load(self.outputDir+'/nodes.npy')
		self.branches = numpy.load(self.outputDir+'/branches.npy')
		self.leaves = numpy.load(self.outputDir+'/leaves.npy')
		
		self.labelBranches = numpy.load(self.outputDir+'/labelBranches.npy')
		self.labelNodes = numpy.load(self.outputDir+'/labelNodes.npy')
		self.branchDict = pickle.load(open(self.outputDir+'/branchDict.dat', 'rb'))
		self.nodeDict = pickle.load(open(self.outputDir+'/nodeDict.dat', 'rb'))
		self.thetaArr = numpy.load(self.outputDir+'/thetaArr.npy')
		
		self.numBranches = self.labelBranches.max()
		self.numNodes = self.labelNodes.max()
	######################################################################
	
	######################################################################
	def branchAndNodes(self,nodeDilate=0):
		print "CLASSIFYING PIXELS INTO BRANCHES, NODES, AND LEAVES"
		self.nodeDilate = nodeDilate
		self.nodes = numpy.zeros([self.row,self.col], dtype='bool')
		self.leaves = numpy.zeros([self.row,self.col], dtype='bool')
		
		for i in range(1,self.row-1):
			for j in range(1,self.col-1):
				if (self.bImg[i,j] == True):
					neighbors = numpy.sum(self.bImg[i-1:i+2,j-1:j+2])-1
					if (neighbors == 1):
						self.leaves[i,j]=True
					elif (neighbors>2):
						self.nodes[i,j]=True
						
		if (self.nodeDilate>0):
			self.nodes = ndimage.binary_dilation(self.nodes,iterations=self.nodeDilate)
		self.nodes = numpy.logical_and(self.nodes,self.bImg)
		self.leaves[self.nodes==True] = False
		self.branches = numpy.logical_xor(self.nodes, self.bImg)
		
		numpy.save(self.outputDir+'/nodes.npy', self.nodes)
		numpy.save(self.outputDir+'/branches.npy', self.branches)
		numpy.save(self.outputDir+'/leaves.npy', self.leaves)
		
		cv2.imwrite(self.outputDir+'/nodes.png', normalize(self.nodes.astype('uint8')))
		cv2.imwrite(self.outputDir+'/branches.png', normalize(self.branches.astype('uint8')))
		cv2.imwrite(self.outputDir+'/leaves.png', normalize(self.leaves.astype('uint8')))
	######################################################################
	
	######################################################################
	def processBranches(self):
		print 'PROCESSING BRANCHES AND DEFINING A FEW MEASUREMENTS'
		self.labelBranches, self.numBranches = ndimage.label(self.branches, structure=self.structure)
		for branch in range(1,self.numBranches+1):
			pixelList = numpy.where(self.labelBranches==branch)
			area = numpy.shape(pixelList)[1]
			if (area==1):
				self.branches[pixelList]=False
				self.nodes[pixelList]=True
				
		self.labelBranches, self.numBranches = ndimage.label(self.branches, structure=self.structure)
		self.branchDict={}
		for branch in range(1,self.numBranches+1):
			str1 = str(branch)+'/'+str(self.numBranches); str2 = '\r'+' '*len(str1)+'\r'
			sys.stdout.write(str1); sys.stdout.flush(); sys.stdout.write(str2)
			
			self.branchDict[branch]={}
			self.branchDict[branch]['pixelList'] = numpy.where(self.labelBranches==branch)
			self.branchDict[branch]['centroid'] = numpy.average(self.branchDict[branch]['pixelList'], axis=1)
			self.branchDict[branch]['area'] = numpy.shape(self.branchDict[branch]['pixelList'])[1]
			self.branchDict[branch]['visitID'] = False
			self.branchDict[branch]['daughters'] = []
			for r,c in zip(self.branchDict[branch]['pixelList'][0],self.branchDict[branch]['pixelList'][1]):
				if (self.leaves[r,c] == True):
					self.branchDict[branch]['endPoint'] = [r,c]
			if (self.branchDict[branch]['area'] >= self.minArea):
				x = numpy.array(self.branchDict[branch]['pixelList'][1])
				y = self.row-numpy.array(self.branchDict[branch]['pixelList'][0])
				try:
					p = numpy.polyfit(x,y,1)
					angle = numpy.rad2deg(numpy.arctan(p[0]))
					if (angle<0):
						angle+=180
				except:
					angle=90
					
				self.branchDict[branch]['angle'] = angle
			else:
				self.branchDict[branch]['angle'] = numpy.nan
				
			sys.stdout.flush()
		numpy.save(self.outputDir+'/nodes.npy', self.nodes)
		numpy.save(self.outputDir+'/branches.npy', self.branches)
		numpy.save(self.outputDir+'/labelBranches.npy', self.labelBranches)
		
		cv2.imwrite(self.outputDir+'/nodes.png', normalize(self.nodes.astype('uint8')))
		cv2.imwrite(self.outputDir+'/branches.png', normalize(self.branches.astype('uint8')))
		
		pickle.dump(self.branchDict, open(self.outputDir+'/branchDict.dat', 'wb'))
	######################################################################
	
	######################################################################
	def processNodes(self):
		print "PROCESSING NODES"
		self.labelNodes, self.numNodes = ndimage.label(self.nodes, structure=self.structure)
		self.nodeDict={}
		for node in range(1,self.numNodes+1):
			str1 = str(node)+'/'+str(self.numNodes); str2 = '\r'+' '*len(str1)+'\r'
			sys.stdout.write(str1); sys.stdout.flush(); sys.stdout.write(str2)
			
			self.nodeDict[node]={}
			self.nodeDict[node]['pixelList'] = numpy.where(self.labelNodes==node)
			self.nodeDict[node]['centroid'] = numpy.average(self.nodeDict[node]['pixelList'], axis=1)
			self.nodeDict[node]['visitID'] = False
			self.nodeDict[node]['outBranch'] = []
			
			sys.stdout.flush()
		numpy.save(self.outputDir+'/labelNodes.npy', self.labelNodes)
		pickle.dump(self.nodeDict, open(self.outputDir+'/nodeDict.dat', 'wb'))
	######################################################################
	
	######################################################################
	def findRelationship(self,seedPoint=[]):
		print "CLASSIFICATION OF MOTHER AND DAUGHTER BRANCHES"
		
		self.seedPoint = seedPoint
		outFile = open(self.outputDir+'/log.dat', 'wb')
		outFile.write('Mother Daughters\n')
		
		if (self.seedPoint):
			branchID=[self.labelBranches[self.seedPoint[0], self.seedPoint[1]]]
		else:
			branchID=[1]
			
		temp = numpy.zeros([self.row, self.col], dtype='uint8')
		for node in range(1,self.numNodes+1):
			self.nodeDict[node]['visitID'] = False
		for branch in range(1,self.numBranches+1):
			self.branchDict[branch]['visitID'] = False
			
		rBranch, cBranch = self.branchDict[branchID[0]]['pixelList']
		for r,c in zip(rBranch, cBranch):
			if (self.leaves[r,c]==True):
				self.branchDict[branchID[0]]['startPoint'] = [r,c]
			for node in numpy.unique(self.labelNodes[r-1:r+2,c-1:c+2])[1:]:
				newNode = node
				
		print 'SEED BRANCH IS: ', branchID[0]
		bigFlag=True
		levelCounter = 1
		while(bigFlag==True):
			str1 = str(levelCounter); str2 = '\r'+' '*len(str1)+'\r'
			sys.stdout.write(str1); sys.stdout.flush(); sys.stdout.write(str2)
			
			allDList=[]
			bigFlag=False
			for branch in branchID:
				flag=False
				rBranch, cBranch = self.branchDict[branch]['pixelList']
				self.branchDict[branch]['visitID'] = True
				self.branchDict[branch]['level'] = levelCounter
				daughterBranchList=[]
				
				temp[rBranch,cBranch] = True
				
				for r,c in zip(rBranch, cBranch):
					for node in numpy.unique(self.labelNodes[r-1:r+2,c-1:c+2])[1:]:
						if (self.nodeDict[node]['visitID'] == True):
							self.branchDict[branch]['startPoint'] = [r,c]
						if (self.nodeDict[node]['visitID'] == False):
							newNode = node
							self.nodeDict[node]['inBranch'] = branch
							self.branchDict[branch]['endPoint'] = [r,c]
							flag = True
					#if (flag==True):
						#break
				if (flag==True):
					bigFlag=True
					node = newNode
					rNode, cNode = numpy.where(self.labelNodes==node)
					self.nodeDict[node]['visitID']=True
					for r,c in zip(rNode,cNode):
						for dBranch in numpy.unique(self.labelBranches[r-1:r+2,c-1:c+2])[1:]:
							if (self.branchDict[dBranch]['visitID']==False):
								daughterBranchList.append(dBranch)
								allDList.append(dBranch)
					daughterBranchList = numpy.unique(daughterBranchList).tolist()
					self.branchDict[branch]['daughters'] = self.nodeDict[node]['outBranch'] = daughterBranchList
					outFile.write('%d ' %(branch))
					for daughterBranch in daughterBranchList:
						outFile.write('%d ' %(daughterBranch))
					outFile.write('\n')
					for daughterBranch in daughterBranchList:
						self.branchDict[daughterBranch]['mother'] = branch
			branchID = numpy.unique(allDList).tolist()
			levelCounter+=1
			
			sys.stdout.flush()
			
		print "LAST IDENTIFIED NODE WAS ", newNode
		
		pickle.dump(self.branchDict, open(self.outputDir+'/branchDict.dat', 'wb'))
		pickle.dump(self.nodeDict, open(self.outputDir+'/nodeDict.dat', 'wb'))
		outFile.close()
		cv2.imwrite(self.outputDir+'/temp.png', normalize(temp.astype('uint8')))
	######################################################################
	
	######################################################################
	def updateAngles(self):
		for branch in range(1,self.numBranches):
			flag=True
			if (numpy.isfinite(self.branchDict[branch]['angle'])):
				try:
					self.branchDict[branch]['startPoint']
				except:
					self.branchDict[branch]['angle'] = numpy.nan; flag=False
					self.branchDict[branch]['area']=1
					print "Start point for branch ", branch, " is not defined"
				try:
					self.branchDict[branch]['endPoint']
				except:
					self.branchDict[branch]['angle'] = numpy.nan; flag=False
					self.branchDict[branch]['area']=1
					print "End point for branch ", branch, " is not defined"
				if (flag==True):
					x1,y1 = self.branchDict[branch]['startPoint'][1], self.row-self.branchDict[branch]['startPoint'][0]
					x2,y2 = self.branchDict[branch]['endPoint'][1], self.row-self.branchDict[branch]['endPoint'][0]
					self.branchDict[branch]['vector'] = [x2-x1, y2-y1]
					
					dotProduct = numpy.dot([1,0], self.branchDict[branch]['vector'])
					cosine = numpy.cos(numpy.deg2rad(self.branchDict[branch]['angle']))
					if (dotProduct*cosine < 0):
						self.branchDict[branch]['angle']+=180
					elif (dotProduct==0):
						if (y2<y1):
							self.branchDict[branch]['angle']+=180
	######################################################################
	
	######################################################################
	def divisionAngles(self):
		print "CALCULATION OF BIFURCATION ANGLE"
		self.thetaColormap = numpy.zeros([self.row, self.col])
		thetaList=[]
		for branch in self.branchDict.keys():
			str1 = str(branch)+'/'+str(self.numBranches); str2 = '\r'+' '*len(str1)+'\r'
			sys.stdout.write(str1); sys.stdout.flush(); sys.stdout.write(str2)
			
			daughterBranchList = self.branchDict[branch]['daughters']
			if (len(daughterBranchList) <= 3):
				tempTheta=[]
				for daughter1,daughter2 in combinations(daughterBranchList,2):
					if (self.branchDict[daughter1]['area']>=self.minArea and self.branchDict[daughter2]['area']>=self.minArea):
						theta = numpy.abs(self.branchDict[daughter1]['angle']-self.branchDict[daughter2]['angle'])
						if (theta>180):
							theta=360-theta
						if (len(daughterBranchList) == 2):
							thetaList.append(theta)
							self.thetaColormap[self.branchDict[branch]['pixelList']] = theta
						elif (len(daughterBranchList) == 3):
							tempTheta.append(theta)
				if (tempTheta):
					tempTheta.remove(max(tempTheta))
					for theta in tempTheta:
						thetaList.append(theta)
					self.thetaColormap[self.branchDict[branch]['pixelList']] = sum(thetaList)/float(len(thetaList))
					
			sys.stdout.flush()
		self.thetaArr = numpy.array(thetaList)
		numpy.save(self.outputDir+'/thetaArr.npy', self.thetaArr)
	######################################################################

#import sys
#import os
#import numpy
#import matplotlib.pyplot as plt
#import cv2

#sys.path.append(os.path.abspath('../myClasses'))
#from trajectoryAnalysis import trajectoryAnalysis

##sys.path.append(os.path.abspath('../myFunctions'))
##import myCythonFunc
##import myPythonFunc


########################################################
#inputDir = r'E:\Copy\Hydration\Hydration\diffusion\water\Coalescing'
##fileName = inputDir+r'\output\data\segmentation\measures\imgDataNM.dat'
#fileName = inputDir+r'\CoalescingData.txt'
#outputDir = inputDir
##gImg = numpy.load(inputDir+r'\output\data\dataProcessing\gImgRawStack\1.npy')
##imgRow, imgCol = gImg.shape
#imgRow, imgCol = 1024,1024
#pixInNM = 545.62/1024
#skiprows=0
#measureInPix=False
########################################################

#hp = trajectoryAnalysis(fileName, outputDir, pixInNM, imgRow, imgCol, skiprows, measureInPix)

##hp.plotTrajectory(offset=True)
##hp.plotTrajectory()
##hp.plotTrajectoryCMAP(gImg, frame=1, image=True)
##hp.plotTrajectory3d()

##hp.plotXY()
##hp.plotVelocity()
##hp.overlayImages(dataDir1=r'G:\PinningOnGold\june12_3\droplet\output\data\dataProcessing\gImgRawStack', dataDir2=r'G:\PinningOnGold\june12_3\droplet\output\data\segmentation\final', outputDir=r'G:\PinningOnGold\june12_3\droplet\output\data\segmentation\overlay', frameList=range(1,50))

##hp.plotAvgRadiusvsTime()
##hp.plotAvgAreavsTime()
##hp.plotParticleRadiusvsTime()
##hp.plotParticleAreavsTime()
##hp.plotNumberofPoints()

##hp.calculateMSD()
##hp.plotMSD()

#hp.calculateDiffusion(percentile=90)
#hp.plotRadiusHistogram(bins=numpy.arange(0,6.1,0.5))
##hp.plotAverageDiffusionHistogram(bins=numpy.arange(0,0.021,0.005))
##hp.plotDiffusionHistogram(bins=10)
#hp.fitDiffusion()
#hp.plotDiffusionMeanWithError()





import numpy
import matplotlib.pyplot as plt

radius1 = numpy.load(r'E:\Copy\Hydration\Hydration\diffusion\water\Coalescing\radiusList.npy')
radius2 = numpy.load(r'E:\Copy\Hydration\Hydration\diffusion\water\NonCoalescing\radiusList.npy')
bins = numpy.arange(0,6.1,0.5)
center = (bins[:-1] + bins[1:]) / 2
width = bins[1]-bins[0]
hist1, bins = numpy.histogram(radius1, bins=bins)
hist2, bins = numpy.histogram(radius2, bins=bins)

fig = plt.figure(figsize = (3,2))
ax = fig.add_axes([0,0,1,1])
ax.bar(center,hist1,align='center',width=width, color='#C0BFBF')
ax.bar(center,hist2,align='center',width=width, bottom=hist1, color='#606060')
ax.set_xlabel('R (nm)')
ax.set_ylabel('Count')
ax.set_xlim(0,6)
ax.set_ylim(0,25000)
plt.savefig('RadiusStackHist.png',format='png')
plt.savefig('RadiusStackHist.pdf',format='pdf')
plt.show()

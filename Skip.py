import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Skip(object):
    
    def __init__(self, time, distance, ID, TT=40, BW=250):
        self.skipID = ID
        self.time = time[distance>-1]
        self.distance = distance[distance>-1]
        self.numSamples = self.getNumSamples()
        self.groups = 0
        self.rTime = self.time.copy()
        self.rDistance = self.distance.copy() # Running distance array
        self.rDistance2 = self.distance.copy() # to keep track of plots
        self.fig, self.ax = 0, 0    
        
        if (self.numSamples <= 10):
            self.numPeds = 0
        else:
            self.numPeds = 0
            self.dt = self.getDt()
            if (TT == 40 and BW == 250):
                self.Algorithm()
            else:
                self.Algorithm(TT=TT, BW=BW)
                
    def getTimeIn(self):
        if self.time.shape[0] >= 1:
            return (round(self.time[0]/1000, 2))
        return "INVALID SKIP"
    def getSkipID(self):
        return self.skipID
        
    def setNumPeds(self, peds):
        self.numPeds = peds
        
    def getNumPeds(self):
        return self.numPeds
        
    def getNumSamples(self):
        if self.time.shape[0] == self.distance.shape[0]:
            return self.time.shape[0]
        else:
            print "Size error in get Num Samples (time, distance) = ({},{})".format(self.time.shape[0],self.distance.shape[0])
            return 0
        
    def getTime(self):
        return self.time.copy()
        
    def getDistance(self):
        return self.distance.copy()
        
    def getDt(self):
        return (self.time[-1] - self.time[0])
        
    def runAlgorithm(self):
        self.setNumPeds(self.Algorithm())
        
    def Algorithm(self, TT=40, BW=250):
        
        if (not self.validNumSamples()):
            print "Returning zero"
            return 0
            
        self.groups = self.assignInitialGroups(TT)
        self.rDistance = self.firstPassAvg(self.groups.copy())
#        self.setNewAxis(groups, "b")
        self.rDistance2 = self.secondPassAvg(BW)
        self.setNumPeds(self.getPeds())
        
#        self.setOriginalAxis()
#        self.setNewAxis(groups,"r")
#        self.plot()
        
       
    def plotAll(self):
        self.fig, self.ax = plt.subplots()
        self.setOriginalAxis()
        self.setNewAxis("b")
        self.setNewAxis2("r")
        plt.show()
        
    def setOriginalAxis(self):
        self.ax.scatter(self.time, self.distance,c='g')
        
    def setNewAxis(self, color):
        self.ax.scatter(self.rTime, self.rDistance, c=color)
        
        for i, text in enumerate(self.groups):
            self.ax.annotate(text, xy=(self.rTime[i],self.rDistance[i]))
            
    def setNewAxis2(self, color):
        self.ax.scatter(self.rTime, self.rDistance2, c=color)
        
        for i, text in enumerate(self.groups):
            self.ax.annotate(text, xy=(self.rTime[i],self.rDistance2[i]))
    
    def plot(self):
        plt.show()
        
    def assignInitialGroups(self, thresh):
        group = 1
        groups = np.zeros(self.getNumSamples(), dtype=int)
        dist = self.getDistance()
        
        TT = thresh #Tightness Threshold
        i1 = i2 = 0
        
        while(i2 < self.getNumSamples() - 1):
            
            while(not atLastIndex(i2, self.getNumSamples()) and not underThresh(dist, i2, TT) ):
                i2 += 1
            
            if (i1 == i2):
                pass
            elif (i2 > i1):
                groups[i1+1:i2] = 0
                
            i1 = i2
            
            while(not atLastIndex(i2, self.getNumSamples()) and underThresh(dist, i2, TT)):
                i2 += 1
                
            if (i1 == i2):
                pass
            elif (i2 > i1):
                groups[i1:i2+1] = group
                group += 1

            i1 = i2
            
        # Cuts down arrays to valid values only
        self.rTime = self.rTime[groups>0]
        self.rDistance = self.rDistance[groups>0]
        groups = groups[groups>0]
        return groups                
     
    def firstPassAvg(self, groups):

        intervals = 10

        uniqueGroups = np.unique(groups)
        rangeIndex = np.linspace(1, intervals, num=intervals, dtype=int)
        firstPassArray = np.empty(self.rDistance.shape, dtype=float)

        for group in uniqueGroups:
            tDist = self.rDistance[groups==group].copy()

            if tDist.shape[0] <= intervals:
                firstPassArray[groups==group] = np.average(tDist, weights=None)
                continue

            maximum = np.max(tDist)
            minimum = np.min(tDist)
            dInt = (maximum - minimum)/float(intervals)

            intervalFilter = np.empty(tDist.shape, dtype=int)
            intervalFilter[:] = 0

            for i in rangeIndex:
                intervalFilter[tDist >= (minimum +(i-1)*dInt)] = i

            firstPassArray[groups==group] = int(getWeightedAvg(intervalFilter.copy(),tDist.copy()))

        # print firstPassArray
        return firstPassArray
        
    def secondPassAvg(self, thresh):
        
        tDist = np.array(self.rDistance.copy(), dtype=int)
        uniqueAvgs, index = np.unique(tDist, return_index = True)
        index = np.sort(index)
        uniqueAvgs = pd.unique(tDist)

        BW = thresh # Body Width Threshold
        i = 0
        i1=i2=0

        while (i<uniqueAvgs.shape[0] - 1):
            
            if (abs(uniqueAvgs[i+1] - uniqueAvgs[i]) < BW):
                 i1 = index[i]
                 i2 = index[i+1] + tDist[tDist == uniqueAvgs[i+1]].shape[0]
                 tDist[i1:i2] = getWeightedAvg2(tDist.copy(), uniqueAvgs[i],uniqueAvgs[i+1])
                 uniqueAvgs, index = np.unique(tDist, return_index = True)
                 index = np.sort(index)
                 uniqueAvgs = pd.unique(tDist)
            else:
                i+=1

        return tDist
        
    def getPeds(self):
        numLines = pd.unique(self.rDistance2).shape[0]
        
        if numLines == 1:
            return 1
        elif numLines > 1:
            return numLines
        
        print "ERROR IN COUNT FOR SKIP {}".format(self.getSkipID())
        return 0
    
    def validNumSamples(self):
        if self.getNumSamples() == 0:
            print "There are no samples in skip {}".format(self.getSkipID())
            return False
        return True

def atLastIndex(index, numSamples):
    if (index < numSamples - 1):
        return False
    elif (index >= numSamples - 1):
        return True

def underThresh(distance, index, TT):
    deltaD = abs(distance[index+1] - distance[index])
    if (deltaD > TT):
        return False
    elif (deltaD <= TT):
        return True
        
def getWeightedAvg(intervalF, tDist):
    wAvg = 0
    totalNumValues = tDist.shape[0]

    for i in np.unique(intervalF):
        wAvg += (tDist[intervalF == i].shape[0]/float(totalNumValues))*np.average(tDist[intervalF==i], weights=None)

    return wAvg
    
def getWeightedAvg2(distance, val1, val2):
    wAvg = 0
    w1 = distance[distance == val1].shape[0]
    w2 = distance[distance == val2].shape[0]
    denom = (w1 + w2)
    wAvg += (w1*val1 + w2*val2)/float(denom)

    return int(wAvg)

data = pd.read_csv("E:\old\codes\NoNoise_TwoSensors_Ped_University_Up_20161109\dataSensor1.csv")
skip = 5

data = data[data.skip==skip]

s = Skip(data.time.values, data.distance.values, skip)

print s.getNumPeds()

print s.plotAll()

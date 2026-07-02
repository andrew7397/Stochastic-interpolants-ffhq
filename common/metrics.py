import torch

class WindowedAverageMetricTracker() :
    '''
    Class that allows to track floating point metrics during training/validation.

    As training/validation progresses, metric value on the current batch can be
    passed to a WindowedAverageMetricTracker, which stores this value internally.

    This class allows two functionalities: (1) it can return the average metric
    value since tracking began; (2) it can return the running average metric
    value over a recent window of a given size of choice.
    '''
    def __init__(self, windowSize: int) :

        self.__metricSumGlobal = 0.0
        self.__metricSumWindow = 0.0
        self.__nRecordedValues = 0

        self.__maxWindowSize = windowSize
        self.__currWindowSize = 0

        # sliding window implemented as circular array
        self.__windowBuffer = torch.zeros(windowSize, dtype=torch.float)
        self.__windowStartIdx = 0
    
    def __enqueue(self, val) :
        if self.__currWindowSize == self.__maxWindowSize :
            raise RuntimeError('Cannot add more values to metric tracking window. Window is full.')
        
        lastIdx = (self.__windowStartIdx + self.__currWindowSize) % self.__maxWindowSize
        self.__windowBuffer[lastIdx] = val
        self.__currWindowSize += 1

    def __dequeue(self) :
        if self.__currWindowSize == 0 :
            raise RuntimeError('Cannot extract value from tracking window. Window is empty.')
        
        val = self.__windowBuffer[self.__windowStartIdx]
        self.__windowStartIdx = (self.__windowStartIdx + 1) % self.__maxWindowSize
        self.__currWindowSize -= 1

        return val.item()
    
    def __updateMetricWindow(self, currVal) :
        # if on full regime
        if self.__currWindowSize == self.__maxWindowSize :
            # remove oldest value from tracking window
            oldestVal = self.__dequeue()
        else :
            oldestVal = 0.0

        self.__metricSumWindow = self.__metricSumWindow - oldestVal + currVal

        self.__enqueue(currVal)
    
    def reset(self) :
        self.__metricSumGlobal = self.__metricSumWindow = 0.0
        self.__nRecordedValues = self.__currWindowSize = 0
        self.__windowStartIdx = 0

    def update(self, currVal: float) :
        self.__metricSumGlobal += currVal

        self.__updateMetricWindow(currVal)

    def getOverallAverage(self) :
        return self.__metricSumGlobal / self.__nRecordedValues
    
    def getWindowAverage(self) :
        return self.__metricSumWindow / self.__currWindowSize
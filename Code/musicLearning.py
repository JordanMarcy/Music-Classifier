import util, random, math, sys, os
from math import exp, log
from util import Counter
from subprocess import call

composerArray = []
typeOfloss = ''

########################################################
#baselineClassifier
def baseline(trainingExamples, validationExamples):
  counts = util.Counter()
  for train in trainingExamples:
    counts[train[1]] += 1
  mostFreq = counts.argMax()
  numMistakes = 0
  for test in validationExamples:
    predicted_y = mostFreq
    if test[1] != predicted_y:
      numMistakes += 1
  print 'Baseline Error: ', 1.0 * numMistakes/len(validationExamples)
  
########################################################
#getClassificationErrorRateNN
def getClassificationErrorRateNN(trainingExamples, validationExamples):
  numMistakes = 0
  # For each example, make a prediction.
  for test in validationExamples:
    predicted_y = predictNN(trainingExamples, test)
    if test[1] != predicted_y:
      numMistakes += 1  
  print 'NN validationError: ', 1.0 * numMistakes / len(validationExamples)


########################################################
#predictNN

def predictNN(trainingExamples, test): 
    minY = 0
    minDistance = float('inf')
    for train, y in trainingExamples:
        distance = 0
        
        for feature, count in train.items():
            distance += math.fabs(test[0][feature] - count)
        if distance < minDistance: 
            minDistance = distance
            minY = y
    return minY


########################################################
#learnSGD

def learnSGD(trainingExamples, validationExamples, numRounds = 100):
    initStepSize = 1
    regularization = 1
    stepSizeReduction = 1
    weights = util.Counter()
    bestTrainError = 1.0
    bestValidationError = 1.0
    bestWeights = util.Counter()
    random.seed(42)
    for round in range(0, numRounds):
        random.shuffle(trainingExamples)
        numUpdates = 0
        for x, y in trainingExamples:
            numUpdates += 1
            weightsCopy = weights.copy()
            lossGrad = logisticLossGradient(x, y, weightsCopy)
            stepSize = 1.0*initStepSize/(numUpdates**stepSizeReduction)
            regTerm = 1.0*regularization / len(trainingExamples)
            for feature, count in x.items():
                weights[feature] = weightsCopy[feature] - stepSize*(lossGrad[feature] + regTerm * weightsCopy[feature])
        trainLoss = 0
        regulariationPenalty = 0

        for x, y in trainingExamples:
            trainLoss += logisticLoss(x, y, weightsCopy)
        for feature, weight in weightsCopy.items():
            regulariationPenalty += weight**2
        regulariationPenalty *= 0.5*regularization
        objective = trainLoss + regulariationPenalty
        trainError = getClassificationErrorRate(trainingExamples, 'train', 0, weights)
        validationError = getClassificationErrorRate(validationExamples, 'validation', 0, weights)
        print trainError, ' ', validationError
        if validationError < bestValidationError: 
            bestTrainError = trainError
            bestValidationError = validationError
            bestWeights = weights
    print 'SGD trainError: ', bestTrainError
    print 'SGD validationError: ', bestValidationError

    
      #for f, v in sorted(bestWeights.items(), key=lambda x: -x[1]):
# print f + "\t" + str(v)
#######################################################
#multiLearn
def multiLearn(trainingExamples, testingExamples, numRounds = 100):
    initStepSize = 1
    regularization = .1
    stepSizeReduction = 1
    weights = util.Counter()
    bestTrainError = 1.0
    bestValidationError = 1.0
    bestWeights = util.Counter()
    for composer in composerArray:
        weights[composer] = util.Counter()
    random.seed(42)
    for round in range(0, numRounds):
        random.shuffle(trainingExamples)
        for composer in composerArray:
            numUpdates = 0
            for x, y in trainingExamples:
                numUpdates += 1
                composerWeights = weights[composer].copy()
                lossGrad = softMaxLossGradient(x, y, weights, composer)            
                stepSize = 1.0*initStepSize/(numUpdates**stepSizeReduction)
                regTerm = 1.0*regularization / len(trainingExamples)
                for feature, count in x.items():
                    weights[composer][feature] = composerWeights[feature] - stepSize*(lossGrad[feature] + regTerm * composerWeights[feature])
        trainLoss = 0
        regulariationPenalty = 0
  
        for x, y in trainingExamples:
            trainLoss += softMaxLoss(x, y, weights)
        trainLoss /= len(trainingExamples)
        for composer in composerArray:
            for feature, weight in weights[composer].items():
                regulariationPenalty += weight**2
        regulariationPenalty *= 0.5*regularization
        objective = trainLoss + regulariationPenalty
      # print objective
        trainError = getClassificationErrorRate(trainingExamples, 'train', 0, weights)
        validationError = getClassificationErrorRate(testingExamples, 'test', 0, weights)
      #print 'trainError: ', trainError
      # print 'validationError: ', validationError
        if validationError < bestValidationError: 
            bestTrainError = trainError
            bestValidationError = validationError
            bestWeights = weights
    print 'Softmax trainError: ', bestTrainError
    print 'Softmax validationError: ', bestValidationError
                      
#######################################################
def getClassificationErrorRate(featureVectors, displayName=None, verbose=0, weights=None):
    numMistakes = 0
    # For each example, make a prediction.
    for x, y in featureVectors:
      predicted_y = predict(x, weights)
      if y != predicted_y:
        numMistakes += 1  
    return 1.0 * numMistakes / len(featureVectors)

#######################################################
#predict
def predict(x, weights):
    if typeOfLoss == 'multi':
        hypothesis = util.Counter()
        for composer in composerArray:

            hypothesis[composer] = exp(weights[composer]*x)
        hypothesis.normalize()
        prediction = composerArray.index(hypothesis.argMax())+1
        return prediction
                                        
    return 1 if x * weights >= 0 else -1

#######################################################
#hingeLoss
def hingeLoss(x, y, weights):
    prediction = x * weights
    return max(1-prediction*y, 0)
  

########################################################
#hingeLossGradient
def hingeLossGradient(x, y, weights):
    gradient = util.Counter()
    if max(1-x*weights*y, 0) != 0:
        for feature, count in x.items(): gradient[feature] = -count * y
    return gradient

#######################################################
#logisticLoss
def logisticLoss(x, y, weights):
    prediction = x * weights
    return log(1+exp(-prediction*y))


########################################################
#logisticLossGradient
def logisticLossGradient(x, y, weights):
    gradient = util.Counter()
    prediction = x*weights
    for feature, count in x.items():
        gradient[feature] = -(1-1.0/(1+exp(-prediction*y)))*count*y
    return gradient

#######################################################
#softMaxLoss
def softMaxLoss(x, y, weights):
  denom = 0
  for composer in composerArray:
    denom += exp(weights[composer] * x)
  numer = exp(weights[composerArray[y-1]] * x)
  return log(numer/denom)


########################################################
#softMaxLossGradient
def softMaxLossGradient(x, y, weights, composer):
  gradient = util.Counter()
  hypothesis = util.Counter()
  for composer in composerArray:
    hypothesis[composer] = exp(weights[composer]*x)
  hypothesis.normalize()
  probComposer = hypothesis[composer]
  correct = 0
  if y == composerArray.index(composer)+1: correct = 1 
  for feature, count in x.items():
    gradient[feature] = (-count * (correct - probComposer))
  return gradient
  


########################################################
#handleTrainingData
def handleTrainingData(path):
    prangePath = path + 'Prange/'
    rcheckPath = path + 'Rcheck/'
    makePrangeDirectory(path, prangePath)
    makeRcheckDirectory(path, rcheckPath)
    prangeData = []
    rcheckData = []
    data = []
    composerOne = ''
    for file in os.listdir(prangePath):
        if file[0] != '.':
            prangeData.append(readPrange(prangePath + '/' + file))
            
    for file in os.listdir(rcheckPath):
        if file[0] != '.':
            rcheckData.append(readRcheck(rcheckPath + '/' + file))
    i = 0
    for file in os.listdir(prangePath):
        if file[0] != '.': 
            indexOne = file.index('_')
            indexTwo = file.index('.txt')
            composer = file[indexOne+1:indexTwo]
            if composerOne == '': composerOne = composer
            if typeOfLoss == 'multi':
                finalValue = composerArray.index(composer)+1
            else:
                if composer == composerOne: finalValue = 1
                else: finalValue = -1
            prangeVector = prangeData[i]
            rcheckVector = rcheckData[i]
            finalVector = prangeVector + rcheckVector
            for x, count in finalVector.items():
                if typeOfLoss == 'multi': finalVector[x] /= 10000
                else: finalVector[x] /= 10000
            data.append([finalVector, finalValue])
            i += 1
    return data
                  

########################################################
#makePrangeDirectory

def makePrangeDirectory(path, goalPath):
    for file in os.listdir(path):
        if file[0] != '.':
            fileLabel = getLabel(path + '/' + file)
            fileToPrange(path + '/' + file, goalPath + '/' + file + '_' + fileLabel)
    
########################################################
#makeRcheckDirectory

def makeRcheckDirectory(path, goalPath):
    for file in os.listdir(path):
        if file[0] != '.':
            fileLabel = getLabel(path + '/' + file)
            fileToRcheck(path + '/' + file, goalPath + '/' + file + '_' + fileLabel)

########################################################
#getLabel:

def getLabel(path):
    f = open(path)
    line = f.readline()
    if line[0:7] != '!!!COM:': print 'LABELING ERROR'
    label = line[8:]
    i = 0
    for token in label.split(','):
      if i == 0: 
          label = token
          global composerArray
          if composerArray.count(label) == 0:
              composerArray.append(label)
      i = i+1
    return label


########################################################
#fileToPrange:

def fileToPrange(path, goalPath):
    command = './prange ' + path + '> ' + goalPath + '.txt'
    call(command, shell = True)

########################################################
#fileToRcheck

def fileToRcheck(path, goalPath):
    command = './rcheck ' + path + '>' + goalPath + '.txt'
    call(command, shell = True)

########################################################
#featurizePrange:

def featurizePrange(pRange, notes):
    
    featureVector = util.Counter()
    pitchClass = util.Counter()
    octaveRange = util.Counter()
    pRangeCopy = pRange.copy()
    
    featureVector['max'] = notes[len(notes)-1]/10
    featureVector['min'] = notes[0]/10
    featureVector['noteCount'] = pRange.totalCount()
    pRangeCopy.normalize()
    featureVector['mostfreq'] = pRangeCopy.argMax()
    for p, count in pRange.items():
        pitchClass['pitch class (' + str(p % 12) + '):'] += count
       
        octaveRange['octave class (' + str(int(p/12)) + ')'] += count
    pitchClass.normalize()
    octaveRange.normalize()
    featureVector += pitchClass
    featureVector += octaveRange
    return featureVector

"""
Our feature vector contains:
Max: The highest pitch 
Min: The lowest pitch
NoteCount: Total number of pitches
MostFreq: Pitch that occurs most often
OctaveFreq: Combines frequency of pitches in same octave

TODO:
Absolute Pitch frequency
Work with durations of notes (need new data)

""" 
########################################################
#featurizeRcheck
def featurizeRcheck(rCheck, durations, absoluteBeats):

    featureVector = util.Counter()
    featureVector['mostFreqDur'] = rCheck.argMax()
    return featureVector

########################################################
#readPrange:

def readPrange(path):

    pRange = util.Counter()
    f = open(path)
    notes = []
    for line in f:
        if line[0] != '*' and line[0] != '!':
            i = 1
            midi = 0
            for token in line.split('\t'):
                token.strip() #removes white space from token
                if i == 1:
                    midi = int(token)
                    notes.append(midi)
                
                if i == 3:
                    freq = int(token)
                    pRange[midi] = freq
                    
                i += 1
    return featurizePrange(pRange, notes)

#########################################################
#readRcheck
def readRcheck(path):
    rCheck = util.Counter()
    f = open(path)
    durations = []
    absoluteBeats = 0
    for line in f:
        if line[0] != 'a' and line[0] != ':':
            i = 1
            duration = 0
            for token in line.split('\t'):
                token.strip()
                if i == 1: absoluteBeats = float(token)
                if i == 2 and float(token) != 0: 
                    durations.append(float(token))
                    rCheck[float(token)] += 1
                i += 1
    return featurizeRcheck(rCheck, durations, absoluteBeats)                

#########################################################                
if __name__ == '__main__':
    if len(sys.argv) < 4: 
        print 'NOT ENOUGH ARGUMENTS'
    else:
        trainingData = sys.argv[1]
        testingData = sys.argv[2]

        global typeOfLoss
        typeOfLoss = sys.argv[3]
        training = handleTrainingData(trainingData)
        testing = handleTrainingData(testingData)
        print 'number of training: ', len(training)
        print 'number of testing: ', len(testing)
        if typeOfLoss == 'multi': multiLearn(training, testing)
        if typeOfLoss == 'sgd': learnSGD(training, testing)
        getClassificationErrorRateNN(training, testing)
            
        baseline(training, testing)

	#This interprets the first argument as a folder containing all the training data,
        #the second as all the testing data, and the third argument as the type of classification
        #perform/model to use
	

		
		

		



  
    
     


    

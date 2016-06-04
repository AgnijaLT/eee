#!usr/bin/env python

from numpy import *
from numpy.fft import fft, ifft
from numpy import random
from scipy.signal import resample, hamming
import wave
import matplotlib.pyplot as P
import os

#Tyrimams naudojama duomenu baze is
#http://www.expressive-speech.net/

def wavread(fileName):
    f1 = wave.open(fileName, 'rb')
    n = f1.getnframes()
    s = f1.readframes(n)
    f1.close()
    return float64((fromstring(s, dtype = int16))/(2**15+0.0))

#funkcija paskaiciuoja paduodamo signalo realu kepstra
#paprastai tai gali buti naudojama pagrindinio tono nustatymui
def cepstrum(x):
    X = abs(fft(x))+1e-10   #mazytis skaiciukas pridedamas, kad nebutu logaritmo is nulio netycia
    return ifft(log(X))

#funkcija grazina fundamentalu dazni
#laikom, kad signalas gautas diskretizuojant 16 kHz dazniu
#butina, kad signale butu 800 elementu (50 ms), tai yra patikrinama funkcijos pradzioje
#metodas paremtas piko ieskojime kepstre
#jeigu kepstrinio piko dydis mazuliukas, tai grazinam nuline reiksme (0.0)
def getF0(x):
    assert len(x) == 800
    #resamplinkim signala iki 16000 ilgio, tada kepstro rezoliucija bus 2 Hz
    xx = resample(x, 16000)
    c = abs(cepstrum(hamming(16000)*xx)[0:8000]).copy()   #paliekam tik pirma puse. Dabar kepstre 8000 atskaitu atitinka 50 ms
    c[0:700] = 0   #nufiltruojam mazycius periodus
    #print c[c.argmax()], c.std()
    #P.figure()
    #P.subplot(211)
    #P.plot(xx)
    #P.subplot(212)
    #P.plot(c)
    #P.show()
    if c.max() > 8*c.std():
        return 160000.0/c.argmax()
    else:
        return 0

#cia x yra ilgas signalas - dazniausiai tai turetu buti irasas, o
#rezultate gauname vektoriu su F0 reiksmemis persidengianciuose languose
def getF0sequence(x):
    data = []
    i = 0
    while i+800 <= len(x):
        xx = x[i:i+800]
        f0 = getF0(xx)
        if f0 != 0:
            data.append(getF0(xx))
        i = i+200
    return asarray(data)


directoryOfWavFiles = "wav/"

#suformuoja (i-ojo vartotojo failu sarasa su neutraliom emocijom (True) arba pytkciu (false))
## vartotojo numeriai gali buti "03", "08", "10" "11" ... "16". Turi buti string'as, o ne skaicius
def getFileListOfUser(userNumber, emotion):
  listOfAllFiles = os.listdir(directoryOfWavFiles)
  listOfUserFiles = []
  if emotion:
    listOfUserFiles = [l for l in listOfAllFiles if l[0:2] == userNumber and l[-6] == "N"]
  else:
    listOfUserFiles = [l for l in listOfAllFiles if l[0:2] == userNumber and l[-6] == "W"]  #sitoj vietoj pakeisti raide, jei norim nagrineti kita emocija
  return listOfUserFiles

##suformuojamas ilgas signalas sudedant viena salia kito atitinkamo vartotojo failu duomenis
## sitas signalas bus naudojamas statistikai sudaryti
def createLongSignalOfUser(userNumber, emotion):
  outSignal = asarray([])
  fileList = getFileListOfUser(userNumber, emotion)
  for f in fileList:
    outSignal = concatenate((outSignal, wavread(directoryOfWavFiles + f)))
  return outSignal

## fff - fundamentaliuju dazniu vektorius, length - is kokio ilgio kalbos segmentu generuoti histogramas
## speechDuration reiksmes turi buti apytiksliai nuo 1 s iki 30 s
def getDataForHistogram(fff, speechDuration):
  numOfFrames = floor(speechDuration*50.0/20)
  numOfsamples = 1000
  data = zeros(numOfsamples, dtype = float)
  for i in xrange(numOfsamples):
    for j in xrange(int(numOfFrames)):
      data[i] = data[i] + fff[random.randint(0, len(fff))]
  data = data/float(numOfFrames)
  return data

## cia pagrindine funkcija, kuria is esmes ir reiketu kviesti kaitaliojant
# vartotojus
def plotHistograms(user, speechDuration):
  xxxNeutral = createLongSignalOfUser(user, True)
  fff1 = getF0sequence(xxxNeutral)
  dataForHist1 = getDataForHistogram(fff1, speechDuration)

  xxxEmotional = createLongSignalOfUser(user, False)
  fff2 = getF0sequence(xxxEmotional)
  dataForHist2 = getDataForHistogram(fff2, speechDuration)

  print findErrorProbability(dataForHist1, dataForHist2)

  P.figure()
  P.hist(dataForHist1, bins = 50, alpha=0.5)
  P.hist(dataForHist2, bins = 50, alpha=0.5)
  P.title("Pagrindinio tono pasiskirstymai ramiai ir emocingai kalbai, kai signalo ilgis" + str(speechDuration) + " s")
  P.xlabel("Pagrindinio tono reiksme")
  P.ylabel("Tikimybes tankis")
  P.show()

## turint duomenu vektorius x ir y, randama apytiksle klaidos tikimybe, jei laikysim,
# kad teisingas sprendimas tada, kai y(i) > x(i)
def findErrorProbability(x, y):
  decisionPoint = (mean(x) + mean(y))/2.0
  numOfTries = 1000000
  numOfErorrs = 0.0
  for i in xrange(numOfTries):
    elem1 = x[random.randint(0, len(x))]
    if elem1 > decisionPoint:
      numOfErorrs = numOfErorrs + 1
  return numOfErorrs/numOfTries * 100.0

plotHistograms("12", 10)

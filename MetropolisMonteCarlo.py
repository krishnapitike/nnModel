#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python

#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))
##Nearest neighbor model
##needs following files
#1) input file with energy matrix
#2) poscar file

#the following two lines is for better display of
#jupyter notebook ; not required for python scrit
#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))

#the following line is for displaying math symbols in latex style
#from IPython.display import display, Math, Latex
#from pandas import *

#for identifying working directories etc
import os
from timeit import default_timer as timer

#numpy and scipy
import numpy as np
np.set_printoptions(precision=3)
from math import exp
import random
import csv

#Atomic simulation environment for NN-list etc.
#see https://wiki.fysik.dtu.dk/ase/
#to install ASE through anaconda: https://anaconda.org/conda-forge/ase
import ase.io
import ase.neighborlist
from   ase import atom

#variables ; can be put into a sperate input file later
global energyGrid
global searchDist

kB=0.00008617     #eV/K
searchDist=4.7
coordinationNum=6

topdir          = './'
inputPFN        = topdir+'POSCAR_3x3x15.vasp'
system          = ase.io.read(inputPFN,format="vasp")
nBsite          = system.get_number_of_atoms()
annealAfter     = nBsite*(nBsite-1)/2

writeRejected   = 0   #1 for true and 0 for false
writeAccepted   = 1   #1 for true and 0 for false

inputFiles      = ['Binaries.txt','EperB.txt','ExcessOPot.txt','ExcessO.txt','FEbin.txt','FEter.txt']
nSpecies        = 7 # Hf Nb Sc Sn Ta Ti Zr
nPhases         = 8 #0.Min, 1.Pyro, 2.Pyro_Vacancy, 3.Fergu, 4.NdTaO4, 5.Lay_Ortho, 6.Perov, 7.Weberite
#PyroPhaseIndex  = 1

initSkipSteps   = 40000
allSkipSteps    = 10000
collectSteps    = 20001
iCheck          = 500

Tmin            = 300
Tmax            = 3000
Tstep           = 500
pO2             = 1

topdir          = './'
tempdirname     = topdir+'tempDir/'
accepteddirname = tempdirname+'accepted/'
rejecteddirname = tempdirname+'rejected/'
outFormat       = 'cif' #or cif, vasp, etc. for more formats see:

print(os.getcwd())


# In[2]:


#converting chemical symbols to unique integers
def chem2Index(chemSeq):
    speciesList=[]
    for i in system.get_chemical_symbols() :
        if (i=='Hf') : speciesList.append(0)
        if (i=='Nb') : speciesList.append(1)
        if (i=='Sc') : speciesList.append(2)
        if (i=='Sn') : speciesList.append(3)
        if (i=='Ta') : speciesList.append(4)
        if (i=='Ti') : speciesList.append(5)
        if (i=='Zr') : speciesList.append(6)
    return speciesList

#Reading all input files
def readFiles(fileName,nColumns):
    #reading lines from input files
    tempList=[]
    tempArray=np.ones((nColumns,nSpecies,nSpecies))
    file=open(fileName,'r')
    for line in file:
        line=line.split()
        tempList.append(line)
    tempList=np.asfarray(tempList,float)
    #print(tempList[0])

    #recasting lines into symmtric matrices
    for p in range(nColumns):
        for it,val in enumerate(tempList):
            tempArray[p,int(val[-1]),int(val[-2])] = val[p]
            tempArray[p,int(val[-2]),int(val[-1])] = val[p]
    #print(tempArray[7])

    #returning the symmetric matrices
    return(tempArray)

#Safely creating directories
def createDirectory(tempdir):
    if not os.path.exists(tempdir):
        os.mkdir(tempdir)
        print("Directory " , tempdir ,  " Created ")
    else:
        print("Directory " , tempdir ,  " already exists")

def getDeltaO2ChemPot(P,T): #P=pressure in atm, T=temperature in K, C=compound string
    a=-3.860E-07      #polynomial fit for janaf table
    b=-1.880E-03
    c=4.430E-02
    mu0=0
    muT=a*T**2+b*T+c
    muP=kB*T*np.log(P)
    return((muT+muP+mu0)/2)

def getO2ChemPot(P,T): #P=pressure in atm, T=temperature in K, C=compound string
    a=-3.860E-07      #polynomial fit for janaf table
    b=-1.880E-03
    c=4.430E-02
    mu0=-9.07679829   #eV/O2 DFT
    mu0=-5.17*2       #ev/O2 experimental
    muT=a*T**2+b*T+c
    muP=kB*T*np.log(P)
    return((muT+muP+mu0)/2)


def getEnergy(system):
    pEnergies   = np.zeros(nPhases)                                         #total energy of the system
    sop         = np.zeros((nSpecies,nSpecies))                             #short range order parameter
    phases      = np.zeros(nPhases)
    adummy      = np.identity(nPhases-1)*nBsite
    pPercent    = np.append(np.zeros((1,nPhases-1)),adummy,axis=0)
    firstatom   = ase.neighborlist.neighbor_list('i', system, searchDist)   #array of first atom indices 0,1,2,3,4.....nBsite
    secondatom  = ase.neighborlist.neighbor_list('j', system, searchDist)   #array of second atom indices 0,1,2,3,4.....nBsite
    speciesList = chem2Index(system.get_chemical_symbols())                 #array of chemical species in whole numbers 0,1,2,3,4,5,6 -> Hf, Nb, Sc, Sn, Ta, Ti, Zr
    localEnergy = np.zeros(nPhases-1)
    itCount     = 0
    if(nBsite*6 != len(firstatom)):
        return(-1)
    else:
        for i in range(len(firstatom)) :
            itCount = itCount+1
            sop[ speciesList[ firstatom[i] ], speciesList[ secondatom[i] ] ] += 1 ##short range order parameter
            for j in range(1,nPhases):
                localEnergy[j-1] += energyGrid[j,speciesList[firstatom[i]],speciesList[secondatom[i]]] -0.5*ExcessO[j,speciesList[firstatom[i]],speciesList[secondatom[i]]]*O2ChemPot #
            localEnergy = localEnergy/6
            if (itCount == coordinationNum):
                pEnergies[0]  += np.min(localEnergy) #excluding the minimum energy mixed phase
                for j in range(1,nPhases):
                    pEnergies[j] += localEnergy[j-1]
                pPercent[0,np.argmin(localEnergy)] += 1                             ## calculating phase percentage for ground phase
                localEnergy    = np.zeros(nPhases-1)
                itCount = 0
        return(pEnergies/nBsite, pPercent/nBsite, sop*5/(nBsite*coordinationNum))

def randomSwap2(chemSeq):
    idx = range(len(chemSeq))
    while (1) :                                          #This while loops until differenct species are selected
        i1, i2 = random.sample(idx, 2)
        if (chemSeq[i1]!=chemSeq[i2]):
            break                                        #Breaks the while loop when differenct species are selected
    chemSeq[i1], chemSeq[i2] = chemSeq[i2], chemSeq[i1]  #atoms are swapped
    return(chemSeq)                                      #swapped list is returned

def anneal(chemSeq):
    #print("annealing..")
    random.shuffle(chemSeq)                              #All atoms are shuffled
    return(chemSeq)                                      #Shuffled list is returned

def phaseUpdate():
    if( exp( -( apE[1]-apE[0] )*beta ) > random.random() ) :
        acceptedPhaseIndex = 1                                 ##Pyrochlore
    elif( exp( -( apE[2]-apE[0] )*beta ) > random.random() ) :
        acceptedPhaseIndex = 2                                 ##Pyrochlore Vacancies
    elif( exp( -( apE[3]-apE[0] )*beta ) > random.random() ) :
        acceptedPhaseIndex = 3                                 ##Fergusonite
    elif( exp( -( apE[4]-apE[0] )*beta ) > random.random() ) :
        acceptedPhaseIndex = 4                                 ##NdTaO4
    elif( exp( -( apE[5]-apE[0] )*beta ) > random.random() ) :
        acceptedPhaseIndex = 5                                 ##Layered Orthorhombic
    elif( exp( -( apE[6]-apE[0] )*beta ) > random.random() ) :
        acceptedPhaseIndex = 6                                 ##Perovskite
    else:
        acceptedPhaseIndex = 0                                 ##updating the ground phase index here
    return(acceptedPhaseIndex)
# In[ ]:


specIndex  = np.unique(chem2Index(system.get_chemical_symbols()))
#reading and recasting energies

#Ebinaries  = readFiles(inputFiles[0],4)            #Columns = Nd, B1, B2 Nd+(B1+B2)/2
#EperB      = readFiles(inputFiles[1],nPhases)      #Columns = phases
#ExcessOPot = readFiles(inputFiles[2],nPhases)      #Columns = phases
ExcessO     = readFiles(inputFiles[3],nPhases)      #Columns = phases
FEbin       = readFiles(inputFiles[4],nPhases+2)    #Columns = phases, minimum, minimum index
FEter       = readFiles(inputFiles[5],nPhases+2)    #Columns = phases, minimum, minimum index

#deltaH_excessO=[]
#for it,val in enumerate(EperB): deltaH_excessO.append( val - Ebinaries[3] )
#energyGrid = np.array(deltaH_excessO)          # deltaH_excessO renamed to energyGrid for consistency
energyGrid = np.array(FEbin)

for it in [tempdirname,accepteddirname,rejecteddirname]: createDirectory(it)

for T in range( Tmin, Tmax+Tstep, Tstep ):

    #BEGIN BLOCK to setup filenames and write variables
    statsFN    = topdir+"%05d.stats.csv"%T # file writing energies
    statsFile  = open(statsFN,'w')
    statsWrite = csv.writer(statsFile)
    phaseFN    = topdir+"%05d.phase.csv"%T # file writing phase percentage
    phaseFile  = open(phaseFN,'w')
    phaseWrite = csv.writer(phaseFile)
    shortFN    = topdir+"%05d.short.csv"%T # file writing short range order parameter
    shortFile  = open(shortFN,'w')
    shortWrite = csv.writer(shortFile)
    ratioFN    = topdir+"%05d.ratio.csv"%T # file writing acceptance ratio
    ratioFile  = open(ratioFN,'w')
    ratioWrite = csv.writer(ratioFile)
    xvalsFN    = topdir+"%05d.xvals.csv"%T # file writing experimental conditions
    xvalsFile  = open(xvalsFN,'w')
    xvalsWrite = csv.writer(xvalsFile)

    accRate    = np.zeros(nPhases)
    tot        = 0
    #END BLOCK to setup filenames and variables

    #BEGIN BLOCK to setup physical parameters
    O2ChemPot  = getDeltaO2ChemPot(pO2,T)
    beta       = 1/(kB*T)
    totEn      = 0
    #END BLOCK to setup physical parameters

    #BEGIN BLOCK to setup steps
    #block to setup number of steps and thermalization steps for each temperature
    #For the first temperature simulation ; we run a lot of thermaization (skip) steps
    if (T==Tmin):
        skipSteps = initSkipSteps
        maxSteps  = initSkipSteps + collectSteps
    else:
        skipSteps = allSkipSteps
        maxSteps  = allSkipSteps + collectSteps
    #END BLOCK to setup steps

    #BEGIN BLOCK for Monte Carlo steps
    apE, apP, sop      = getEnergy(system) ##first step is always accepted
    acceptedPhaseIndex = phaseUpdate()     ##Checking the phase composition
    ##BEGIN BLOCK TO UPDATE accepted state
    acceptedEnergy         = apE[acceptedPhaseIndex]
    acceptedPhasePercent   = apP[acceptedPhaseIndex]
    acceptedConfig         = system.get_chemical_symbols()
    ##END BLOCK TO UPDATE accepted state
    ##BEGIN BLOCK TO UPDATE stats
    stats    = acceptedEnergy
    phase    = acceptedPhasePercent
    short    = sop
    accRate[acceptedPhaseIndex] += 1  #updating acceptance rate
    #tot     += 1                     #this is not done here for some reason
    ##END BLOCK TO UPDATE stats
    for i in range(maxSteps):
        acceptedConfigCopy  = acceptedConfig[:]               ##making a copy of configuration before trail move
        if (i%2==0):                                          ##atom swap at even steps
            trialConfig     = randomSwap2(acceptedConfigCopy) ##making trial move: swapping two atoms
            system.set_chemical_symbols(trialConfig)          ##setting trial move to system
            tpE, tpP, sop   = getEnergy(system)               ##getting energy of the trial configuration
            if( exp( -( tpE[0]-apE[0] )*nBsite*beta ) > random.random() or tpE[0] < apE[0] ):
                acceptedPhaseIndex = acceptedPhaseIndex       ##not updating the accepted phase index here
                apE                = tpE
                apP                = tpP
                acceptedConfig     = trialConfig
            #else: do nothing
        else:                                                 ##phase update at odd steps
            acceptedPhaseIndex = phaseUpdate()                ##Checking the phase composition
        ##BEGIN BLOCK TO UPDATE accepted state
        accRate[acceptedPhaseIndex] += 1
        acceptedEnergy               = apE[acceptedPhaseIndex]
        acceptedPhasePercent         = apP[acceptedPhaseIndex]
        ##END BLOCK TO UPDATE accepted state
        ##BEGIN BLOCK TO UPDATE and write stats
        if (i>=skipSteps):
            stats += acceptedEnergy
            phase += acceptedPhasePercent
            short += sop
            tot   += 1
            xvals  = [T,pO2,O2ChemPot,tot]
            if (tot%iCheck==0) :
                ##BEGIN BLOCK TO cast short range order parameter in a 1x25 array for easy csv printing
                sopList=[]
                for ix in specIndex:
                    for jx in specIndex:
                        sopList.append(short[ix,jx]/tot)
                shortWrite.writerow( '{:3.4e}'.format(x) for x in sopList )
                ##END BLOCK TO cast short range order parameter in a 1x25 array for easy csv printing
                ##BEGIN BLOCK to write energies
                statsList=[stats/tot,stats**2/tot]
                statsWrite.writerow('{:3.4e}'.format(x) for x in statsList)
                ##END BLOCK to write energies
                ##BEGIN BLOCK to write phases, acceptance rate and xvals
                phaseWrite.writerow('{:3.4e}'.format(x) for x in phase/tot)
                ratioWrite.writerow('{:3.4e}'.format(x) for x in accRate/tot)
                xvalsWrite.writerow('{:3.4e}'.format(x) for x in xvals)
                ##END BLOCK to write phases and acceptance rate
                ##BEGIN BLOCK to flush all files
                shortFile.flush()
                statsFile.flush()
                phaseFile.flush()
                ratioFile.flush()
                xvalsFile.flush()
                ##END BLOCK to flush all files
        ##END BLOCK TO UPDATE and write stats
        if (i==0 or tot%iCheck==0):
            system.write(accepteddirname+"%05d.stats.out"%T+"%010d."%tot+outFormat,format=outFormat)
    shortFile.close()
    statsFile.close()
    phaseFile.close()
    ratioFile.close()
    print(T,i)

#%%
import sys
import os
import time
import copy

# Add the project root directory to the Python path when working with source code, 
# not necessary when package is installed
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path )

print(path)

from WaveSpace.Utils import ImportHelpers
from WaveSpace.Decomposition import EMD as emd
from WaveSpace.PlottingHelpers import Plotting

dataPath  = os.path.join(path, "Examples/ExampleData/Output") 
waveData = ImportHelpers.load_wavedata_object(dataPath + "/SimulatedData")

tempWaveData = copy.deepcopy(waveData)
tempWaveData.DataBuckets["SimulatedData"].set_data(waveData.get_data("SimulatedData")[0:2,:,:,:], "trl_posx_posy_time")

emd.EMD(tempWaveData, 
        siftType = 'multivariate_sift',
        nIMFs=7, 
        dataBucketName="SimulatedData", 
        noiseVar = 0.05, 
        n_noiseChans = 10, 
        ndir=None, 
        stp_crit ='stop', 
        sd=0.075, 
        sd2=0.75, 
        tol=0.075,
        stp_cnt=2)

#plot imfs
TrialOfInterest = 0
SelectedChannel = (1,8)
IMFOfInterest = 3
dataInds = (slice(None), TrialOfInterest, SelectedChannel[0], SelectedChannel[1])
Plotting.plot_imfs(tempWaveData, dataInds, IMFOfInterest)
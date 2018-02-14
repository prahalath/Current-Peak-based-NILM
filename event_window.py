import numpy as np
import blued
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool,cpu_count
import os
from time import time

events = pd.read_csv('event_data.txt',sep = ',',header=0,index_col=0)

#Splitting phase 'A' and Phase 'B'
phase_a_events=events[events['phase']=='A']
phase_a_events.index=[i for i in range(len(phase_a_events))]
phase_b_events=events[events['phase']=='B']
phase_b_events.index=[i for i in range(len(phase_b_events))]
events

def extract(eve):
    path=blued.get_path(eve['file'])
    print(eve['file'])
    #load the file
    data = np.loadtxt(path,delimiter = ',',skiprows=24)
    # Missing data fix
    if data[0,0]>eve['time']-bef_time:
        _data=np.loadtxt(blued.get_path(eve['file']-1),delimiter=',',skiprows=24)
        data=np.vstack((_data,data))
    elif data[-1,0]<eve['time']+aft_time:
        _data=np.loadtxt(blued.get_path(eve['file']+1),delimiter=',',skiprows=24)
        data=np.vstack((data,_data))
    #find the event
    data=data[np.where(np.logical_and(data[:,0]>eve['time']-bef_time,data[:,0]<eve['time']+aft_time ))]  
    #delete the phase in which the event does't occur
    data=np.delete(data,2,1) if eve['phase']=='A' else np.delete(data,1,1)
    #selecting the current values
    data=data[:,1]   
    np.savetxt('final_events/dataset_%d/%d_%s_%f.txt'%(blued.get_dataset_num(eve['file']),eve['device'],eve['phase'],eve['time']),data,delimiter=',')
    
params=[]
data_list=[]
events_t=events.T
for e in events_t.columns:
    params.append(events_t[e])
len(params)

start=time()
print("Found %d processors on the pc"%cpu_count())
pool = Pool(processes=cpu_count())
data_list=pool.map(extract,params)
print("Time taken = %f mins"%(time()-start)/60)


import numpy as np
import os
import pandas as pd
from multiprocessing import Pool,cpu_count

def peak_extract(zx):
    l = []
    path = 'final_events/dataset_'+str(zx)+'/'
    list_of_files = os.listdir(path)
    count=0
    features,labels,phase=[],[],[]
#     print(len(list_of_files))
    for file in list_of_files:
#         try:
            #selecting only the curent values
            data = np.loadtxt(path+file, delimiter = ',')
            #finding the peaks
            ipeaks = np.array([max(data[index:index+200]) for index in range(0,len(data),200)])
#             ipeaks = (ipeaks-np.mean(ipeaks[0:29]))[30:]
            mean = np.mean(ipeaks[:29])
            ipeaks = ipeaks[30:]-mean
            if len(ipeaks)!=(60):
                print(file,len(data)/200)
                continue
            else:
                label=str(file.split('_')[0])
                features.append(ipeaks)
                labels.append(label)
                phase.append(str(file.split('_')[1]))
                count+=1
#         except:
#             print('exception at ' + file)
    print (len(ipeaks))
    features=np.array(features)
    labels=np.array(labels)
    phase=np.array(phase)
    return features,labels,phase
    
    df_list = []
print ("Found %d processors on the pc"%cpu_count())
pool = Pool(processes=cpu_count())
data_list=pool.map(peak_extract,[i for i in range(1,17)])

f=[x[0] for x in data_list]
l=[x[1] for x in data_list]
p=[x[2] for x in data_list]
features=f[0]
labels=l[0]
phase=p[0]
#creating feature np.array
for ff in f:
    features=np.vstack((features,ff))
#creating label np.array
for ll in l:
    labels=np.hstack((labels,ll))
#creating phase np.array
for pp in p:
    phase=np.hstack((phase,pp))
print(features.shape)
print(labels.shape)
print(phase.shape)
# f=np.array(features,dtype=np.float164)
# l=np.array(labels,dtype=np.int16)

f=pd.DataFrame(features)
f['labels']=labels
f['phase']=phase
f.to_csv('final_events/final_peaks_updated_new.txt',sep=',')

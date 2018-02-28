#feature engineering

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import numpy as np


rawdata=pd.read_csv(r"2_mobike_track_2017-06-20.csv")
groupdata=rawdata.groupby(['Orderid'])

#Calculate distance between two points
def Distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r * 1000

#new feature:total time 'Timelen'
timelen=(groupdata['Time'].max()-groupdata['Time'].min()).reset_index()
timelen.rename(columns={'Time': 'Timelen'}, inplace=True)

#new feature: Time range 'Hour'
hours=(groupdata['Time'].mean()/3600).astype(int).reset_index()
hours.rename(columns={'Time':'Hour'},inplace=True)

#new feature: Distance, standard deviation of speed 'Distance', 'STDSpeed'
Alldistance=pd.DataFrame(columns=['Orderid','Distance','STDSpeed'])
Alldistance['Orderid']=timelen['Orderid']
Lng0,Lat0,Lng1,Lat1,time0,time1=0,0,0,0,0,0

i=0
for name,group in groupdata:
    distance=0
    vlist=[]
    for index,row in group.iterrows():
        Lng1,Lat1,time1=row['Lng'],row['Lat'],row['Time']
        if(Lng0!=0 and Lat0!=0 and time0!=0 and time1!=time0):
            distance+=Distance(Lng0,Lat0,Lng1,Lat1)
            vlist.append(Distance(Lng0,Lat0,Lng1,Lat1)/(time1-time0))
        Lng0,Lat0,time0=Lng1,Lat1,time1
    varray=np.array(vlist)
    Alldistance.loc[i,'STDSpeed']=varray.std()
    Alldistance.loc[i,'Distance']=distance
    i+=1

#new feature：average speed，rush hour(7:00-9:00,17:00-19:00), unexpected hour(0:00-5:00) 'ASpeed','Rush','Few'
newdata=pd.merge(timelen,Alldistance)
newdata=pd.merge(newdata,hours)
newdata['ASpeed']=newdata['Distance']/newdata['Timelen']
newdata['Rush']=newdata['Hour'].apply(lambda x:1 if (x>=7 and x <=9) or (x>=17 and x<=19)else 0)#是否上班上学高峰时间
newdata['Few']=newdata['Hour'].apply(lambda x:1 if (x==24 or (x>=0 and x<=5) )else 0)#是否人数稀少时间

newdata = StandardScaler().fit_transform(newdata)

#PCA
pca=PCA(n_components=3)
newdata=pca.fit_transform(newdata)
newdata=pd.DataFrame(data=newdata)

newdata.to_csv('prodata.csv',index=False)
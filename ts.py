# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 17:47:37 2020

@author: crull
"""


import pandas as pd
import seaborn as sns
import numpy
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import statistics
from scipy.stats import t

#import pathlib
#from tensorflow import keras
#from tensorflow.keras import layers

#Constants

ro=1000
densitat=ro
viscositat=0.00102



df=pd.read_csv("N07.csv", sep=";")

df[[ 'Disc', 'Helix', "Turbina"]] = pd.get_dummies(df["rodet"])

df.rename(columns={'Diametre Rodet':'diameter','V (rpm)': 'frequency',"Potencia":"power"},inplace=True)

# df=df[df["frequency"]!=1150]
# df=df[df["frequency"]!=1250]

df["frequency"]=df["frequency"]/60
df["frequency"]=df["frequency"].astype(float)

#df=df[df["Serie"]!="B1.1"]
df=df[df["Serie"]!="A1.1"]
#df=df[df["Serie"]!="B1.2"]


corr=df.corr()

df1=df[["frequency","power"]]


ts=TheilSenRegressor(fit_intercept=True)
ts.fit(X=df[["frequency"]],y=df["power"])
df["ts-estimated"]=ts.predict(df[["frequency"]])


lsq=LinearRegression()
lsq.fit(X=df[["frequency"]],y=df["power"])
df["lsq-estimated"]=lsq.predict(df[["frequency"]])

r_ts=mean_squared_error(df["power"],df["ts-estimated"])
r_lsq=mean_squared_error(df["power"],df["lsq-estimated"])

conf_max=[]
conf_min=[]
frequencydummy=[]


for freq in df["frequency"].unique():
    if freq != 1150/60 and freq != 1250/60:
        serie=df[df["frequency"]==freq]["power"]
        mu=statistics.mean(serie)
        sigma=numpy.std(serie)
        gl=len(serie)
        conf_int = t.interval(0.90,gl, loc=mu,scale=sigma)
        conf_min.append(conf_int[0])
        conf_max.append(conf_int[1])
        frequencydummy.append(freq)

conf=pd.DataFrame({"freq":frequencydummy,"low":conf_min,"high":conf_max})
conf["ts"]=ts.predict(conf[["freq"]])
conf["lsq"]=lsq.predict(conf[["freq"]])
conf.to_csv("powerfreqmodel.csv")

fig, ax = plt.subplots(figsize=[13,8])

plt.plot(df["frequency"],df["lsq-estimated"])
plt.plot(df["frequency"],df["ts-estimated"])
plt.legend(["Least-Squares", "Theil-Sen"])
# plt.scatter(df["frequency"].unique(),conf_max, c="grey")
# plt.scatter(df["frequency"].unique(),conf_min)

sns.lineplot(x=frequencydummy,y=conf_max, color="gray")
ax=sns.lineplot(x=frequencydummy,y=conf_min,color="gray")

plt.scatter(df["frequency"],df["power"], c="black",marker="x", s=12)

ax.set(xlabel='Frequency (Hz)', ylabel='Power (W)')

plt.savefig('powerfreq.png')



#repeat with reynolds

df["Re"]=ro*df["frequency"]*df["diameter"]/viscositat

df["Np"]=(df["power"]/(df["diameter"]**5*df["frequency"]**3*ro))

df["Fr"]=df["frequency"]**2*df["diameter"]/9.81

df["logRe"]=numpy.log(df["Re"])
df["logNp"]=numpy.log(df["Np"])
df["logFr"]=numpy.log(df["Fr"])

ts=TheilSenRegressor(fit_intercept=True)
ts.fit(X=df[["logRe"]],y=df["logNp"])
df["ts-estimatedNp"]=ts.predict(df[["logRe"]])


lsq=LinearRegression()
lsq.fit(X=df[["logRe"]],y=df["logNp"])
df["lsq-estimatedNp"]=lsq.predict(df[["logRe"]])
print(lsq.coef_)
print(lsq.intercept_)
print(r2_score(df["lsq-estimatedNp"], df["logNp"]))

conf_max=[]
conf_min=[]

for re in df["logRe"].unique():
    serie=df[df["logRe"]==re]["logNp"]
    mu=statistics.mean(serie)
    sigma=numpy.std(serie)
    gl=len(serie)
    conf_int = t.interval(0.90,gl, loc=mu,scale=sigma)
    conf_min.append(conf_int[0])
    conf_max.append(conf_int[1])


plt.clf()
plt.plot(df["logRe"],df["lsq-estimatedNp"], c="red")
plt.plot(df["logRe"],df["ts-estimatedNp"])
plt.legend(["Least-Squares", "Theil-Sen"])
plt.scatter(df["logRe"],df["logNp"], c="black",marker="x",s=12)
plt.xlabel("log(Re)")
plt.ylabel("log(Np)")
plt.savefig('adminesional.')

#df['Serie'] = df['Serie'].astype(str).str[0:2]
plt.clf()
fig, ax = plt.subplots(figsize=[13,8])
for series in df["Serie"].unique():
    seriesdf=df[df["Serie"]==series]
    seriesdf["Serie"]=seriesdf["Serie"][0:2]
    plt.scatter(seriesdf["Re"],seriesdf["Np"], label=series, marker="x", s=70)
plt.xlabel("Re")
plt.ylabel("Np")    
ax.legend(loc='upper right')
ax.grid(True)

#multiple regs
lr=LinearRegression()
lsq.fit(X=df[["logRe","logFr"]],y=df["logNp"])
df["lsq-estimatedNp2"]=lsq.predict(df[["logRe","logFr"]])
print(lsq.coef_)
print(lsq.intercept_)
print(r2_score(df["lsq-estimatedNp2"], df["logNp"]))

RMSE1=mean_squared_error(df["logNp"],df["lsq-estimatedNp"])
RMSE2=mean_squared_error(df["logNp"],df["lsq-estimatedNp2"])

#df['Serie'] = df['Serie'].astype(str).str[0:2]
plt.clf()
fig, ax = plt.subplots(figsize=[13,8])
for series in df["Serie"].unique():
    seriesdf=df[df["Serie"]==series]
    seriesdf["Serie"]=seriesdf["Serie"][0:2]
    plt.scatter(seriesdf["frequency"],seriesdf["power"], label=series, marker="x", s=70)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (W)")    
ax.legend(loc='upper left')
ax.grid(True)

plt.show()


def byseries(df,x,y):
    plt.clf()
    fig, ax = plt.subplots(figsize=[13,8])
    for series in df["Serie"].unique():
        seriesdf=df[df["Serie"]==series]
        #seriesdf["Serie"]=seriesdf["Serie"][0:2]
        plt.scatter(seriesdf[x],seriesdf[y], label=series, marker="x", s=70)
    plt.xlabel("log(Re)")
    plt.ylabel("log(Np)")    
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.savefig("adminesionalseries.png")
    return

def byseries2(df,x,y):
    plt.clf()
    fig, ax = plt.subplots(figsize=[13,8])
    for series in df["Serie"].unique():
        seriesdf=df[df["Serie"]==series]
        seriesdf=seriesdf[["Serie","power","frequency"]]
        #seriesdf["Serie"]=seriesdf["Serie"][0:2]
        lsq=LinearRegression()
        lsq.fit(X=seriesdf[["frequency"]],y=seriesdf["power"])
        seriesdf["hello"]=lsq.predict(seriesdf[["frequency"]])
        seriesdf["difference"]=seriesdf["power"]-seriesdf["hello"]
        plt.scatter(y=seriesdf["difference"],x=seriesdf["frequency"], label=series, marker="x", s=70)
    plt.xlabel("Freqüència (Hz)")
    plt.ylabel("Potència real - Potència segons mínims quadrats (W)")    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3)
    ax.grid(True)
    plt.savefig('errors.png')
    return




# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:04:20 2020

@author: crull
"""

#import

import pandas as pd
import seaborn as sns
import numpy
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
import matplotlib.pyplot as plt
import statistics
from scipy.stats import t

#constants
ro=1000
densitat=ro
viscositat=0.00102

#read file
df=pd.read_csv("N07.csv", sep=";")
df.rename(columns={'Diametre Rodet':'diameter','V (rpm)': 'frequency',"Potencia":"power"},inplace=True)

#filter rows
#df=df[df["frequency"]!=1150]
#df=df[df["frequency"]!=1250]
df=df[df["Fluid"]!="Oli"]

df["frequency"]=df["frequency"]/60
df["frequency"]=df["frequency"].astype(float)

#df=df[df["Serie"]!="B1.1"]
df=df[df["Serie"]!="A1.1"]
#df=df[df["Serie"]!="B1.2"]


###Power-Frequency correlation

#Theil-Sen
ts=TheilSenRegressor(fit_intercept=True)
ts.fit(X=df[["frequency"]],y=df["power"])
df["ts-estimated"]=ts.predict(df[["frequency"]])

#Least-Squares
lsq=LinearRegression()
lsq.fit(X=df[["frequency"]],y=df["power"])
df["lsq-estimated"]=lsq.predict(df[["frequency"]])
print('Least Squares: P={}·n +{}, Rsq{}'.format(lsq.coef_, lsq.intercept_, lsq.score(X=df[["frequency"]],y=df["power"])))
print(mean_squared_error(df["power"],df["lsq-estimated"]))

#Get confidence
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



matplotlib.rcParams.update({'font.size': 16})
#Plot it confidence intervals
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=[13,8],dpi=200)
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


###Adimensional model

#Get and set Adimensional Numbers
df["Re"]=ro*df["frequency"]*df["diameter"]/viscositat
df["Np"]=(df["power"]/(df["diameter"]**5*df["frequency"]**3*ro))
df["Fr"]=df["frequency"]**2*df["diameter"]/9.81

df["logRe"]=numpy.log10(df["Re"])
df["logNp"]=numpy.log10(df["Np"])
df["logFr"]=numpy.log10(df["Fr"])

#Theeil-Sen
ts=TheilSenRegressor(fit_intercept=True)
ts.fit(X=df[["logRe"]],y=df["logNp"])
df["ts-estimatedNp"]=ts.predict(df[["logRe"]])

#Least-Squares
lsq=LinearRegression()
lsq.fit(X=df[["logRe"]],y=df["logNp"])
df["lsq-estimatedNp"]=lsq.predict(df[["logRe"]])
print('Least Squares: Np={}·log(Re) +{}, Rsq{}'.format(lsq.coef_, lsq.intercept_, lsq.score(X=df[["logRe"]],y=df["logNp"])))
print(mean_squared_error(df["logNp"],df["lsq-estimatedNp"]))

#Plot and save
plt.clf()
fig, ax = plt.subplots(figsize=[13,8],dpi=200)
plt.plot(df["logRe"],df["lsq-estimatedNp"], c="red")
plt.plot(df["logRe"],df["ts-estimatedNp"])
plt.legend(["Least-Squares", "Theil-Sen"])
plt.scatter(df["logRe"],df["logNp"], c="black",marker="x",s=12)
plt.xlabel("log(Re)")
plt.ylabel("log(Np)")
plt.savefig('adminesional.png')

##With Fr
lr=LinearRegression()
lr.fit(X=df[["logRe","logFr"]],y=df["logNp"])
df["lsq-estimatedNp2"]=lr.predict(df[["logRe","logFr"]])

print('Least Squares: log(Np)={}log(Re)+{}log(Fr)+{}, Rsq{}'.format(lr.coef_[0],lr.coef_[1], lr.intercept_, lr.score(X=df[["logRe","logFr"]],y=df["logNp"])))
print(mean_squared_error(df["logNp"],df["lsq-estimatedNp2"]))
#Other plots

def byseries(df,x,y, name, xlab, ylab):
    df2 = df.copy(deep=True)
    df2['Serie'] = df2['Serie'].astype(str).str[0:4]
    plt.clf()
    fig, ax = plt.subplots(figsize=[13,8], dpi=200)
    for series in df2["Serie"].unique():
        seriesdf=df2[df2["Serie"]==series]
        plt.scatter(seriesdf[x],seriesdf[y], label=series, marker="x", s=70)
    plt.xlabel(xlab)
    plt.ylabel(ylab)  
    ax.legend(ncol=3)
    ax.grid(True)
    plt.savefig(name+".png")
    return
byseries(df,"frequency","power","pow-freq series","Frequency (Hz)","Power (W)" )
byseries(df,"logRe","logNp","adimensional series", "log(Re)", "log(Np)")


def byrodet1(df,x,y, name):
    df2 = df.copy(deep=True)
    df2['Serie'] = df2['Serie'].astype(str).str[0:1]
    plt.clf()
    fig, ax = plt.subplots(figsize=[13,8], dpi=200)
    for series in df2["Serie"].unique():
        seriesdf=df2[df2["Serie"]==series]
        plt.scatter(seriesdf[x],seriesdf[y], label=series, marker="x", s=70)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (W)")  
    ax.legend(ncol=1, labels=df["Serie.1"].unique())
    ax.grid(True)
    plt.savefig(name+".png")
    return

def byrodet2(df,x,y, name):
    df2 = df.copy(deep=True)
    df2['Serie'] = df2['Serie'].astype(str).str[0:1]
    plt.clf()
    fig, ax = plt.subplots(figsize=[13,8], dpi=200)
    for series in df2["Serie"].unique():
        seriesdf=df2[df2["Serie"]==series]
        plt.scatter(seriesdf[x],seriesdf[y], label=series, marker="x", s=70)
    plt.xlabel("log(Re)")
    plt.ylabel("log(Np)")  
    ax.legend(ncol=1, labels=df["Serie.1"].unique())
    ax.grid(True)
    plt.savefig(name+".png")
    return

byrodet1(df,"frequency","power", "pow-freq series-rodet")
byrodet2(df,"logRe","logNp","adimensional series-rodet")

def rodet_fit():
    df['Serie'] = df['Serie'].astype(str).str[0]
    plt.clf()
    fig, ax = plt.subplots(figsize=[13,8],dpi=200)
    for series in df["Serie"].unique():
        seriesdf=df[df["Serie"]==series]
        lsq=LinearRegression()
        lsq.fit(X=seriesdf[["frequency"]],y=seriesdf["power"])
        seriesdf["predP"]=lsq.predict(seriesdf[["frequency"]])
        seriesdf.sort_values(by='frequency', inplace=True)
        plt.scatter(y=seriesdf["power"],x=seriesdf["frequency"], marker="x", s=70)
        plt.plot(seriesdf["frequency"],seriesdf["predP"])
        print('{}: P={}·N +{}, Rsq{}'.format(series, lsq.coef_, lsq.intercept_, lsq.score(X=seriesdf[["frequency"]],y=seriesdf["power"])))
        print(mean_squared_error(seriesdf["power"],seriesdf["predP"]))
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("Powe (W)")    
    ax.legend(loc='upper left', labels=df["Serie.1"].unique())
    ax.grid(True)
    plt.savefig("powrfreqseries_fit.png")
    return
rodet_fit()

def rodet_fit():
    df['Serie'] = df['Serie'].astype(str).str[0]
    plt.clf()
    fig, ax = plt.subplots(figsize=[13,8],dpi=200)
    for series in df["Serie"].unique():
        seriesdf=df[df["Serie"]==series]
        lsq=LinearRegression()
        lsq.fit(X=seriesdf[["logRe"]],y=seriesdf["logNp"])
        seriesdf["predNp"]=lsq.predict(seriesdf[["logRe"]])
        seriesdf.sort_values(by='logRe', inplace=True)
        plt.scatter(y=seriesdf["logNp"],x=seriesdf["logRe"], marker="x", s=70)
        plt.plot(seriesdf["logRe"],seriesdf["predNp"])
        print('{}: Np={}·log(Re) +{}, Rsq{}'.format(series, lsq.coef_, lsq.intercept_, lsq.score(X=seriesdf[["logRe"]],y=seriesdf["logNp"])))
        print(mean_squared_error(seriesdf["logNp"],seriesdf["predNp"]))
    plt.xlabel("log(Re)")
    plt.ylabel("log(Np)")    
    ax.legend(loc='upper right', labels=df["Serie.1"].unique())
    ax.grid(True)
    plt.savefig("adminesionalseries_fit.png")
    return
rodet_fit()

###Plots errors

def error_lineal():
    plt.clf()
    fig, ax = plt.subplots(figsize=[13,8],dpi=200)
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
    ax.legend(loc='upper left', labels=df["Serie.1"].unique())
    ax.grid(True)
    plt.savefig('errors_lineal.png')
    return
error_lineal()

def error_adimensional():
    plt.clf()
    df['Serie'] = df['Serie'].astype(str).str[0]
    lsq=LinearRegression()
    lsq.fit(X=df[["logRe"]],y=df["logNp"])
    fig, ax = plt.subplots(figsize=[13,8],dpi=200)
    for series in df["Serie"].unique():
        seriesdf=df[df["Serie"]==series]
        seriesdf=seriesdf[["Serie","logRe","logNp"]]
        seriesdf["hello"]=lsq.predict(seriesdf[["logRe"]])
        seriesdf["difference"]=seriesdf["logNp"]-seriesdf["hello"]
        plt.scatter(y=seriesdf["difference"],x=seriesdf["logRe"], label=series, marker="x", s=70)
    plt.xlabel("logRe")
    plt.ylabel("log(Np) real - log(Np) segons mínims quadrats")    
    ax.legend(labels=df["Serie.1"].unique())
    ax.grid(True)
    plt.savefig('errorsNpRe.png')
    return

error_adimensional()

def error_adimensional2():
    plt.clf()
    df['Serie'] = df['Serie'].astype(str).str[0]
    lsq=LinearRegression()
    lsq.fit(X=df[["logRe","logFr"]],y=df["logNp"])
    fig, ax = plt.subplots(figsize=[13,8],dpi=200)
    for series in df["Serie"].unique():
        seriesdf=df[df["Serie"]==series]
        seriesdf=seriesdf[["Serie","logRe","logNp","logFr"]]
        seriesdf["hello"]=lsq.predict(seriesdf[["logRe","logFr"]])
        seriesdf["difference"]=seriesdf["logNp"]-seriesdf["hello"]
        plt.scatter(y=seriesdf["difference"],x=lsq.coef_[0]*seriesdf["logRe"]+lsq.coef_[1]*seriesdf["logFr"]+lsq.intercept_, label=series, marker="x", s=70)
    plt.xlabel("αlog(Re)+βlog(Np)+log(k)")
    plt.ylabel("log(Np) real - log(Np) segons mínims quadrats")    
    ax.legend(labels=df["Serie.1"].unique(), loc="upper left", ncol=1)
    ax.grid(True)
    plt.savefig('errorsNpReFr.png')
    return

error_adimensional2()

def comp_errors():
    plt.clf()
    fig, ax = plt.subplots(figsize=[13,8],dpi=200)
    lsq1=LinearRegression()
    lsq1.fit(X=df[["logRe"]],y=df["logNp"])
    lsq2=LinearRegression()
    lsq2.fit(X=df[["logRe","logFr"]],y=df["logNp"])
    df["dif1"]=lsq1.predict(df[["logRe"]])-df["logNp"]
    df["dif2"]=lsq2.predict(df[["logRe", "logFr"]])-df["logNp"]
    plt.scatter(y=df["dif1"],x=df["logRe"], c="#ff7f0e",label="Error amb aproximació")
    plt.scatter(y=df["dif2"],x=df["logRe"], c="#1f77b4", label="Error sense aproximació")
    ax.legend()
    plt.xlabel("log(Re)")
    plt.ylabel("log(Np)") 
    plt.savefig('comp_errpor.png')
    return(df)

comp_errors()


# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np



# fig = plt.figure(figsize=[13,8],dpi=200)
# ax = fig.add_subplot(111, projection='3d')


# xs = df["logRe"]
# ys = df["logNp"]
# zs = df["logFr"]
# ax.scatter(xs, ys, zs, c="b", marker="x", depthshade=True)

# ax.set_xlabel('log(Re)')
# ax.set_ylabel('log(Fr)')
# ax.set_zlabel('log(Np)')

# plt.show()

# xs = df["logRe"]
# ys = df["logNp"]
# zs = df["logFr"]

# fig = plt.figure(figsize=[13,8])
# ax = fig.gca(projection='3d')

# ax.plot_trisurf(xs, ys, zs, linewidth=0.2, antialiased=True)

# for angle in range(0, 360, 3):
#     fig = plt.figure(dpi=100)
#     ax = fig.gca(projection='3d')
#     ax.scatter(xs, ys, zs, c="b", marker="x", depthshade=True)
#     ax.view_init(30, angle)
#     ax.set_xlabel('log(Re)')
#     ax.set_zlabel('log(Fr)')
#     ax.set_ylabel('log(Np)')
#     plt.savefig("animated/"+str(angle)+'comp_errpor.png')
#     plt.pause(.001)

xs = df["logRe"]
ys = df["logNp"]
zs = df["logFr"]   
ys2 = df["lsq-estimatedNp2"]

for angle in range(0, 360, 3):
    fig = plt.figure(figsize=[13,8])
    ax = fig.gca(projection='3d')
    ax.scatter(xs, ys2, zs,c="r", marker="x", depthshade=True)
    ax.view_init(30, angle)
    ax.set_xlabel('log(Re)')
    ax.set_zlabel('log(Fr)')
    ax.set_ylabel('log(Np)')
    plt.savefig("animated2/"+str(angle)+'comp_errpor.png')
    plt.pause(.001)



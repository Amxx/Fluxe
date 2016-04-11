#!/usr/bin/python

from numpy import *
import scipy.integrate as si
import matplotlib.pyplot as mp
import threading as th
import queue as qu
import os
import multiprocessing

import subprocess
proc=subprocess.check_output("grep proc /proc/cpuinfo".split())
n=int(chr(proc[-2]))+1
# n=int(proc[-4])
# n = multiprocessing.cpu_count()
print(n)

def cos_part(x,n):
    return cos(n*x)

def coeff_fourier(f,n,a):
    def cosf(x): return cos_part(x,n)*f((1-cos(x))/2)
    if n==0: return a-1/pi*(si.quad(cosf,0,pi))[0]
    return 2/pi*(si.quad(cosf,0,pi))[0]

def approximation_de_fourier(f,n,a):
    return [coeff_fourier(f,k,a) for k in range(n+1)]

def tourbillons(f,n,x,a):
    l=approximation_de_fourier(f,n,a)
    s=l[0]*(1+cos(x))/(sin(x))
    for i in range(1,n+1):
        s+=sin_part(x,i)*l[i]
    return s

def vitesse_z(f,x,z,a):
    def gamma(t):
        return tourbillons(f,20,t,a)*(x-(1-cos(t))/2)/(1+(x-(1-cos(t))/2)**2+z**2)*sin(t)
    return -1/(2*pi)*si.quad(gamma,0,pi)[0]

def vitesse_x(f,x,z,a):
    def gamma(t):
        return tourbillons(f,20,t,a)*sin(t)*z/(1+(x-(1-cos(t))/2)**2+z**2)
    return 1/(2*pi)*si.quad(gamma,0,pi)[0]

def carte_vitesse(profil,n,p,alpha):
    z=[-0.01+i*0.002 for i in range(p+2)]
    x=[i*1/n for i in range(n+2)]
    vx=[]
    vz=[]
    xx,zz=meshgrid(x,z)
    for k in x:
        for l in z:
            vx+=[vitesse_z(profil,k,l,alpha)]
            vz+=[profil(k)*vitesse_x(profil,k,l,alpha)]
    mp.quiver(xx,zz,vx,vz)

def velocity_x(f,X,Z,a,q):
    def gamma(t,x,z):
        return tourbillons(f,20,t,a)*sin(t)*z/(1+(x-(1-cos(t))/2)**2+z**2)
    for x in X:
        for z in Z:
            r=1/(2*pi)*si.quad(gamma,0,pi,(x,z))[0]
            q.put(r)

def velocity_z(f,X,Z,a,q):
    def gamma(t,x,z):
        return tourbillons(f,20,t,a)*(x-(1-cos(t))/2)/(1+(x-(1-cos(t))/2)**2+z**2)*sin(t)
    for x in X:
        for z in Z:
            r=1/(2*pi)*si.quad(gamma,0,pi,(x,z))[0]
            q.put(r)
def decoupage_grille(n,p):
    z=[i*1/p for i in range(-p//2,p//2)]
    x=[i*1/(p-1) for i in range(p)]
    pas=p//n
    res=[0 for i in range(n)]
    for i in range(n):
        res[i]=[]
        for j in range(pas):
            res[i]+=[x[i*pas+j]]
    return res,z

X,z=decoupage_grille(4,16)

print(X, z)
exit()

def mince_profil_NACA_4(x,serie):
    m=int(serie[0])/100
    p=int(serie[1])/10
    xx=int(serie[2:4])/100
    if x<p:
        return m*2/p**2*(p-x)
    else:
        return m*2/(1-p)**2*(p-x)
"""
def mince_profil_NACA_5(x,serie):
    l=int(serie[0])*0.15
    p=int(serie[1])/20
    q=int(serie[2])
    xx=int(serie[3:5])/100
    if q==0:
        return False
"""
def f(x):
    return mince_profil_NACA_4(x,'4212')

def brut_profil(x):
    if x<0.15:
        return 14.0347*(x**3-0.44646*x**2+0.06314*x)
    else:
        return 0.04626*(1-x)

q=qu.Queue()

t=[0 for i in range(4)]
for i in range(4):
    t[i]=th.Thread(None,velocity_x,None,(f,X[i],z,0,q))
    t[i].start()
for i in range(4):
    print(z)
    t[i].join()

R=[]
while not q.empty():
    R+=[q.get()]

print(R)
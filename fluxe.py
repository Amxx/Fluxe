#!/usr/bin/python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numexpr as ne
import scipy.integrate

import itertools
import multiprocessing
import time

matplotlib.style.use('ggplot')


# class profil:
# 	__init__(self, deriv, theta=0, depth=30):
# 		# self.p  = 
# 		self.dp = deriv
# 		self.sf = serie_fourier(deriv, depth, theta)

# =============================================================================
# Profils
# =============================================================================
def mince_profil_NACA_4(x,serie):
	m  = int(serie[0]  ) / 100
	p  = int(serie[1]  ) / 10
	xx = int(serie[2:4]) / 100
	return 2*m*(p-x)/p**2 if x<p else 2*m*(p-x)/(1-p)**2

# =============================================================================
# Fourrier
# =============================================================================
def coeff_fourier(f,k,a):
	def cosf(x): return np.cos(x*k) * f((1-np.cos(x))/2)
	if k==0: return a - 1 / np.pi * (scipy.integrate.quad(cosf,0,np.pi))[0]
	else:    return     2 / np.pi * (scipy.integrate.quad(cosf,0,np.pi))[0]

def serie_fourier(f,n,a):
	return np.vectorize(coeff_fourier)(f, np.arange(n+1, dtype=np.float64), a)

# =============================================================================
def tourbillons(sf,t):
	ssf      = np.copy(sf)
	ssf[0]  *= (1+np.cos(t)) / np.sin(t)
	ssf[1:] *= np.sin(t*np.arange(1, np.size(ssf), dtype=np.float64))
	return np.sum(ssf)

def velocity_x(sf,x,z):
	def gamma(t): return tourbillons(sf,t) * np.sin(t) * z / max(1e-6, (x-(1-np.cos(t))/2)**2 + z**2)
	return + 1 / (2*np.pi)*scipy.integrate.quad(gamma,0,np.pi)[0]
	# def gamma(dx): return tourbillons(sf, np.arccos(1-2*dx)) * z / max(1e-6, (x-dx)**2 + z**2)
	# return + 1 / (2*np.pi) * scipy.integrate.quad(gamma,0,1)[0]

def velocity_z(sf,x,z):
	def gamma(t,): return tourbillons(sf,t) * np.sin(t) * (x-(1-np.cos(t))/2) / max(1e-6, (x-(1-np.cos(t))/2)**2 + z**2)
	return - 1 / (2*np.pi)*scipy.integrate.quad(gamma,0,np.pi)[0]
	# def gamma(dx): return tourbillons(sf, np.arccos(1-2*dx)) * (x-dx) / max(1e-6, (x-dx)**2 + z**2)
	# return - 1 / (2*np.pi) * scipy.integrate.quad(gamma,0,1)[0]

def velocity(sf,x,z):
	vx = velocity_x(sf,x,z)
	vz = velocity_z(sf,x,z) * np.abs(mince_profil_NACA_4(x, '4212'))
	return [vx, vz]

def decoupage_grille(gsize):
	x = np.linspace(-0.5, +1.5, gsize, endpoint=True,  dtype=np.float64)
	z = np.linspace(-1.0, +1.0, gsize, endpoint=False, dtype=np.float64)
	return np.meshgrid(x, z, sparse=False)

# =============================================================================

if __name__ == '__main__':

	fdepth = 30
	gsize  = 32
	flux   = 1e0
	theta  = 10 * np.pi / 180

	f = lambda x: mince_profil_NACA_4(x, '4212')

	# ---------------------------------------------------------------------------
	start_time = time.time()
	sf = serie_fourier(f, fdepth, theta)
	print("Fourier compute time  --- %3.6s seconds" % (time.time() - start_time))
	# ---------------------------------------------------------------------------

	xv, zv = decoupage_grille(gsize)

	# ---------------------------------------------------------------------------
	start_time = time.time()
	pool  = multiprocessing.Pool()
	res   = pool.starmap(velocity, zip(itertools.repeat(sf), xv.flat, zv.flat))
	print("Dynamics compute time --- %3.6f seconds" % (time.time() - start_time))
	# ---------------------------------------------------------------------------

	npres = np.array(res).reshape(gsize, gsize, 2)
	# norm  = np.linalg.norm(npres, axis=2)

	velx = npres[:,:,0] + flux * np.cos(theta)
	velz = npres[:,:,1] + flux * np.sin(theta)
	norm = np.sqrt(velx**2 + velz**2)

	plt.contourf(xv, zv, norm, 100)

	plt.quiver(xv, zv, velx, velz)

	plt.show()
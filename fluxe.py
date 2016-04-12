#!/usr/bin/python

# import functools
import itertools
import math
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
import numexpr as ne
import numpy as np
import scipy.integrate
import time

matplotlib.style.use('ggplot')

# =============================================================================
# Memoize
# replace with : @functools.lru_cache(maxsize=None) ?
# =============================================================================
class Memoize:
	def __init__(self, f):
		self.func = f
		self.memo = {}
	def __call__(self, *args):
		if not args in self.memo: self.memo[args] = self.func(*args)
		return self.memo[args]

# =============================================================================
# Fourrier
# =============================================================================
class Fourrier:
	def coeff(f,k,a):
		def cosf(x): return np.cos(x*k) * f((1-np.cos(x))/2)
		if k==0: return a - 1 / np.pi * (scipy.integrate.quad(cosf,0,np.pi))[0]
		else:    return     2 / np.pi * (scipy.integrate.quad(cosf,0,np.pi))[0]

	def serie(f,n,a):
		return np.vectorize(Fourrier.coeff)(f, np.arange(n+1, dtype=np.float64), a)

# =============================================================================
# Geometry - Grid
# =============================================================================
class Grid:
	def __init__(self, xmin=+0.0, xmax=+1.0, xstep=32, zmin=-0.5, zmax=+0.5, zstep=64):
		self.xstep = xstep
		self.zstep = zstep
		self.x = np.linspace(xmin, xmax, self.xstep, endpoint=True, dtype=np.float64)
		self.z = np.linspace(zmin, zmax, self.zstep, endpoint=True, dtype=np.float64)
		self.xv, self.zv = np.meshgrid(self.x, self.z, sparse=False)

# =============================================================================
# Profiles
# =============================================================================
class Airfoil:
	def __init__(self, deriv, theta=0, depth=30):
		self.profile = Memoize(self._profile)
		self.deriv   = Memoize(deriv)
		self.theta   = theta
		Airfoil.computeFourrier(self, depth)

	def _profile(self, x):
		return scipy.integrate.quad(self.deriv, 0, x)[0]

	def computeFourrier(self, depth=30):
		self.sf = Fourrier.serie(self.deriv, depth, self.theta)

	def plot(self, axis=plt, resolution=100, color='black', lw=3):
		x = np.linspace(0,1,resolution)
		y = np.vectorize(self.profile)(x)
		axis.plot(x, y, color=color, lw=lw)


# -----------------------------------------------------------------------------

class Naca4(Airfoil):
	def __init__(self, serie, theta=0, depth=30):
		self.m  = int(serie[0]  ) / 100
		self.p  = int(serie[1]  ) / 10
		self.xx = int(serie[2:4]) / 100
		Airfoil.__init__(self, self, theta, depth)

	def __call__(self, x):
		return 2*self.m*(self.p-x)/self.p**2 if x<self.p else 2*self.m*(self.p-x)/(1-self.p)**2

# =============================================================================
# Simulation
# =============================================================================
class Simulation:
	def __init__(self, airfoil, grid):
		pool         = multiprocessing.Pool()
		res          = pool.starmap(Simulation.velocity, zip(itertools.repeat(airfoil), grid.xv.flat, grid.zv.flat))
		self.results = np.array(res).reshape(grid.zstep, grid.xstep, 2)

	def velocity(airfoil,x,z):
		vx = Simulation.velocity_x(airfoil,x,z)
		vz = Simulation.velocity_z(airfoil,x,z) * np.abs(airfoil.deriv(x))
		return [vx, vz]

	def velocity_x(airfoil,x,z):
		def gamma(t): return Simulation.tourbillons(airfoil,t) * np.sin(t) * z / max(1e-6, (x-(1-np.cos(t))/2)**2 + z**2)
		return + 1 / (2*np.pi)*scipy.integrate.quad(gamma,0,np.pi)[0]
		# def gamma(dx): return Simulation.tourbillons(airfoil, np.arccos(1-2*dx)) * z / max(1e-6, (x-dx)**2 + z**2)
		# return + 1 / (2*np.pi) * scipy.integrate.quad(gamma,0,1)[0]

	def velocity_z(airfoil,x,z):
		def gamma(t,): return Simulation.tourbillons(airfoil,t) * np.sin(t) * (x-(1-np.cos(t))/2) / max(1e-6, (x-(1-np.cos(t))/2)**2 + z**2)
		return - 1 / (2*np.pi)*scipy.integrate.quad(gamma,0,np.pi)[0]
		# def gamma(dx): return Simulation.tourbillons(airfoil, np.arccos(1-2*dx)) * (x-dx) / max(1e-6, (x-dx)**2 + z**2)
		# return - 1 / (2*np.pi) * scipy.integrate.quad(gamma,0,1)[0]

	def tourbillons(airfoil,t):
		sf      = np.copy(airfoil.sf)
		sf[0]  *= (1+np.cos(t)) / np.sin(t)
		sf[1:] *= np.sin(t*np.arange(1, np.size(sf), dtype=np.float64))
		return np.sum(sf)

	def plot(self, grid, axis=plt, flux=1, theta=0):
		velx = self.results[:,:,0] + flux * np.cos(theta)
		velz = self.results[:,:,1] + flux * np.sin(theta)
		norm = np.sqrt(velx**2 + velz**2)
		axis.contourf(grid.xv, grid.zv, norm, 100)
		axis.quiver(grid.xv, grid.zv, velx, velz)
		axis.set_aspect('equal')


# =============================================================================




if __name__ == '__main__':

	# ---------------------------------------------------------------------------
	start_time = time.time()

	theta      = math.radians(5)
	grid       = Grid(xstep=32, zstep=16, xmin=-0.5, xmax=+1.5)
	airfoil    = Naca4('4212', theta=theta)
	simulation = Simulation(airfoil, grid)


	print("Simulation compute time --- %3.6s seconds" % (time.time() - start_time))

	# ---------------------------------------------------------------------------


	ax = plt.figure().add_subplot(111)

	airfoil.plot(axis=ax)
	simulation.plot(grid, axis=ax, theta=theta)

	plt.show()
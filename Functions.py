import numpy as np
from numba import float32
from numba.experimental import jitclass

spec = [
	('range', float32[:]),
	('radio', float32),
	('z', float32),
	('chi', float32),
	('Vref', float32),
	('zo', float32),
	('theta', float32),
	('rref', float32),
	('zref', float32),
	('thetaref', float32),
	('V', float32),
	('uref', float32),
	('Ts', float32),
	('radioT', float32),
	('thetaT', float32),
	('r0', float32),
	('theta0', float32),
	('R', float32),
	('R_ref', float32),
	('phi', float32),
	('psi', float32),
	('qr', float32),
	('qz', float32),
	('qtheta', float32),
	('qchi', float32),
	('qv', float32),
	('r', float32),
	('rchi', float32),
]


@jitclass(spec)
class NMPC:
	def __init__(self, radio, theta, z, chi, V, rref,  thetaref, zref, Vref, uref, Ts, phi, psi, qr, qtheta, qz, qchi, qv, r, rchi):
		self.range = np.array([-np.pi/4 - self.uref, np.pi/4 - self.uref], dtype=np.float32)
		self.radio = radio
		self.theta = theta
		self.z = z
		self.chi = chi
		self.V = V
		self.Ts = Ts
		self.rref = rref
		self.thetaref = thetaref
		self.zref = zref
		self.Vref = Vref
		self.uref = uref
		self.phi = phi
		self.psi = psi
		self.qr = qr
		self.qtheta = qtheta
		self.qchi = qchi
		self.qz = qz
		self.qv = qv
		self.r = r
		self.rchi = rchi

	def query(self, u):
		#V = self.V
		#Ts = self.Ts
		#rref = self.rref
		#uref = self.uref
		#thetaref = self.thetaref
		#r0 = self.radio - self.rref
		#theta0 = self.theta - self.thetaref
		r0 = self.radio - self.rref
		theta0 = self.theta - self.thetaref
		uref = self.uref
		z0 = self.z - self.zref
		V0 = self.V - self.Vref
		chi0 = self.chi
		phi = self.phi
		zo = 0
		Ts = self.Ts
		V = self.V
		R_ref = self.rref
		PSI0 = self.psi
		qr = self.qr
		qz = self.qz
		qtheta = self.qtheta
		qchi = self.qchi
		qv = self.qv
		r = self.r
		rchi = self.rchi
		#radioT = 0
		#thetaT = 0

		#z =  Qr*(r0 - Ts*V*(np.cos(theta0)+np.cos(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V))) - rref)**2 + Qr*(r0 - Ts*V*(np.cos(theta0)+np.cos(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V)) +np.cos(theta0 + Ts*(V*np.sin(theta0)/r0 + V*np.sin(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V))/(r0 - Ts*V*np.cos(theta0)) + ((U[0]-uref)+(U[1]-uref))/V))) - rref)**2 + Qr*(r0 - Ts*V*(np.cos(theta0)+np.cos(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V)) +np.cos(theta0 + Ts*(V*np.sin(theta0)/r0 + V*np.sin(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V))/(r0 - Ts*V*np.cos(theta0)) + ((U[0]-uref)+(U[1]-uref))/V)) + np.cos(theta0 + Ts*(V*np.sin(theta0)/r0 + V*np.sin(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V))/(r0 - Ts*V*np.cos(theta0)) + V*np.sin(theta0 + Ts*(V*np.sin(theta0)/r0 + V*np.sin(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V))/(r0 - Ts*V*np.cos(theta0)) + ((U[0]-uref)+(U[1]-uref))/V))/(r0 - Ts*V*(np.cos(theta0)+np.cos(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V)))) + ((U[0]-uref)+(U[1]-uref)+(U[2]-uref))/V))) - rref)**2 + Qtheta*(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V) - thetaref)**2 + Qtheta*(theta0 + Ts*(V*np.sin(theta0)/r0 + V*np.sin(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V))/(r0 - Ts*V*np.cos(theta0)) + ((U[0]-uref) + (U[1]-uref))/V) - thetaref)**2 + Qtheta*(theta0 + Ts*(V*np.sin(theta0)/r0 + V*np.sin(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V))/(r0 - Ts*V*np.cos(theta0)) + V*np.sin( theta0 + Ts*(V*np.sin(theta0)/r0 + V*np.sin(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V))/(r0 - Ts*V*np.cos(theta0)) + ((U[0]-uref)+(U[1]-uref))/V))/(r0 - Ts*V*(np.cos(theta0)-np.cos(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V)))) + ((U[0]-uref)+(U[1]-uref)+(U[2]-uref))/V) - thetaref)**2 + Qtheta*(theta0 + Ts*(V*np.sin(theta0)/r0 + V*np.sin(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V))/(r0 - Ts*V*np.cos(theta0)) + V*np.sin( theta0 + Ts*(V*np.sin(theta0)/r0 + V*np.sin(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V))/(r0 - Ts*V*np.cos(theta0)) + ((U[0]-uref)+(U[1]-uref))/V))/(r0 - Ts*V*(np.cos(theta0)-np.cos(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V)))) + V*np.sin(theta0 + Ts*(V*np.sin(theta0)/r0 + V*np.sin(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V))/(r0 - Ts*V*np.cos(theta0)) + V*np.sin( theta0 + Ts*(V*np.sin(theta0)/r0 + V*np.sin(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V))/(r0 - Ts*V*np.cos(theta0)) + ((U[0]-uref)+(U[1]-uref))/V))/(r0 - Ts*V*(np.cos(theta0)-np.cos(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V)))) + ((U[0]-uref)+(U[1]-uref)+(U[2]-uref))/V))/(r0 - Ts*V*(np.cos(theta0)-np.cos(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V)) -np.cos(theta0 + Ts*(V*np.sin(theta0)/r0 + V*np.sin(theta0 + Ts*(V*np.sin(theta0)/r0 + (U[0]-uref)/V))/(r0 - Ts*V*np.cos(theta0)) + ((U[0]-uref)+(U[1]-uref))/V)))) + ((U[0]-uref)+(U[1]-uref)+(U[2]-uref)+(U[3]-uref))/V) - thetaref)**2 + R*(U[0]-uref)**2 + R*(U[1]-uref)**2 + R*(U[2]-uref)**2 + R*(U[3]-uref)**2


		for i in range(0, len(u), 3):
			#if i == 0:
			#	radio = r0
			#	theta = theta0
			#thetaT = theta0 + Ts*((V*np.cos(theta0)/(r0+rref)) - V/rref + (U[i]-uref))

			thetaT = theta0 + Ts * ((V * (np.cos(chi0)) * np.cos(theta0) / (r0 + R_ref)) + u[i] + uref )
			radioT = r0 + Ts*V*(np.cos(chi0))*np.sin(theta0)
			PSIT = PSI0 + Ts * u[i] + uref
			zT = z0 + Ts * V * np.sin(chi0)
			chiT = chi0 + Ts * u[i + 1]
			vT = V0 + u[i + 2]
			####### Modelo que tenia antes
			#radioT = r0 - self.Ts * (V0 + self.Vref)*(np.cos(self.chi))*np.cos(theta0 + self.thetaref) + self.Ts*dx*np.cos(phi) + self.Ts*dy*np.sin(phi)
			#thetaT = theta0 + self.Ts*((V0 + self.Vref)*(np.cos(self.chi))*np.sin(theta0 + self.thetaref)/(r0+self.rref) + (u[i]+self.uref) + du + (dx*np.sin(phi) - dy*np.cos(phi))/(r0+self.rref))
			#zT = z0 + self.Ts*(V0 + self.Vref)*np.sin(chi0)
			#chiT = chi0 + self.Ts*u[i+1]
			#vT = V0 + self.Ts*u[i+2]
			#######
			#thetaT = theta0 + Ts * ((V / r0) * np.sin(theta0) + (u[i]))
			#radioT = r0 - Ts * V * np.cos(theta0)

			#if i != len(u):
			#zo = zo + qr*radioT**2 + qtheta*thetaT**2 + qz*zT**2 + qchi*chiT**2 + qv*vT**2 + R*(u[i])**2 + R*(u[i+1])**2 + R*(u[i+2])**2
			zo = zo + qr*radioT**2 + qtheta*thetaT**2 + qz*zT**2 + qchi*chiT**2 + qv*vT**2 + r * (u[i]) ** 2 + rchi*(u[i+1]) ** 2 + r*(u[i+2])**2

			#z = z + Qr*(radioT-rref)**2 + Qtheta*(thetaT-thetaref)**2 + R*(u[i]-uref)**2
			#else:
				#z = z + 0.0255*(radioT-rref)**2 + 7.1442*(thetaT-thetaref)**2
			#	zo = zo + 0.0255*radioT**2 + 7.1442*thetaT**2
			#+ 2*thetaT*radioT*0.6702
			#thetaT = theta0 + Ts * ((V / r0) * np.sin(theta0) + (U[i]))
			#radioT = r0 - Ts * V * np.cos(theta0)

			#z = z + Qr * (radioT - rref) ** 2 + Qtheta * (thetaT - thetaref) ** 2 + R * (U[i] - uref) ** 2
			r0 = radioT
			theta0 = thetaT
			PSI0 = PSIT
			z0 = zT
			chi0 = chiT
			V0 = vT

		return zo

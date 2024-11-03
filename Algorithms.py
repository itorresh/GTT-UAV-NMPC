import numpy as np
from random import random, uniform

from numba import float32, int32
from numba.experimental import jitclass

from Functions import NMPC
bat_algorithm_spec = [
	('range', float32[:]),
	('Population_Size', int32),
	('Population', float32[:, :]),
	('Fitness', float32[:, :]),
	('Loudness', float32[:, :]),
	('Pulse_Rate', float32[:, :]),
	('Frequency', float32[:, :]),
	('Velocity', float32[:, :]),
	('Movements', int32),
	('Frequency_Range', float32[:]),
	('Loudness_Decay', float32),
	('Loudness_Limit', float32),
	('Pulse_Rate_Decay', float32),
	('Gamma', float32),
	('Best_Position', float32[:, :]),
	('Best_Fitness', float32),
	('Best_Bat', int32),
	('dimension', int32),
	('pop_init', float32[:, :]),
]


@jitclass(bat_algorithm_spec)
class Bat_Algorithm(object):
	def __init__(self, optimiser, Population_Size=100, Num_Movements=100, dimension=10, pop_init=np.ones(shape=(30, 1), dtype=np.float32)):
		self.range = optimiser.range

		self.dimension = 3*dimension
		self.Population_Size = Population_Size
		self.Population, self.Fitness = self.Initialise_Population(optimiser)

		self.Loudness = np.ones(shape=(self.Population_Size, 1), dtype=np.float32)
		self.Pulse_Rate = np.zeros(shape=(self.Population_Size, 1), dtype=np.float32)
		self.Frequency = np.zeros(shape=(self.Population_Size, self.dimension), dtype=np.float32)
		self.Velocity = np.zeros(shape=(self.Population_Size, self.dimension), dtype=np.float32)

		self.Movements = Num_Movements
		self.Frequency_Range = np.array([0, 1], dtype=np.float32)
		self.Loudness_Decay = 0.5
		self.Loudness_Limit = 0.0
		self.Pulse_Rate_Decay = 0.9
		self.Gamma = 0.9

		self.Best_Position = np.empty(shape=(1, self.dimension), dtype=np.float32)
		self.Best_Fitness = 0.0
		self.Best_Bat = 0

		self.pop_init = pop_init

	def Initialise_Population(self, optimiser):
		Population = np.empty(shape=(self.Population_Size, self.dimension), dtype=np.float32)
		Fitness = np.empty(shape=(self.Population_Size, 1), dtype=np.float32)

		for i in range(self.Population_Size):
			for j in range(self.dimension):
				Population[i, j] = (self.range[1] - self.range[0]) * float32(random()) + self.range[0]

			Fitness[i, 0] = optimiser.query(Population[i])

		Population[1, 0] = self.pop_init
		Fitness[1, 0] = optimiser.query(Population[1])

		return Population, Fitness

	def Update_Dynamics(self, bat, best):
		self.Frequency[bat, :] = self.Frequency_Range[0] + (self.Frequency_Range[1] - self.Frequency_Range[0]) * np.random.rand(1, self.Population.shape[1])
		self.Velocity[bat, :] = (self.Population[best, :] - self.Population[bat, :]) * self.Frequency[bat, :]
		Position = self.Population[bat, :] + self.Velocity[bat, :]

		return Position

	def Run(self, optimiser):
		best_fit = self.Fitness.min()
		best_index = self.Fitness.argmin()

		for step in range(1, self.Movements + 1):
			for bat in range(self.Population_Size):
				Position = self.Update_Dynamics(bat, best_index)

				if random() < self.Pulse_Rate[bat, :]:
					Position += self.Loudness.mean() * float32(uniform(-1, 1))

				for j in range(0, len(Position), 3):
					if Position[j] < optimiser.range[0]:
						Position[j] = optimiser.range[0]
					elif Position[j] > optimiser.range[1]:
						Position[j] = optimiser.range[1]
					if Position[j+1] < -0.2:
						Position[j+1] = -0.2
					elif Position[j+1] > 0.2:
						Position[j+1] = 0.2
					if Position[j + 2] < -0.1:
						Position[j + 2] = -0.1
					elif Position[j + 2] > 0.1:
						Position[j + 2] = -0.1

				Bat_Fitness = optimiser.query(Position)

				if Bat_Fitness < self.Fitness[bat, :]:
					self.Fitness[bat, :] = Bat_Fitness
					self.Population[bat, :] = Position
					self.Pulse_Rate[bat, :] = self.Pulse_Rate_Decay*(1-np.exp(-self.Gamma*step))

					if self.Loudness[bat, 0] > self.Loudness_Limit:
						self.Loudness[bat, 0] *= self.Loudness_Decay
					else:
						self.Loudness[bat, 0] = self.Loudness_Limit

					if Bat_Fitness < best_fit:
						best_fit = Bat_Fitness
						best_index = bat

						self.Best_Position = Position.reshape(1, self.dimension)
						self.Best_Fitness = Bat_Fitness
						self.Best_Bat = bat

if __name__ == "__main__":

	fn = NMPC(10, np.pi/2, 10, np.pi/2, 10, 10/10, 1)
	optimiser = Bat_Algorithm(fn, Population_Size=10, Num_Movements=30, dimension=4)
	print(optimiser.Population)
	print(optimiser.Best_Position)
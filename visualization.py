from odeSolver import PiecewiseLinearInterpolant
from matplotlib import pyplot as plt
import quantityOfInterest

def plotHosts(solutionInterpolant:PiecewiseLinearInterpolant):
    _, axs = plt.subplots(2,2,sharey=True)
    axs[0,0].plot(solutionInterpolant.grid, solutionInterpolant.values[:,5])
    axs[0,0].set_ylabel('S(Hosts)')
    axs[0,1].plot(solutionInterpolant.grid, solutionInterpolant.values[:,6])
    axs[0,1].set_ylabel('E(Hosts)')
    axs[1,0].plot(solutionInterpolant.grid, solutionInterpolant.values[:,7])
    axs[1,0].set_ylabel('I(Hosts)')
    axs[1,1].plot(solutionInterpolant.grid, solutionInterpolant.values[:,8])
    axs[1,1].set_ylabel('R(Hosts)')
    plt.show()

def plotMosquitos(solutionInterpolant:PiecewiseLinearInterpolant):
    _, axs = plt.subplots(3,2,sharey=True)
    axs[0,0].plot(solutionInterpolant.grid, solutionInterpolant.values[:,0])
    axs[0,0].set_ylabel('Eggs')
    axs[0,1].plot(solutionInterpolant.grid, solutionInterpolant.values[:,1])
    axs[0,1].set_ylabel('Juveniles')
    axs[1,0].plot(solutionInterpolant.grid, solutionInterpolant.values[:,2])
    axs[1,0].set_ylabel('S(Mosquitos)')
    axs[1,1].plot(solutionInterpolant.grid, solutionInterpolant.values[:,3])
    axs[1,1].set_ylabel('E(Mosquitos)')
    axs[2,0].plot(solutionInterpolant.grid, solutionInterpolant.values[:,4])
    axs[2,0].set_ylabel('I(Mosquitos)')
    totalMosquitoPopulation = quantityOfInterest.numberOfMosquitos(solutionInterpolant)
    axs[2,1].plot(totalMosquitoPopulation.grid, totalMosquitoPopulation.values)
    axs[2,1].set_ylabel('Total Population(Mosquitos)')
    plt.show()

def plotMosquitoToHostRatio(solutionInterpolant: PiecewiseLinearInterpolant):
    ratioInterpolant = quantityOfInterest.mosquitoToHostRatio(solutionInterpolant)
    plt.plot(ratioInterpolant.grid, ratioInterpolant.values)
    plt.show()
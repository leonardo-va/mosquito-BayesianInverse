from odeSolver import PiecewiseLinearInterpolant
from matplotlib import pyplot as plt
import quantityOfInterest
import numpy as np
from scipy.stats import gaussian_kde

def plotQoi(qoi, label = ""):
    plt.plot(qoi.grid, qoi.values)
    plt.ylabel(label)
    plt.show()
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

def plotMosquitoPopulation(solutionInterpolants: list[PiecewiseLinearInterpolant]):
    # Mosquito Eggs/Juveniles/Total
    
    for solutionInterpolant in solutionInterpolants:
        eggs = quantityOfInterest.eggs(solutionInterpolant)
        juveniles = quantityOfInterest.juveniles(solutionInterpolant)
        total = quantityOfInterest.numberOfMosquitos(solutionInterpolant)
        plt.plot(eggs.grid, eggs.values, label='Eggs')
        plt.plot(juveniles.grid, juveniles.values, label='Juveniles')
        plt.plot(total.grid, total.values, label='Total Mosquitos')
    
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Mosquito Population Over Time')
    plt.legend()
    plt.show()

def compare_gt_and_prediction(solution_interpolant_gt:PiecewiseLinearInterpolant,
                              solution_interpolant_pred:PiecewiseLinearInterpolant,
                              observable:dict,
                              stddev = 0,
                              prediction_label = 'mean',
                              xlabel = 'days',
                              ylabel = 'quantity'):
    qoi_coefficients = observable['linear_combination']
    qoi_label = observable['name']
    qoi_gt = quantityOfInterest.linearCombinationQOI(solution_interpolant_gt, qoi_coefficients)
    qoi_pred = quantityOfInterest.linearCombinationQOI(solution_interpolant_pred, qoi_coefficients)
    fig, axs = plt.subplots(1,1,figsize=(12,8))
    
    axs.plot(qoi_gt.grid, qoi_gt.values, label=f"{qoi_label} groundtruth", color="green")
    if(stddev > 0):
        axs.fill_between(qoi_gt.grid.flatten(), qoi_gt.values - stddev, qoi_gt.values + stddev, 
                         color='lightblue', alpha=0.5, label='±1 Std Dev')
    axs.plot(qoi_pred.grid, qoi_pred.values, label=f"{qoi_label} {prediction_label} prediction", color="red")
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.legend()
    plt.show()
    

def visualize_artificial_data(data, qoi_names):
      
    noisy_us = np.array(data.noisyData['y'])
    true_us = np.array(data.truth['y'])
    for quantity_idx in np.arange(noisy_us.shape[1]):
        plt.fill_between(data.truth['ts'], 
                            true_us[:,quantity_idx] - data.standardDeviations[quantity_idx], 
                            true_us[:,quantity_idx] + data.standardDeviations[quantity_idx], 
                            color='lightblue', alpha=0.5, label='±1 Std Dev')
        plt.plot(data.noisyData['ts'],noisy_us[:,quantity_idx], 'o',color='r',label='measurements')
        plt.plot(data.truth['ts'],true_us[:,quantity_idx], color='g',label='truth')
        plt.title(f"{qoi_names[quantity_idx]}")
        plt.xlabel("days")
        plt.ylabel("quantity")
        plt.legend()
        plt.show()

def visualize_prior_to_posterior(posterior_samples, parameter_name = '', true_param_value = None, prior_samples = None, save_result_path = None):
    kde = gaussian_kde(posterior_samples)
    xs = np.linspace(min(posterior_samples), max(posterior_samples), 1000)
    ys = kde(xs)
    posterior_map = xs[np.argmax(ys)]
    posterior_mean = np.mean(posterior_samples)
    nBins_posterior = int(np.ceil(len(posterior_samples/100)))
    plt.hist(posterior_samples,nBins_posterior, color = 'blue',alpha=0.6, label = 'posterior')
    plt.axvline(posterior_mean, color='red', label=f"samples mean: {posterior_mean:.4g}")
    plt.axvline(posterior_map, color = 'blue', label=f"MAP point: {posterior_map:.4g}")
    plt.title(parameter_name)
    if(true_param_value is not None):
        plt.axvline(true_param_value, color = 'lightgreen', label=f'true value: {true_param_value:.4g}')
    if(prior_samples is not None):
        nBins_prior = int(np.ceil(len(prior_samples/100)))
        plt.hist(prior_samples, nBins_prior, color='lightblue', alpha=0.6, label = 'prior')
    plt.legend()
    if(save_result_path is not None):
        plt.savefig(save_result_path)
    plt.show()
    
    
    
   

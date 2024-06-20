import pickle
import os
import numpy as np
from tqdm import tqdm
def compress_ground_truth():
    # print("current directory: ", os.getcwd())
    truth_map = {}
    directory = "results/GridWorld/GroundTruth"
    error_combos = []
    for file in os.listdir(directory):
        model = file[:file.find("_")]
        seed = file[file.find("_") + 1:file.find(".")]
        with open(directory + "/" + file, "rb") as pickle_file:
            truth_data = pickle.load(pickle_file)
            try:
                truth_steps = truth_data["truth_steps"]
                truth_values = truth_data["truth_value"]
            except:
                error_combos.append((model, seed))
                continue
            truth_map[model + "_" + seed] = (truth_steps, truth_values)
        os.remove(directory + "/" + file)
    print(error_combos)
    with open("results/GridWorld/GroundTruth/truth_map.pkl", "wb") as pickle_file:
        pickle.dump(truth_map, pickle_file)
    print('done')    
def compress_results():
    name = 'MultiBandit'
    steps = 4000
    seeds = [0,1,2,3,4,5,6,7,8,9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    models = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1125, 1150, 1175, 1200, 1225, 1250, 1275, 1300, 1325, 1350, 1375, 1400, 1425, 1450, 1475, 1500, 1525, 1550, 1575, 1600, 1625, 1650, 1675, 1700, 1725, 1750, 1775, 1800, 1825, 1850, 1875, 1900, 1925, 1950, 1975, 2000, 2025, 2050, 2075, 2100, 2125, 2150, 2175, 2200, 2225, 2250, 2275, 2300, 2325, 2350, 2375, 2400, 2425, 2450, 2475, 2500, 2525, 2550, 2575, 2600, 2625, 2650, 2675, 2700, 2725, 2750, 2775, 2800, 2825, 2850, 2875, 2900, 2925, 2950, 2975, 3000, 3025, 3050, 3075, 3100, 3125, 3150, 3175, 3200, 3225, 3250, 3275, 3300, 3325, 3350, 3375, 3400, 3425, 3450, 3475, 3500, 3525, 3550, 3575, 3600, 3625, 3650, 3675, 3700, 3725, 3750, 3775, 3800, 3825, 3850, 3875, 3900, 3925, 3950, 3975, 4000, 4025, 4050, 4075, 4100, 4125, 4150, 4175, 4200, 4225, 4250, 4275, 4300, 4325, 4350, 4375, 4400, 4425, 4450, 4475, 4500, 4525, 4550, 4575, 4600, 4625, 4650, 4675, 4700, 4725, 4750, 4775, 4800, 4825, 4850, 4875, 4900, 4925, 4950, 4975, 5000]
    samplers = ["MonteCarlo", "ROS_1e4", "ROS_1e5"]
    # aggregate error data into the dataframe
    for sampler in samplers:
        first = True
        print(f'Loading data for {sampler}')
        for model in models:
            for seed in seeds:
                results_file = f'results/{name}/{sampler}/{model}_{seed}.pkl'
                with open(results_file, 'rb') as f:
                    try:
                        errs = pickle.load(f)
                        if first:
                            first = False
                            results = errs
                        else:
                            results = np.concatenate((results,errs))
                    except:
                        print(f'Error loading {results_file}')
                
        with open(f"results/{name}/{sampler}/results.pkl", "wb") as pickle_file:
            pickle.dump(results, pickle_file)
            
def make_histogram():
    '''Compares the performance of the different samplers on policy-env pairs and plots their distribution'''
    name = 'MultiBandit'
    seeds = [0,1,2,3,4,5,6,7,8,9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    models = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1125, 1150, 1175, 1200, 1225, 1250, 1275, 1300, 1325, 1350, 1375, 1400, 1425, 1450, 1475, 1500, 1525, 1550, 1575, 1600, 1625, 1650, 1675, 1700, 1725, 1750, 1775, 1800, 1825, 1850, 1875, 1900, 1925, 1950, 1975, 2000, 2025, 2050, 2075, 2100, 2125, 2150, 2175, 2200, 2225, 2250, 2275, 2300, 2325, 2350, 2375, 2400, 2425, 2450, 2475, 2500, 2525, 2550, 2575, 2600, 2625, 2650, 2675, 2700, 2725, 2750, 2775, 2800, 2825, 2850, 2875, 2900, 2925, 2950, 2975, 3000, 3025, 3050, 3075, 3100, 3125, 3150, 3175, 3200, 3225, 3250, 3275, 3300, 3325, 3350, 3375, 3400, 3425, 3450, 3475, 3500, 3525, 3550, 3575, 3600, 3625, 3650, 3675, 3700, 3725, 3750, 3775, 3800, 3825, 3850, 3875, 3900, 3925, 3950, 3975, 4000, 4025, 4050, 4075, 4100, 4125, 4150, 4175, 4200, 4225, 4250, 4275, 4300, 4325, 4350, 4375, 4400, 4425, 4450, 4475, 4500, 4525, 4550, 4575, 4600, 4625, 4650, 4675, 4700, 4725, 4750, 4775, 4800, 4825, 4850, 4875, 4900, 4925, 4950, 4975, 5000]
    samplers = ["MonteCarlo", "ROS_1e4", "ROS_1e5"]
    ros4errs = []
    ros5errs = []
    for model in tqdm(models):
        for seed in seeds:
            mc = f'results/{name}/MonteCarlo/{model}_{seed}.pkl'
            ros4 = f'results/{name}/ROS_1e4/{model}_{seed}.pkl'
            ros5 = f'results/{name}/ROS_1e5/{model}_{seed}.pkl'
            try:
                with open(mc, 'rb') as f:
                    mc_err = pickle.load(f)
                with open(ros4, 'rb') as f:
                    ros4_err = pickle.load(f)
                with open(ros5, 'rb') as f:
                    ros5_err = pickle.load(f)
                
                mcerr = np.mean(mc_err, axis=0)[-1]
                ros4errs.append(mcerr - np.mean(ros4_err, axis=0)[-1])
                ros5errs.append(mcerr - np.mean(ros5_err, axis=0)[-1])
            except Exception as e:
                print(f'Error loading model {model} seed {seed} and {e}')
                continue
    with open(f"results/{name}/comparison.pkl", "wb") as pickle_file:
        pickle.dump({'ros4':ros4errs, 'ros5':ros5errs}, pickle_file)
            
compress_ground_truth()


# make_histogram()
# compress_results()
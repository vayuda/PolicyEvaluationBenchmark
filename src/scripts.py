import pickle
import os
import numpy as np
from tqdm import tqdm
import glob
import time
import yaml


def load_yaml(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)
    
def track_offline_experiment(cfg_path, result_folder):
    config = load_yaml(cfg_path)
        
    parameters = config.get('parameters', {})
    combinations = 1
    samplers = parameters.get('sampler', {}).get('values', [])
    for param, values in parameters.items():
        if isinstance(values, dict):
            if 'values' in values:
                combinations *= len(values['values'])
                print(f"Parameter {param} has {len(values['values'])} values")
            elif 'min' in values and 'max' in values:
                combinations *= values['max'] - values['min'] + 1
                print(f"Parameter {param} has {values['max'] - values['min'] + 1} values")
        elif isinstance(values, list):
            combinations *= len(values) 
            print(print(f"Parameter {param} has {len(values['values'])} values"))
    runs = 0
    with tqdm(total=combinations, desc="Runs completed", unit="runs") as pbar:
        while runs < combinations:
            new_val = 0
            for sampler in samplers:
                new_val += len(glob.glob(f"{result_folder}/{sampler}/*.pkl"))
            pbar.update( new_val - runs)
            runs = new_val
            time.sleep(5)
    print('finished')
def compress_ground_truth(directory):
    # print("current directory: ", os.getcwd())
    truth_map = {}
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
    with open(f"{directory}/../truth_map.pkl", "wb") as pickle_file:
        pickle.dump(truth_map, pickle_file)
    print('done')    
    
def bounds_to_list(min, max):
    return list(range(min, max + 1))

def compress_results(result_folder):
    config = load_yaml('config/policy_eval.yaml')
    parameters = config.get('parameters', {})
    steps = parameters["num_episodes"]["value"]
    seeds = bounds_to_list(parameters["seed"]["min"], parameters["seed"]["max"])
    models = parameters["policy"]["values"]
    samplers = parameters["sampler"]["values"]
    # aggregate error data into the dataframe
    for sampler in samplers:
        first = True
        print(f'Loading data for {sampler}')
        for model in models:
            for seed in seeds:
                results_file = f'{result_folder}/{sampler}/{model}_{seed}.pkl'
                try:
                    with open(results_file, 'rb') as f:
                        errs = pickle.load(f)
                    if first:
                        first = False
                        results = errs
                    else:
                        results = np.concatenate((results,errs))
                except:
                    print(f'Error loading {results_file}')
                    continue
                
        with open(f"{result_folder}/{sampler}_results.pkl", "wb") as pickle_file:
            pickle.dump(results, pickle_file)
            
            
def save_final_values(result_folder):
    '''Compares the performance of the different samplers on policy-env pairs and plots their distribution'''
    config = load_yaml('config/policy_eval.yaml')
    parameters = config.get('parameters', {})
    steps = parameters["num_episodes"]["value"]
    seeds = bounds_to_list(parameters["seed"]["min"], parameters["seed"]["max"])
    models = parameters["policy"]["values"]
    samplers = parameters["sampler"]["values"]
    data = {}
    for sampler in samplers:
        sampler_err = []
        for model in tqdm(models):
            for seed in seeds:
                filepath1 = f'{result_folder}/MonteCarlo/{model}_{seed}.pkl'
                filepath2 = f'{result_folder}/{sampler}/{model}_{seed}.pkl'
                try:
                    with open(filepath1, 'rb') as f:
                        mc = pickle.load(f)
                    with open(filepath2, 'rb') as f:
                        other = pickle.load(f)
                    
                    mcerr = np.mean(mc, axis=0)[-1]
                    sampler_err.append(mcerr - np.mean(other, axis=0)[-1])
                except Exception as e:
                    print(f'Error loading model {model} seed {seed}: {e}')
                    continue
        data[sampler] = sampler_err
    with open(f"{result_folder}/final_means.pkl", "wb") as pickle_file:
        pickle.dump(data, pickle_file)
            
if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'compress_ground_truth':
        compress_ground_truth(sys.argv[2])
    elif sys.argv[1] == 'track':
        track_offline_experiment(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'compress_results':
        compress_results(sys.argv[2])
    elif sys.argv[1] == 'save_final':
        save_final_values(sys.argv[2])
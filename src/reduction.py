import pickle
import os


def compress_ground_truth():
    # print("current directory: ", os.getcwd())
    truth_map = {}
    directory = "results/MultiBandit/GroundTruth"
    for file in os.listdir(directory):
        model = file[:file.find("_")]
        seed = file[file.find("_") + 1:file.find(".")]
        with open(directory + "/" + file, "rb") as pickle_file:
            truth_data = pickle.load(pickle_file)
            truth_steps = truth_data["truth_steps"]
            truth_values = truth_data["truth_value"]
            truth_map[model + "_" + seed] = (truth_steps, truth_values)
        # os.delete(directory + "/" + file)
    with open("results/MultiBandit/GroundTruth/truth_map.pkl", "wb") as pickle_file:
        pickle.dump(truth_map, pickle_file)
    
            

# compress_ground_truth()
with open("results/MultiBandit/GroundTruth/truth_map.pkl", "rb") as pickle_file:
    truth_map =pickle.load(pickle_file)
    print(truth_map["3500_0"])
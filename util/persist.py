import os
import platform
import re
import pickle
import pandas as pd
import matplotlib.pyplot as plt


def store_generated_data(data, name):
    print("Saving {} data to files".format(name))
    for advertiser, data in data.items():
        file_name = "./generated/{}_{}.npz".format(name, advertiser)
        if platform.system() == "Darwin":
            n_bytes = 2 ** 31
            max_bytes = 2 ** 31 - 1
            bytes_out = pickle.dumps(data)
            with open(file_name, 'wb') as f_out:
                for idx in range(0, n_bytes, max_bytes):
                    f_out.write(bytes_out[idx:idx + max_bytes])
        else:
            data.to_pickle(file_name)
    print("Saved")


def load_processed_data(name, folder="./generated", include=None, exclude=None):
    data = {}
    filtered_files = [x for x in os.listdir(folder) if os.path.isfile(os.path.join(folder, x)) and name in x]

    for file in filtered_files:
        match_obj = re.match(".*" + name + "_([^_]+)\.npz", file)
        if match_obj is not None:
            filename = os.path.join(folder, file)
            if exclude is not None:
                if match_obj.group(1) not in exclude:
                    print("Loading {}".format(filename))
                    data[match_obj.group(1)] = pd.read_pickle(filename)
            elif include is not None:
                if match_obj.group(1) in include:
                    print("Loading {}".format(filename))
                    data[match_obj.group(1)] = pd.read_pickle(filename)
            else:
                print("Loading {}".format(filename))
                data[match_obj.group(1)] = pd.read_pickle(filename)

    return data


def save_with_pickle(thing, file_name):
    with open(file_name, 'wb') as fid:
        pickle.dump(thing, fid)


def load_from_pickle(file_name):
    with open(file_name, 'rb') as fid:
        thing = pickle.load(fid)
    return thing

try:
    closed_figure_count
except NameError:
    closed_figure_count = 0

def save_all_figures():
    global closed_figure_count

    for i in plt.get_fignums():
        plt.figure(i)
        plt.savefig('./generated/figure_{}.png'.format(closed_figure_count, plt))
        plt.close()
        closed_figure_count += 1

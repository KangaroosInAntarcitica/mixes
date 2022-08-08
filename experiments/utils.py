import pandas as pd
from mixes import Evaluator
import numpy as np
from sklearn import metrics
import matplotlib.colors as pltcolor


UCI_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
COLOR_GREEN = pltcolor.hex2color("#48ff97")
COLOR_BLUE = pltcolor.hex2color("#4871ff")
COLOR_ORANGE = pltcolor.hex2color("#fdaa24")
COLOR_RED = pltcolor.hex2color("#ff485d")


def test_algorithms_on_data(
        alg_functions, alg_names, data, labels, num_repeats=1,
        verbose=True, return_evaluators=False):
    assert len(alg_functions) == len(alg_names)
    print("Num datapoints: %d, num features: %d, num clusters: %d" %
      (data.shape[0], data.shape[1], len(np.unique(labels))))

    result = []
    evals = {}

    for alg_i in range(len(alg_names)):
        silhouette = []
        accuracy = []
        ari = []
        log_lik = []
        num_repeated = 0

        if return_evaluators:
            evals[alg_names[alg_i]] = []

        while num_repeated < num_repeats:
            if verbose:
                print("\rAlg %s run %3d / %3d" %
                      (alg_names[alg_i], num_repeated + 1, num_repeats), end='')
            evaluator = Evaluator(print_metrics=False)
            alg = alg_functions[alg_i](evaluator)

            try:
                alg.fit(data)
                clust = alg.predict(data)
            except Exception as e:
                continue

            num_repeated += 1
            silhouette.append(metrics.silhouette_score(data, clust))
            accuracy.append(Evaluator.accuracy(labels, clust))
            ari.append(metrics.adjusted_rand_score(labels, clust))
            if len(evaluator.get_values()["log_lik"]) > 0:
                log_lik.append(evaluator.get_result_metric('log_lik'))

            if return_evaluators:
                evals[alg_names[alg_i]].append(evaluator)

        best_i = np.argmax(silhouette)
        print("")
        result.append([
            np.mean(silhouette),
            silhouette[best_i],
            1 - np.mean(accuracy),
            1 - accuracy[best_i],
            np.mean(ari),
            ari[best_i],
            np.mean(log_lik) if len(log_lik) > 0 else "-",
            np.max(log_lik) if len(log_lik) > 0 else "-"
        ])

    columns=["sil", "sil best",
             "m.r.", "m.r. best",
             "ari", "ari best",
             "log lik", "log lik best"]
    result = pd.DataFrame(result, columns=columns, index=alg_names)

    if return_evaluators:
        return result, evals
    return result

def map_labels(labels):
    mapping = {v: i for i, v in enumerate(np.unique(labels))}
    labels = np.array([mapping[x] for x in labels])
    return labels

def load_olive():
    OLIVE_DATA_URL = 'https://www.scss.tcd.ie/~arwhite/Teaching/STU33011/olive.csv'
    data = pd.read_csv(OLIVE_DATA_URL)

    # Use data[:,1] for area
    data, labels = data.values[:,2:], data.values[:,0]
    labels = map_labels(labels)
    return data, labels

def load_ecoli():
    ECOLI_DATA_URL = UCI_URL + 'ecoli/ecoli.data'
    data = pd.read_csv(ECOLI_DATA_URL, header=None, delim_whitespace=True)

    data, labels = data.values[:,1:-1].astype('float'), data.values[:,-1]
    labels = map_labels(labels)
    return data, labels

def load_vehicle():
    VEHICLE_DATA_URL = 'https://datahub.io/machine-learning/vehicle/r/vehicle.csv'
    data = pd.read_csv(VEHICLE_DATA_URL)

    data, labels = data.values[:,:-1], data.values[:,-1]
    labels = map_labels(labels)
    return data, labels

def load_satellite():
    DATA_URL_TRN = UCI_URL + 'statlog/satimage/sat.trn'
    DATA_URL_TST = UCI_URL + 'statlog/satimage/sat.tst'
    data_trn = pd.read_csv(DATA_URL_TRN, header=None, delim_whitespace=True)
    data_tst = pd.read_csv(DATA_URL_TST, header=None, delim_whitespace=True)
    data = pd.concat([data_trn, data_tst])

    data, labels = data.values[:,:-1], data.values[:,-1]
    labels = map_labels(labels)
    return data, labels

def load_gestures():
    """
    Source: https://archive.ics.uci.edu/ml/datasets/Gesture+Phase+Segmentation
    """
    DATA_URL = UCI_URL + '00302/gesture_phase_dataset.zip'

    import tempfile
    import zipfile
    import os
    import urllib.request

    # Make temporary folder (will be deleted on PC restart)
    folder = tempfile.mkdtemp()

    # Download zip folder
    response = urllib.request.urlopen(DATA_URL)
    zip_file = '%s/gesture.zip' % folder
    with open(zip_file,'wb') as output:
      output.write(response.read())

    # Extract zip folder
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(folder)

    # Read the files and concatenate
    filenames = np.sort(["%s/%s" % (folder, f) for f in os.listdir(folder)
                    if f.endswith('raw.csv')])

    # Return as pandas dataframe
    df = pd.concat([pd.read_csv(f) for f in filenames])\
        .reset_index().drop(columns='index')
    data, labels = df.values[:,:-2], df.values[:,-1]
    labels = map_labels(labels)
    return data, labels

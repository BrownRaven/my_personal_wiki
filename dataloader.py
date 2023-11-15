import os
import requests
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

def download(url, filename, delete_if_interrupted=True, chunk_size=4096):
    """ saves file from url to filename with a fancy progressbar """
    try:
        with open(filename, "wb") as f:
            print("Downloading {} > {}".format(url, filename))
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                total_length = int(total_length)
                with tqdm(total=total_length) as progressbar:
                    for data in response.iter_content(chunk_size=chunk_size):
                        if data:  # filter-out keep-alive chunks
                            f.write(data)
                            progressbar.update(len(data))
    except Exception as e:
        if delete_if_interrupted:
            print("Removing incomplete download {}.".format(filename))
            os.remove(filename)
        raise e
    return filename

def fetch_YEAR(path, train_size=None, valid_size=None, test_size=51630):
    data_path = os.path.join(path, 'data.csv')
    if not os.path.exists(data_path):
        os.makedirs(path, exist_ok=True)
        download('https://www.dropbox.com/s/l09pug0ywaqsy0e/YearPredictionMSD.txt?dl=1', data_path)
    n_features = 91
    types = {i: (np.float32 if i != 0 else np.int64) for i in range(n_features)}
    data = pd.read_csv(data_path, header=None, dtype=types)
    data_train, data_test = data.iloc[:-test_size], data.iloc[-test_size:]

    X_train, y_train = data_train.iloc[:, 1:].values, data_train.iloc[:, 0].values
    X_test, y_test = data_test.iloc[:, 1:].values, data_test.iloc[:, 0].values

    if all(sizes is None for sizes in (train_size, valid_size)):
        train_idx_path = os.path.join(path, 'stratified_train_idx.txt')
        valid_idx_path = os.path.join(path, 'stratified_valid_idx.txt')
        if not all(os.path.exists(fname) for fname in (train_idx_path, valid_idx_path)):
            download("https://www.dropbox.com/s/00u6cnj9mthvzj1/stratified_train_idx.txt?dl=1", train_idx_path)
            download("https://www.dropbox.com/s/420uhjvjab1bt7k/stratified_valid_idx.txt?dl=1", valid_idx_path)
        train_idx = pd.read_csv(train_idx_path, header=None)[0].values
        valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
    else:
        assert train_size, "please provide either train_size or none of sizes"
        if valid_size is None:
            valid_size = len(X_train) - train_size
            assert valid_size > 0
        if train_size + valid_size > len(X_train):
            warnings.warn('train_size + valid_size = {} exceeds dataset size: {}.'.format(
                train_size + valid_size, len(X_train)), Warning)

        shuffled_indices = np.random.permutation(np.arange(len(X_train)))
        train_idx = shuffled_indices[:train_size]
        valid_idx = shuffled_indices[train_size: train_size + valid_size]

    return dict(
        X_train=X_train[train_idx], y_train=y_train[train_idx],
        X_valid=X_train[valid_idx], y_valid=y_train[valid_idx],
        X_test=X_test, y_test=y_test,
    )

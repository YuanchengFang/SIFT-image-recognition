import os
from tqdm import tqdm
import numpy as np
import joblib
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from tiny import load_tiny_features
from sift import load_sift_features, load_vocab


# NOTE: train and test data directory
TRAIN_DIR = './train'
TEST_DIR = './test'

# NOTE: directory to store kmeans model
MODEL_DIR = './models'


def tiny_classifier():
    features, labels = load_tiny_features(TRAIN_DIR)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(features, labels)
    return clf


def evaluate_tiny(clf):
    X, y = load_tiny_features(TEST_DIR)
    y_pred = clf.predict(X)
    accuracy = (y_pred == y).sum() / y.shape[0]
    return accuracy


def sift_cluster_classifer(n_clusters=64):
    features, labels = load_sift_features(TRAIN_DIR)
    vocab = load_vocab(features)

    model_path = os.path.join(MODEL_DIR, 'kmeans_sift.joblib')
    if os.path.exists(model_path):
        print('Load kmeans models from', model_path)
        cluster = joblib.load(model_path)
    else:
        print('Training KMeans cluster...')
        cluster = KMeans(n_clusters=n_clusters).fit(vocab)
        joblib.dump(cluster, model_path)
    
    X = []
    for des in tqdm(features, desc='Generate histograms'):
        pred = cluster.predict(des)
        # bincount
        count = np.bincount(pred, minlength=64)
        # normalize
        count = count / np.linalg.norm(count)
        X.append(count)
    
    X = np.stack(X, axis=0)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X, labels)
    return cluster, clf


def evaluate_sift(cluster, clf):
    features, y = load_sift_features(TEST_DIR)
    X = []
    for des in tqdm(features, desc='Generate histograms'):
        pred = cluster.predict(des)
        # bincount
        count = np.bincount(pred, minlength=64)
        # normalize
        count = count / np.linalg.norm(count)
        X.append(count)
    X = np.stack(X, axis=0)

    y_pred = clf.predict(X)
    accuracy = (y_pred == y).sum() / y.shape[0]
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='select method: sift or tiny')
    parser.add_argument('--method', type=str, default='sift')

    args = parser.parse_args()

    if args.method.lower() == 'tiny':
        clf = tiny_classifier()
        acc = evaluate_tiny(clf)
        print('\n\nTiny images method accuracy: %f'%acc)
    elif args.method.lower() == 'sift':
        cluster, clf = sift_cluster_classifer()
        acc = evaluate_sift(cluster, clf)
        print('\n\nSIFT bag of words method accuracy: %f'%acc)
    else:
        raise AttributeError('%s is not a valid method, only support sift or tiny.'%args.method)

    
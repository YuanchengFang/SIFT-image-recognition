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

# NOTE:
RESULTS_DIR = './results'


def tiny_classifier():
    features, labels = load_tiny_features(TRAIN_DIR)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(features, labels)
    return clf


def evaluate_tiny(clf):
    X, y = load_tiny_features(TEST_DIR)
    y_pred = clf.predict(X)
    accuracy = (y_pred == y).sum() / y.shape[0]
    return accuracy, y_pred, y


def sift_cluster_classifer(features, labels, n_clusters=64, vocab_size=64000):
    vocab = load_vocab(features, vocab_size=vocab_size)

    model_path = os.path.join(MODEL_DIR, 'kmeans_%d.joblib'%vocab_size)
    if os.path.exists(model_path):
        print('Load kmeans models from', model_path)
        cluster = joblib.load(model_path)
    else:
        print('Training KMeans cluster...')
        cluster = KMeans(n_clusters=n_clusters).fit(vocab)
        if not os.path.exists(MODEL_DIR):
            os.mkdir(MODEL_DIR)
        joblib.dump(cluster, model_path)
    
    X = []
    for des in tqdm(features, desc='Generate train histograms'):
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


def evaluate_sift(features, labels, cluster, clf):
    X = []
    for des in tqdm(features, desc='Generate test histograms'):
        pred = cluster.predict(des)
        # bincount
        count = np.bincount(pred, minlength=64)
        # normalize
        count = count / np.linalg.norm(count)
        X.append(count)
    X = np.stack(X, axis=0)

    y_pred = clf.predict(X)
    accuracy = (y_pred == labels).sum() / y_pred.shape[0]
    return accuracy, y_pred


def analyze_vocabsize():
    test_sizes = np.int32([16, 32, 64, 96]) * 1000
    # test_sizes = np.int32([16, 32, 40, 50, 64, 70, 80, 96]) * 1000

    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    with open(os.path.join(RESULTS_DIR, 'vocab.csv'), 'w', encoding='utf-8') as f:
        f.write('Vocabsize\tAccuracy\n')
    
    features, labels = load_sift_features(TRAIN_DIR)
    features_test, labels_test = load_sift_features(TEST_DIR)
    for vocab_size in test_sizes:
        cluster, clf = sift_cluster_classifer(features, labels, vocab_size=vocab_size)
        acc, _ = evaluate_sift(features_test, labels_test, cluster, clf)
        with open(os.path.join(RESULTS_DIR, 'vocab.csv'), 'a', encoding='utf-8') as f:
            f.write('%d\t%f\n'%(vocab_size, acc))
        

def analyze_category():
    features_train, labels_train = load_sift_features(TRAIN_DIR)
    cluster, clf = sift_cluster_classifer(features_train, labels_train)
    
    categories = os.listdir(TRAIN_DIR)
    categories = [cat.lower() for cat in categories]

    features, labels = load_sift_features(TEST_DIR)
    acc, y_pred = evaluate_sift(features, labels, cluster, clf)

    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    with open(os.path.join(RESULTS_DIR, 'category.csv'), 'w', encoding='utf-8') as f:
        f.write('Category\tNumber\tAccuracy\n')
        f.write('%s\t%d\t%f\n'%('Total', labels.shape[0], acc))
    
    for index, cat in enumerate(categories):
        correct = (y_pred == labels)
        correct_cat = correct[labels == index]
        number = correct_cat.shape[0]
        acc = correct_cat.sum() / number
        with open(os.path.join(RESULTS_DIR, 'category.csv'), 'a', encoding='utf-8') as f:
            f.write('%s\t%d\t%f\n'%(cat, number, acc))


def analyze_tiny():
    clf = tiny_classifier()
    acc, y_pred, labels= evaluate_tiny(clf)
    
    categories = os.listdir(TRAIN_DIR)
    categories = [cat.lower() for cat in categories]

    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    with open(os.path.join(RESULTS_DIR, 'category_tiny.csv'), 'w', encoding='utf-8') as f:
        f.write('Category\tNumber\tAccuracy\n')
        f.write('%s\t%d\t%f\n'%('Total', labels.shape[0], acc))
    
    for index, cat in enumerate(categories):
        correct = (y_pred == labels)
        correct_cat = correct[labels == index]
        number = correct_cat.shape[0]
        acc = correct_cat.sum() / number
        with open(os.path.join(RESULTS_DIR, 'category_tiny.csv'), 'a', encoding='utf-8') as f:
            f.write('%s\t%d\t%f\n'%(cat, number, acc))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='select METHOD: sift or tiny or select analyze: vocab or category')
    parser.add_argument('--method', type=str, default='sift')
    parser.add_argument('--analyze', type=str, default='null')

    args = parser.parse_args()
    
    if args.analyze == 'vocab':
        analyze_vocabsize()
        exit()
    elif args.analyze == 'category':
        analyze_category()
        exit()
    elif args.analyze == 'tiny':
        analyze_tiny()
        exit()
    elif args.analyze != 'null':
        raise AttributeError('%s is not a valid analyze target, only support vocab or category.'%args.analyze)

    # Only if do no analysis, run below codes
    if args.method.lower() == 'tiny':
        clf = tiny_classifier()
        acc, _, _ = evaluate_tiny(clf)
        print('\n\nTiny images method accuracy: %f'%acc)
    elif args.method.lower() == 'sift':
        features, labels = load_sift_features(TRAIN_DIR)
        cluster, clf = sift_cluster_classifer(features, labels)
        features_test, labels_test = load_sift_features(TEST_DIR)
        acc, _ = evaluate_sift(features_test, labels_test, cluster, clf)
        print('\n\nSIFT bag of words method accuracy: %f'%acc)
    else:
        raise AttributeError('%s is not a valid method, only support sift or tiny.'%args.method)

    
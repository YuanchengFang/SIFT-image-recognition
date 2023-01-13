import cv2 as cv
import numpy as np
from tqdm import tqdm

from dataset import Dataset


def load_sift_features(basedir:str):
    """Read images from the specified directory, load image labels and then 
    use SIFT to detect features. 

    Parameters
    ----------
    basedir: str
        the directory path

    Returns
    -------
    features: np.ndarray
        SIFT features

    labels: np.ndarray
        labels
    """
    dataset = Dataset(basedir)
    imgs, labels = dataset.data

    descriptor = cv.SIFT_create()
    features = [descriptor.detectAndCompute(img, None)[1] for img in tqdm(imgs, desc='Detect SIFT features')]
    return features, labels


def load_vocab(features, vocab_size=64000):
    """Read images from the specified directory, then use SIFT
    Bag of Words method to caculate features and labels. 

    Parameters
    ----------
    features: List[np.ndarray]
        List of SIFT feature for each image.
    
    vocab_size: int
        total number of SIFT features sampled from train set.

    Returns
    -------
    vocab: np.ndarray
        sampled vocabulary, shape (≈vocab_size, 128)
    """
    samp_each = vocab_size // len(features)

    des_sample = []
    for des in features:
        ids = np.random.choice(des.shape[0], size=samp_each)
        des_sample.append(np.array(des[ids]))
    vocab = np.concatenate(des_sample, axis=0)
    return vocab



if __name__ == '__main__':
    f, l = load_sift_features('./train')
    v = load_vocab(f)
    print(v.shape)   
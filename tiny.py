import cv2 as cv
import numpy as np

from dataset import Dataset

def load_tiny_features(basedir:str):
    """Read images from the specified directory, then use tiny 
    images method to caculate features and labels. 

    Parameters
    ----------
    basedir: str
        the directory path

    Returns
    -------
    features: np.ndarray
        features used to classify images
        
    labels: np.ndarray
        labels
    """
    dataset = Dataset(basedir)
    imgs, labels = dataset.data

    # resize to tiny images
    imgs_tiny = [cv.resize(gray, (16, 16)) for gray in imgs]
    # normalize
    imgs_tiny = [(img - img.mean()) / img.std() for img in imgs_tiny]
    features = np.stack(imgs_tiny, axis=0)
    # flatten
    features = features.reshape((features.shape[0], -1))
    return features, labels



if __name__ == '__main__':
    f, l = load_tiny_features('./train')
    print(f.shape, l.shape)
import cv2 as cv
import numpy as np
import os


class Dataset:
    """Read images and their labels"""
    def __init__(self, basedir:str):

        self.basedir = basedir

        categories = os.listdir(self.basedir)
        filenames = {}
        for cat in categories:
            dirpath = os.path.join(self.basedir, cat)
            filenames[cat.lower()] = [os.path.join(dirpath, name) for name in os.listdir(dirpath)]
        
        self.labelnames = list(filenames.keys())
        self.filenames = filenames

        self.images = None
        self.labels = None

        print('Read image data in', self.basedir)

        self.read_data()


    @property
    def data(self):
        """
        Returns
        ----------------
        images: list of np.ndarray
        labels: np.ndarray with np.int32 dtype.
        """
        return self.images, self.labels
    
    
    def read_data(self):
        filenames = self.filenames
        images = []
        labels = []
        for i, filenames in enumerate(filenames.values()):
            imgs = [cv.imread(filename, cv.IMREAD_GRAYSCALE) for filename in filenames]
            y = np.ones(len(imgs), dtype=np.int32) * i
            images += imgs
            labels.append(y)
        self.images = images
        self.labels = np.concatenate(labels, axis=0)
    

if __name__ == '__main__':
    dataset = Dataset('./train')
    X, y = dataset.data
    print(dataset.labelnames)
    print('read %d images.'%len(X))
    print('labels shape: ', y.shape)

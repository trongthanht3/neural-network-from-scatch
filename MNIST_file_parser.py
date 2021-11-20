import os
import numpy as np
import gzip

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def read(dataset = "training", path = os.path.join('./')):

    path = os.path.join (path, 'MNIST_data')

    if dataset is "training":
        image_file = os.path.join(path, 'train-images-idx3-ubyte.gz')
        label_file = os.path.join(path, 'train-labels-idx1-ubyte.gz')
        num_images = 50000

    elif dataset is "testing":
        image_file = os.path.join(path, 't10k-images-idx3-ubyte.gz')
        label_file = os.path.join(path, 't10k-labels-idx1-ubyte.gz')
        num_images = 10000

    # extract images & labels; need to reshape labels for np.concatenate
    images = _extract_images(image_file, num_images)
    labels = _extract_labels(label_file, num_images).reshape(num_images, 1)

    images -= int(np.mean(images))
    images /= int(np.std(images))

    return images, labels

def _extract_images(image_file, num_images):

    ROWS = 28
    COLS = 28

    with gzip.open(image_file) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(ROWS * COLS * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, ROWS * COLS)

    return data    

def _extract_labels(label_file, num_images):
    
    with gzip.open(label_file) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    
    return labels

x, y = read("testing")
k = np.ravel(y)
print(k.shape)
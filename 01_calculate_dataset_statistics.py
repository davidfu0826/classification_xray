import os, glob

import yaml
import numpy as np
from tqdm import tqdm
import torch
from torchvision.transforms import Compose, Resize, ToTensor

from utils.data import CustomImageDataset

def get_test_transforms(img_size: int) -> Compose:
    return Compose([
        Resize([img_size, img_size], interpolation=3),
        ToTensor()
    ])

def compute_std_of_two_stds(distr1, distr2):
    
    (mean1, std1, n1) = distr1
    (mean2, std2, n2) = distr2
    var1 = std1**2
    var2 = std2**2
    assert not np.isnan(var1)
    assert not np.isnan(var2)
    assert not np.isnan(mean1)
    assert not np.isnan(mean2)    
    assert not np.isnan(n1)
    assert not np.isnan(n2)

    mean = (n1 /(n1+n2))*mean1 + (n2 /(n1+n2))*mean2
    variance = ( n1**2*var1 + n2**2*var2 - n1*var1 - n1*var2 - n2*var1 - n2*var2 + n1*n2*var1 + n1*n2*var2 +n1*n2*(mean1 - mean2)**2 ) / ( (n1+n2-1)*(n1+n2) )
    return mean, variance**0.5, n1+n2

def divide_and_merge(arr):
    length = len(arr)
    if length == 2:
        return compute_std_of_two_stds(arr[0], arr[1])
    elif length == 1:
        return arr[0]
    elif length > 2:
        cutoff = int(length/2)
        child_result1 = divide_and_merge(arr[cutoff:])
        child_result2 = divide_and_merge(arr[:cutoff])
        return compute_std_of_two_stds(child_result1, child_result2)

if __name__ == "__main__":
    print("Compute statistics start")
    train_imgs = glob.glob("../../Datasets/train_test_classification_full_size/train/*/*.jpg")
    labels = set([os.path.basename(os.path.dirname(img_path)) for img_path in train_imgs])
    dataset = CustomImageDataset(train_imgs, get_test_transforms(256), labels)

    
    means, stds, ns = list(), list(), list()
    t_iter = tqdm(dataset)
    for x, y in t_iter:
        means.append(float(x.mean()))
        stds.append(float(x.std()))
        ns.append(np.prod(x.shape))
        t_iter.set_description(f"Mean: {means[-1]}, Std: {stds[-1]}")

    print("Computing statistics recursivelly...")

    arr = list()
    for a in zip(means, stds, ns):
        arr.append(a)
    results = divide_and_merge(arr)
    print(f"(mean, std, num_of_pixels) = {results}")

    try:
        with open('data.yaml', 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
    except:
        data = dict()

    data['pixel_mean'] = float(results[0])
    data['pixel_std']  = float(results[1])
    data['labels'] = list(labels)
    with open("data.yaml", "w") as file:
        yaml.dump(data, file)
    
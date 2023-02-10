import torch

def getMask(seq, mode):
    ratio = np.random.uniform()
    print(ratio)
    if mode == "linear":
        return 1 - ratio
    elif mode == "cosine":
        return np.cos(ratio * np.pi / 2)
    elif mode == "square":
        return 1 - ratio ** 2
    elif mode == "cubic":
        return 1 - ratio ** 3

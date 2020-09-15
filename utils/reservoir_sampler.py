import torch
import pdb
# class ReservoirSampler:
#     def __init__(self):
#         pass


def time_window_random_historical_sampling(time2quads_train, time, train_seq_len, num_samples_each_time_step):
    start_time_step = max(0, time - train_seq_len)
    quad_samples = []

    for t in range(start_time_step, time):
        quads = time2quads_train[t]
        perm = torch.randperm(quads.size(0))
        idx = perm[:num_samples_each_time_step]
        samples = quads[idx]
        quad_samples.append(samples)
    return torch.cat(quad_samples)

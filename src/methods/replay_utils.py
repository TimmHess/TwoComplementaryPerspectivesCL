import random
import copy

import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset


@torch.no_grad()
def compute_dataset_logits(dataset, model, batch_size, device, num_workers=0):
    was_training = model.training
    model.eval()

    logits = []
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    for x, _, _ in loader:
        x = x.to(device)
        out = model(x)
        out = out.detach().cpu()

        for row in out:
            logits.append(torch.clone(row))

    if was_training:
        model.train()

    return logits


def retrieve_random_buffer_batch(storage_policy, n_samples):
    """
    Retrieve a batch of exemplars from the rehearsal memory.
    First sample indices for the available tasks at random, then actually extract from rehearsal memory.
    There is no resampling of exemplars.

    :param n_samples: Number of memories to return
    :return: input-space tensor, label tensor
    """
    assert n_samples > 0, "Need positive nb of samples to retrieve!"

    # Determine how many mem-samples available
    q_total_cnt = 0  # Total samples
    free_q = {}  # idxs of which ones are free in mem queue
    tasks = []
    for t, ex_buffer in storage_policy.buffer_groups.items():
        mem_cnt = len(ex_buffer.buffer)  # Mem cnt
        free_q[t] = list(range(0, mem_cnt))  # Free samples
        q_total_cnt += len(free_q[t])  # Total free samples
        tasks.append(t)

    # Randomly sample how many samples to idx per class
    free_tasks = copy.deepcopy(tasks)
    tot_sample_cnt = 0
    sample_cnt = {c: 0 for c in tasks}  # How many sampled already
    max_samples = n_samples if q_total_cnt > n_samples else q_total_cnt  # How many to sample (equally divided)
    while tot_sample_cnt < max_samples:
        t_idx = random.randrange(len(free_tasks))
        t = free_tasks[t_idx]  # Sample a task

        if sample_cnt[t] >= len(storage_policy.buffer_groups[t].buffer):  # No more memories to sample    
            free_tasks.remove(t)
            continue
        sample_cnt[t] += 1
        tot_sample_cnt += 1

    # Actually sample
    s_cnt = 0
    subsets = []
    for t, t_cnt in sample_cnt.items():
        if t_cnt > 0:
            # Set of idxs
            cnt_idxs = torch.randperm(len(storage_policy.buffer_groups[t].buffer))[:t_cnt]
            sample_idxs = cnt_idxs.unsqueeze(1).expand(-1, 1)
            sample_idxs = sample_idxs.view(-1)

            # Actual subset
            s = Subset(storage_policy.buffer_groups[t].buffer, sample_idxs.tolist())
            subsets.append(s)
            s_cnt += t_cnt
    assert s_cnt == tot_sample_cnt == max_samples
    new_dset = ConcatDataset(subsets)
    return new_dset


def load_buffer_batch(storage_policy, train_mb_size, device):
    """
    Wrapper to retrieve a batch of exemplars from the rehearsal memory
    :param nb: Number of memories to return
    :return: input-space tensor, label tensor
    """

    ret_x, ret_y, ret_t = None, None, None
    # Equal amount as batch: Last batch can contain fewer!
    n_exemplars = train_mb_size
    new_dset = retrieve_random_buffer_batch(storage_policy, n_exemplars)  # Dataset object
    
    # Load the actual data
    for sample in DataLoader(new_dset, batch_size=len(new_dset), pin_memory=True, shuffle=False):
        x_s, y_s = sample[0].to(device), sample[1].to(device)
        t_s = sample[-1].to(device)  # Task label (for multi-head)

        ret_x = x_s if ret_x is None else torch.cat([ret_x, x_s])
        ret_y = y_s if ret_y is None else torch.cat([ret_y, y_s])
        ret_t = t_s if ret_t is None else torch.cat([ret_t, t_s])

    return ret_x, ret_y, ret_t


def load_entire_buffer(storage_policy, device):
    buffers = []
    for t, _ in storage_policy.buffer_groups.items():
        buffers.append(storage_policy.buffer_groups[t].buffer)
    new_dset = ConcatDataset(buffers)
    return new_dset
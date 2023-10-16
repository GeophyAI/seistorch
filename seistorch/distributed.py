from mpi4py import MPI
from mpi4py.util import pkl5


comm = pkl5.Intracomm(MPI.COMM_WORLD)
rank = comm.Get_rank()
size = comm.Get_size()

def task_distribution_and_data_reception(shots, pbar, mode, num_batches=10, **kwargs):
    """This function is used to distribute tasks to workers and receive results from workers.

    Args:
        shots (np.ndarray): The array of shots.
        pbar (_type_): A progress bar.
        mode (str): The mode of the task, either forward or inversion.

    Returns:
        _type_: Tuple of results.
    """

    assert mode in ['forward', 'inversion'], "mode must be either forward or inversion"

    if mode == 'forward':
        record = kwargs['record']
    elif mode == 'inversion':
        loss = kwargs['loss']
        epoch = kwargs['epoch']
        grad3d = kwargs['grad3d']
        idx_freq = kwargs['idx_freq']
    # if num_batches is not specified, then num_batches=shots.size
    # i.e. each shot is a batch
    num_batches = shots.size if num_batches == -1 else num_batches
    # split shots into batches
    batched_shots = split_batches(shots.tolist(), num_batches)
    num_tasks = num_batches  # total number of tasks is the number of bathes
    task_index = 0
    num_completed_tasks = 0
    active_workers = min(size-1, num_tasks)
    # send initial tasks to all workers
    for i in range(1, size):

        if task_index < num_tasks:
            # original
            # comm.send(shots[task_index], dest=i, tag=1)
            # batched version
            comm.send(batched_shots[task_index], dest=i, tag=1)

            task_index += 1
        else:
            comm.send(-1, dest=i, tag=0)

    while num_completed_tasks < num_tasks:
        # receive results from any worker
        completed_task, sender_rank, *results = comm.recv(source=MPI.ANY_SOURCE, tag=1)

        if mode == 'forward':
            for idx, shot in enumerate(completed_task):
                record[shot] = results[0][idx]

        elif mode == 'inversion':
            grad3d[completed_task[0]][:] = results[0]
            loss[idx_freq][epoch][completed_task] = results[1]

        # task_index plus one
        num_completed_tasks += 1
        pbar.update(1)

        # if there are still tasks to be completed, 
        # assign them to the worker who just completed a task
        if task_index < num_tasks:
            comm.send(batched_shots[task_index], dest=sender_rank, tag=1)
            task_index += 1
        else:
            # send stop signal to the worker who just completed a task
            comm.send(-1, dest=sender_rank, tag=0)
            active_workers -= 1

def split_batches(shot_nums, num_batches):
    """Split shots into batches. Assume num_shots=10, num_batches=3, 
       then the result is [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]].

    Args:
        shots (np.ndarray): The array of shots.
        batch_size (int): The size of each batch.

    Returns:
        _type_: List of batches.
    """

    groups = [[] for _ in range(num_batches)]  # A list of empty lists

    for i, num in enumerate(shot_nums):
        group_index = i % num_batches  # The index of the group to which the shot belongs
        groups[group_index].append(num)  # Add the shot to the group

    return groups


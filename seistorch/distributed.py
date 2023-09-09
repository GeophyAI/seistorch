from mpi4py import MPI
from mpi4py.util import pkl5


comm = pkl5.Intracomm(MPI.COMM_WORLD)
rank = comm.Get_rank()
size = comm.Get_size()

def task_distribution_and_data_reception(shots, pbar, mode, **kwargs):
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

    num_tasks = shots.size  # total number of tasks is the number of shots
    task_index = 0
    completed_tasks = 0
    active_workers = min(size-1, num_tasks)
    # send initial tasks to all workers
    for i in range(1, size):

        if task_index < num_tasks:
            comm.send(shots[task_index], dest=i, tag=1)
            task_index += 1
        else:
            comm.send(-1, dest=i, tag=0)

    while completed_tasks < num_tasks:
        # receive results from any worker
        completed_task, sender_rank, *results = comm.recv(source=MPI.ANY_SOURCE, tag=1)

        if mode == 'forward':
            record[completed_task] = results[0]

        elif mode == 'inversion':
            grad3d[completed_task][:] = results[0]
            loss[idx_freq][epoch][completed_task] = results[1]

        # task_index plus one
        completed_tasks += 1
        pbar.update(1)

        # if there are still tasks to be completed, 
        # assign them to the worker who just completed a task
        if task_index < num_tasks:
            comm.send(shots[task_index], dest=sender_rank, tag=1)
            task_index += 1
        else:
            # send stop signal to the worker who just completed a task
            comm.send(-1, dest=sender_rank, tag=0)
            active_workers -= 1
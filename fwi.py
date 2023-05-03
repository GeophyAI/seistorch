"""Perform full waveform inversion."""
import setproctitle
import wavetorch
import numpy as np
import argparse, os, sys, time, tqdm, torch, socket
from mpi4py import MPI
from wavetorch.utils import ricker_wave, to_tensor, cpu_fft
# from tensorflow.keras.models import load_model
from wavetorch.model import build_model
from wavetorch.loss import NormalizedCrossCorrelation, ElasticLoss
from wavetorch.optimizer import NonlinearConjugateGradient as NCG
from wavetorch.shape import Shape
# from skopt import Optimizer
from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from wavetorch.setup_source_probe import setup_src_coords_customer, get_sources_coordinate_list

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, 
                    help='Configuration file for geometry, training, and data preparation')
parser.add_argument('mpiconfig', type=str, 
                    help='Configuration file for multi-nodes mpirun')
parser.add_argument('--num_threads', type=int, default=2,
                    help='Number of threads to use')
parser.add_argument('--use-cuda', action='store_true',
                    help='Use CUDA to perform computations')
parser.add_argument('--encoding', action='store_true',
                    help='Use phase encoding to accelerate performance')
parser.add_argument('--name', type=str, default=time.strftime('%Y%m%d%H%M%S'),
                    help='Name to use when saving or loading the model file. If not specified when saving a time and date stamp is used')
parser.add_argument('--opt', choices=['adam', 'lbfgs', 'ncg'], default='adam',
                    help='optimizer (adam)')
parser.add_argument('--mode', choices=['forward', 'inversion', 'rtm'], default='forward',
                    help='forward modeling, inversion or reverse time migration mode')

if __name__ == '__main__':
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.use_cuda and torch.cuda.is_available():
        # Get the local_rank on each node
        if socket.gethostname()!="gitlg15":
            torch.cuda.set_device(rank%torch.cuda.device_count())
        else:
            torch.cuda.set_device(rank%2)
        args.dev = torch.cuda.current_device()
        if rank ==0:
            print("Configuration: %s" % args.config)
            print("Using CUDA for calculation")
            print(f"Optimizer: {args.opt}")
    else:
        args.dev = torch.device('cpu')
        if rank ==0:
            print("Configuration: %s" % args.config)
            print("Using CPU for calculation")

    'Sets the number of threads used for intraop parallelism on CPU.'
    torch.set_num_threads(args.num_threads)
    # Build model
    cfg, model = build_model(args.config, device=args.dev, mode=args.mode)
    #model = torch.compile(model)

    # Set the name of the process
    if rank!=0:
        setproctitle.setproctitle(cfg['name'])
    else:
        setproctitle.setproctitle("TaskAssign")

    """Short cuts of the configures"""
    ELASTIC = cfg['equation'] == 'elastic'
    ACOUSTIC = cfg['equation'] == 'acoustic'
    EPOCHS = cfg['training']['N_epochs']
    NSHOTS = cfg['geom']['Nshots']
    LEARNING_RATE = cfg['training']['lr']
    NORMALIZATION = cfg["training"]["normalize"]
    FILTER_ORDER = cfg['training']['filter_ord']
    ### Get source-x and source-y coordinate in grid cells
    source_x_list, source_y_list = get_sources_coordinate_list(cfg)

    use_mpi = size > 1
    if (use_mpi and rank!=0) or (not use_mpi):
        model.to(args.dev)
        model.train()

        """# Read the wavelet"""
        x = ricker_wave(cfg['geom']['fm'], cfg['geom']['dt'], cfg['geom']['nt'])
        #.to(args.dev)

    shape = Shape(cfg)

    """---------------------------------------------"""
    """-------------------MODELING------------------"""
    """---------------------------------------------"""
    
    if args.mode == 'forward':

        if rank==0:
            record = np.zeros(shape.record3d, dtype=np.float32)
        else:
            record = np.zeros(shape.record2d, dtype=np.float32)
            x = torch.unsqueeze(x, 0)

        comm.Barrier()
        # 主节点
        if rank == 0:
            pbar = tqdm.trange(cfg['geom']['Nshots'], position=0)
            pbar.set_description(cfg['equation'])
            num_tasks = cfg['geom']['Nshots']  # 任务总数=炮数
            task_index = 0
            completed_tasks = 0
            active_workers = min(size-1, num_tasks)
            # 向所有其他节点发送初始任务
            for i in range(1, size):
                if task_index < num_tasks:
                    comm.send(task_index, dest=i, tag=1)
                    task_index += 1
                else:
                    comm.send(-1, dest=i, tag=0)

            while completed_tasks < num_tasks:
                # 接收已完成任务的节点信息
                completed_task, sender_rank, record[completed_task][:]= comm.recv(source=MPI.ANY_SOURCE, tag=1)
                # 任务计数器
                completed_tasks += 1
                pbar.update(1)

                # 如果还有未完成任务，分配新任务给完成任务的节点
                if task_index < num_tasks:
                    comm.send(task_index, dest=sender_rank, tag=1)
                    task_index += 1
                else:
                    # 向已完成任务的节点发送停止信号
                    comm.send(-1, dest=sender_rank, tag=0)
                    active_workers -= 1
        else:
            while True:
                # 接收任务
                task = comm.recv(source=0, tag=MPI.ANY_TAG)

                # 如果接收到停止信号，跳出循环
                if task == -1:
                    break

                # Forward modeling
                with torch.no_grad():
                    shot = task
                    source = setup_src_coords_customer(source_x_list[shot], source_y_list[shot], cfg['geom']['Nx'], cfg['geom']['Ny'], cfg['geom']['pml']['N'])
                    model.reset_sources(source)
                    y = model(x)
                    record[:] = y.cpu().detach().numpy()

                comm.send((task, rank, record), dest=0, tag=1)

        comm.Barrier()

        """Save the modeled data, which stored in rank0 <record>"""
        if rank==0:
            pbar.close()
            print("Modeling done, data will be writing to disk.")
            np.save(cfg['geom']['obsPath'], record)


    """---------------------------------------------"""
    """-------------------INVERSION-----------------"""
    """---------------------------------------------"""

    if args.mode == 'inversion':


        """Write configure file to the inversion folder"""
        if rank==0:
            with open(os.path.join(cfg['geom']['inv_savePath'], "configure.yml"), "w") as f:
                dump(cfg, f)

        """Define Optimizer"""
        if args.opt=='adam':
            if ACOUSTIC:
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-16)
            if ELASTIC: 
                optimizer = torch.optim.Adam([
                        {'params': model.cell.get_parameters('vp'), 'lr':LEARNING_RATE},
                        {'params': model.cell.get_parameters('vs'), 'lr':LEARNING_RATE/1.73},
                        {'params': model.cell.get_parameters('rho'), 'lr':0.}], 
                        betas=(0.9, 0.999), eps=1e-16)
                
                # opt_bayesian = Optimizer(dimensions=[(-10.0, 10.0)]*shape.numel, 
                #                          base_estimator="gp", n_initial_points=0, acq_func="EI")

        if args.opt == "ncg":
            optimizer = NCG(model.parameters(), lr=0.001, max_iter=10)

        """Define the learning rate decay"""
        lr_milestones = [EPOCHS*(i+1) for i in range(len(cfg['geom']['multiscale']))]

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                            milestones=lr_milestones, 
                                                            gamma = cfg['training']['lr_decay'], 
                                                            verbose=False)
    
        """Define the misfit function"""
        if NORMALIZATION and ACOUSTIC:
            criterion = NormalizedCrossCorrelation()
        else:
            criterion = torch.nn.MSELoss()


        """Only rank0 will read the full band data"""
        """Rank0 will broadcast the data after filtering"""
        if rank == 0:
            full_band_data = np.load(cfg['geom']['obsPath'])
            filtered_data = np.zeros(shape.record3d, dtype=np.float32)
            loss = np.zeros((len(cfg['geom']['multiscale']), EPOCHS, NSHOTS), np.float32)
            # The gradient in rank0 is a 3D array.
            grad3d = np.zeros(shape.grad3d, np.float32)
            grad2d = np.zeros(shape.grad2d, np.float32)
        else:
            filtered_data = np.zeros(shape.record3d, dtype=np.float32)
            # The gradient of other ranks are 2D arrays.
            grad2d = np.zeros(shape.grad2d, np.float32)
            #hessian = np.zeros(shape.hessian, np.float32)

        """Loop over all scale"""
        for idx_freq, freq in enumerate(cfg['geom']['multiscale']):

            if rank==0:
                print(f"Data filtering: frequency:{freq}")
                # Filter both record and ricker
                filtered_data[:] = cpu_fft(full_band_data.copy(), cfg['geom']['dt'], N=FILTER_ORDER, low=freq, axis = 1, mode='lowpass')
            else:
                filtered_data[:] = 0.

            # Broadcast the filtered data to other processors
            comm.Bcast(filtered_data, root=0)

            if (use_mpi and rank!=0) or (not use_mpi):
                # Low pass filtered wavelet
                if isinstance(x, torch.Tensor): x = x.numpy()
                lp_wavelet = cpu_fft(x.copy(), cfg['geom']['dt'], N=FILTER_ORDER, low=freq, axis=0, mode='lowpass')
                lp_wavelet = torch.unsqueeze(torch.from_numpy(lp_wavelet), 0)

            """Loop over all epoches"""
            for epoch in range(EPOCHS):

                # 主节点
                if rank == 0:
                    pbar = tqdm.tqdm(range(0, NSHOTS), leave=False)
                    pbar.set_description(f"Freq{idx_freq}Epoch{epoch}")
                    num_tasks = NSHOTS  # 任务总数=炮数
                    task_index = 0
                    completed_tasks = 0
                    active_workers = min(size-1, num_tasks)
                    # 向所有其他节点发送初始任务
                    for i in range(1, size):
                        if task_index < num_tasks:
                            comm.send(task_index, dest=i, tag=1)
                            task_index += 1
                        else:
                            comm.send(-1, dest=i, tag=0)

                    while completed_tasks < num_tasks:
                        # 接收已完成任务的节点信息
                        completed_task, sender_rank, _grad, _loss = comm.recv(source=MPI.ANY_SOURCE, tag=1)
                        grad3d[completed_task][:] = _grad
                        loss[idx_freq][epoch][completed_task] = _loss
                        # 任务计数器
                        completed_tasks += 1
                        pbar.update(1)
                        # 如果还有未完成任务，分配新任务给完成任务的节点
                        if task_index < num_tasks:
                            comm.send(task_index, dest=sender_rank, tag=1)
                            task_index += 1
                        else:
                            # 向已完成任务的节点发送停止信号
                            comm.send(-1, dest=sender_rank, tag=0)
                            active_workers -= 1
                else:
                    while True:
                        # 接收任务
                        task = comm.recv(source=0, tag=MPI.ANY_TAG)
                        # 如果接收到停止信号，跳出循环
                        if task == -1:
                            break
                        sources = []
                        shot = task
                        src = setup_src_coords_customer(source_x_list[shot],
                                                        source_y_list[shot],
                                                        cfg['geom']['Nx'],
                                                        cfg['geom']['Ny'],
                                                        cfg['geom']['pml']['N'])
                        sources.append(src)

                        """Calculate one shot gradient"""
                        def closure(srcs):
                            optimizer.zero_grad()
                            shot_nums_cur_epoch = [shot]
                            """Although it is a for loop """
                            """But only one shot here when traditional workflow is using"""
                            for _src, shot_num in zip(srcs, shot_nums_cur_epoch):
                                model.reset_sources(_src)
                                ypred = model(lp_wavelet)
                                target = to_tensor(filtered_data[shot_num]).to(ypred.device)#.unsqueeze(0)
                                loss = criterion(ypred, target)
                                """HvP Start"""
                                # Perform a backward pass to compute the gradients
                                # grads = torch.autograd.grad(loss, [model.cell.geom.vp], create_graph=True)
                                # # Define a vector v with the same size as the model parameters
                                # v = [torch.randn_like(param) for param in [model.cell.geom.vp]]
                                # # Perform a forward pass with the vector v
                                # grads_v = torch.autograd.grad(grads, [model.cell.geom.vp], grad_outputs=v)
                                # # Perform a backward pass to compute the Hessian-vector product
                                # HvP = torch.autograd.grad(grads_v, [model.cell.geom.vp], retain_graph=True)
                                # np.save(os.path.join(cfg["geom"]["inv_savePath"], f"HvPE{epoch}S{shot_num}.npy"), HvP)
                                """HvP End"""
                                """START"""
                                # Model regularization
                                # l1_reg = 0
                                # for mname in model.cell.geom.model_parameters:
                                #     if mname == 'rho':
                                #         continue
                                #     l1_reg += torch.norm(model.cell.geom.__getattr__(mname), p=1)
                                # # Assign the weight of the model regulazation to %10 of the obj.
                                # alpha = loss.item()*1e-16
                                # loss += alpha*l1_reg
                                """END"""

                                loss.backward()
                            return loss.item()

                        # Run the closure
                        loss = closure(sources)

                        GRAD = list()
                        for mname in model.cell.geom.model_parameters:
                            GRAD.append(model.cell.geom.__getattr__(mname).grad.cpu().detach().numpy())
                        GRAD = np.array(GRAD)
                        # Get the gram_schmidt_orthogonalization

                        # GRAD[1], GRAD[0] = gram_schmidt_orthogonalization(GRAD[1], GRAD[0])
                        comm.send((task, rank, GRAD, loss), dest=0, tag=1)

                comm.Barrier()

                """"Assigning and Saving"""
                if rank == 0:
                    pbar.close()
                    # Calculate the gradient of other ranks
                    grad2d[:] = np.sum(grad3d, axis=0)
                    np.save(f"{cfg['geom']['inv_savePath']}/loss.npy", loss)

                comm.Bcast(grad2d, root=0)

                if rank!=0:
                    # Assign gradient of other ranks
                    for idx, para in enumerate(model.cell.geom.model_parameters):
                        var = model.cell.geom.__getattr__(para)
                        var.grad.data = to_tensor(grad2d[idx]).to(args.dev)
                    # Update the model parameters and learning rate
                    optimizer.step()
                    lr_scheduler.step()

                if rank==1:
                    # Save vel and grad
                    model.cell.geom.save_model(cfg['geom']['inv_savePath'], 
                                               paras=["vel", "grad"], 
                                               freq_idx=idx_freq, 
                                               epoch=epoch)


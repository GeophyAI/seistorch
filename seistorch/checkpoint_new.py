# This file is modified from torch.utils.checkpoint.checkpoint
# for making it available for BPTT in seismic inversion.
import torch
import inspect

import warnings
import weakref
import itertools
import numpy as np
from typing import Any, Iterable, List, Tuple

from seistorch.equations3d.utils import save_boundaries as sb3d
from seistorch.equations2d.utils import save_boundaries as sb2d

__all__ = [
    "checkpoint", "checkpoint_sequential", "CheckpointFunction",
    "check_backward_validity", "detach_variable", "get_device_states",
    "set_device_states",
]

def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)


def check_backward_validity(inputs: Iterable[Any]) -> None:
    if not any(inp.requires_grad for inp in inputs if isinstance(inp, torch.Tensor)):
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")


# We can't know if the run_fn will internally move some args to different devices,
# which would require logic to preserve rng states for those devices as well.
# We could paranoically stash and restore ALL the rng states for all visible devices,
# but that seems very wasteful for most cases.  Compromise:  Stash the RNG state for
# the device of all Tensor args.
#
# To consider:  maybe get_device_states and set_device_states should reside in torch/random.py?
def get_device_states(*args) -> Tuple[List[int], List[torch.Tensor]]:
    # This will not error out if "arg" is a CPU tensor or a non-tensor type because
    # the conditionals short-circuit.
    fwd_gpu_devices = list({arg.get_device() for arg in args
                            if isinstance(arg, torch.Tensor) and arg.is_cuda})

    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.cuda.device(device):
            fwd_gpu_states.append(torch.cuda.get_rng_state())

    return fwd_gpu_devices, fwd_gpu_states

def flatten_list(tuple_list):
    flattened_list = list(itertools.chain(*tuple_list))
    return flattened_list

def packup_boundaries(flattened_list, tuple_size):
    bd_tuple = (tuple(flattened_list[i:i+tuple_size]) for i in range(0, len(flattened_list), tuple_size))
    return bd_tuple

def set_device_states(devices, states) -> None:
    for device, state in zip(devices, states):
        with torch.cuda.device(device):
            torch.cuda.set_rng_state(state)

def _get_autocast_kwargs():
    gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                           "dtype": torch.get_autocast_gpu_dtype(),
                           "cache_enabled": torch.is_autocast_cache_enabled()}

    cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                           "dtype": torch.get_autocast_cpu_dtype(),
                           "cache_enabled": torch.is_autocast_cache_enabled()}

    return gpu_autocast_kwargs, cpu_autocast_kwargs

def set_requires_grad(nested_list, requires_grad_iter):
    for i, item in enumerate(nested_list):
        if isinstance(item, list):
            set_requires_grad(item, requires_grad_iter)
        elif isinstance(item, torch.Tensor):
            nested_list[i] = item.requires_grad_(next(requires_grad_iter))

def requires_grad_generator(requires_grad_list):
    for requires_grad in requires_grad_list:
        yield requires_grad

def nested_grads(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.append(nested_grads(item))
        elif isinstance(item, torch.Tensor):
            result.append(item.grad)
        else:
            result.append(None)
    return result

class CheckpointFunction(torch.autograd.Function):

    wavefields = []
    counts=0
    @staticmethod
    def forward(ctx, 
                run_function, 
                back_function,
                source_function, 
                save_condition, 
                para_counts, 
                habcs,
                *args):
        # check_backward_validity(args)
        ctx.requires_grad_list = [arg.requires_grad for arg in itertools.chain(args)]
        ctx.run_function = run_function
        ctx.back_function = back_function        
        ctx.save_condition = save_condition
        ctx.source_function = source_function
        # CheckpointFunction._sources.append(source_function[-1])
        save_boundaries = sb2d if '2d' in inspect.getmodule(run_function).__name__ else sb3d

        with torch.no_grad():
            outputs = run_function(*args, habcs=habcs)

        ctx.models = args[:para_counts]
        ctx.geoms = args[::-1][0:3][::-1]

        boundarys = [save_boundaries(output) for output in outputs]
        # CheckpointFunction._bound.append(boundarys)
        ctx.save_for_backward(*itertools.chain(*boundarys))
        # Save the wavefields of the last time step
        ctx.is_last_time_step = save_condition
        ctx.lastframe = outputs if ctx.is_last_time_step else None
        return outputs

    @staticmethod
    def backward(ctx, *args):
        ACOUSTIC2nd = ctx.run_function.ACOUSTIC2nd
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")

        # Get the boundarys:
        if ctx.is_last_time_step:
            CheckpointFunction.wavefields = list(ctx.lastframe)
            if ACOUSTIC2nd == True:
                CheckpointFunction.wavefields.reverse()
            wavefields = CheckpointFunction.wavefields
            return (None, None, None, None, None, None) + tuple(None for _ in range(len(ctx.requires_grad_list)))
        else:
            wavefields = CheckpointFunction.wavefields

        CheckpointFunction.counts+=1
        if CheckpointFunction.counts == 1 and ACOUSTIC2nd:
            # When the equaion == "acoustic", it has little difference from other 
            # first order wave equations, since we need to start the backpropagation 
            # from the nt-2.
            return (None, None, None, None, None, None) + tuple(None for _ in range(len(ctx.requires_grad_list)))

        inputs = ctx.models + tuple(wavefields) + ctx.geoms
        
        inputs = [inp.detach().requires_grad_(value) for inp, value in zip(inputs, ctx.requires_grad_list)]

        # Inputs for backwards
        num_boundaries = 4 if '2d' in inspect.getmodule(ctx.run_function).__name__ else 6
        boundaries = packup_boundaries(ctx.saved_tensors, num_boundaries)
        inputs = inputs + [boundaries] + [ctx.source_function]

        with torch.enable_grad():
            outputs = ctx.back_function(*inputs)
        # if True:
        #     np.save(f"./wf_pml/wf_backward{CheckpointFunction.counts:04d}.npy", 
        #             outputs[0].detach().cpu().numpy())

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])

        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True,"
                " this checkpoint() is not necessary")

        torch.autograd.backward(outputs_with_grad, args_with_grad)
        # outputs are wavefields at previous time steps

        # assign boundary values
        outputs = list(outputs)
        #save_boundaries = sb2d if '2d' in inspect.getmodule(run_function).__name__ else sb3d
        #outputs[0] = restore_boundaries(outputs[0], list(packup_boundaries(ctx.saved_tensors, 4))[0])
        # Update wavefields
        if not (CheckpointFunction.counts == 1 and ACOUSTIC2nd) or not CheckpointFunction.counts == 0:
            CheckpointFunction.wavefields.clear()
            CheckpointFunction.wavefields.extend(list(outputs))
        
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None
                      for inp in inputs[:len(ctx.requires_grad_list)])
                
        return (None, None, None, None, None, None) + grads
 

def checkpoint(function, backfunction, source_function, save_condition, para_counts, *args, use_reentrant: bool = True, habcs=None, **kwargs):
    r"""Checkpoint a model or part of the model

    Checkpointing works by trading compute for memory. Rather than storing all
    intermediate activations of the entire computation graph for computing
    backward, the checkpointed part does **not** save intermediate activations,
    and instead recomputes them in backward pass. It can be applied on any part
    of a model.

    Specifically, in the forward pass, :attr:`function` will run in
    :func:`torch.no_grad` manner, i.e., not storing the intermediate
    activations. Instead, the forward pass saves the inputs tuple and the
    :attr:`function` parameter. In the backwards pass, the saved inputs and
    :attr:`function` is retrieved, and the forward pass is computed on
    :attr:`function` again, now tracking the intermediate activations, and then
    the gradients are calculated using these activation values.

    The output of :attr:`function` can contain non-Tensor values and gradient
    recording is only performed for the Tensor values. Note that if the output
    consists of nested structures (ex: custom objects, lists, dicts etc.)
    consisting of Tensors, these Tensors nested in custom structures will not
    be considered as part of autograd.


    .. warning::
        If :attr:`function` invocation during backward does anything different
        than the one during forward, e.g., due to some global variable, the
        checkpointed version won't be equivalent, and unfortunately it can't be
        detected.

    .. warning::
        If ``use_reentrant=True`` is specified, then if the checkpointed segment
        contains tensors detached from the computational graph by `detach()` or
        `torch.no_grad()`, the backward pass will raise an error. This is
        because `checkpoint` makes all the outputs require gradients which
        causes issues when a tensor is defined to have no gradient in the model.
        To circumvent this, detach the tensors outside of the `checkpoint`
        function. Note that the checkpointed segment can contain tensors
        detached from the computational graph if ``use_reentrant=False`` is
        specified.

    .. warning::
        If ``use_reentrant=True`` is specified, at least one of the inputs needs
        to have :code:`requires_grad=True` if grads are needed for model inputs,
        otherwise the checkpointed part of the model won't have gradients. At
        least one of the outputs needs to have :code:`requires_grad=True` as
        well. Note that this does not apply if ``use_reentrant=False`` is
        specified.

    .. warning::
        If ``use_reentrant=True`` is specified, checkpointing currently only
        supports :func:`torch.autograd.backward` and only if its `inputs`
        argument is not passed. :func:`torch.autograd.grad`
        is not supported. If ``use_reentrant=False`` is specified, checkpointing
        will work with :func:`torch.autograd.grad`.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        use_reentrant(bool, optional): Use checkpointing
            implementation that requires re-entrant autograd.
            If ``use_reentrant=False`` is specified, ``checkpoint`` will use an
            implementation that does not require re-entrant autograd. This
            allows ``checkpoint`` to support additional functionality, such as
            working as expected with ``torch.autograd.grad`` and support for
            keyword arguments input into the checkpointed function. Note that future
            versions of PyTorch will default to ``use_reentrant=False``.
            Default: ``True``
        args: tuple containing inputs to the :attr:`function`

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs and use_reentrant:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    if use_reentrant:
        return CheckpointFunction.apply(function, backfunction, source_function, save_condition, para_counts, habcs, *args)
    else:
        return _checkpoint_without_reentrant(
            function,
            preserve,
            *args,
            **kwargs,
        )


def checkpoint_sequential(functions, segments, input, use_reentrant=True, **kwargs):
    r"""A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a model in various segments
    and checkpoint each segment. All segments except the last will run in
    :func:`torch.no_grad` manner, i.e., not storing the intermediate
    activations. The inputs of each checkpointed segment will be saved for
    re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    .. warning:
        Since PyTorch 1.4, it allows only one Tensor as the input and
        intermediate outputs, just like :class:`torch.nn.Sequential`.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or
            functions (comprising the model) to run sequentially.
        segments: Number of chunks to create in the model
        input: A Tensor that is input to :attr:`functions`
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        use_reentrant(bool, optional): Use checkpointing
            implementation that requires re-entrant autograd.
            If ``use_reentrant=False`` is specified, ``checkpoint`` will use an
            implementation that does not require re-entrant autograd. This
            allows ``checkpoint`` to support additional functionality, such as
            working as expected with ``torch.autograd.grad`` and support for
            keyword arguments input into the checkpointed function.
            Default: ``True``

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> # xdoctest: +SKIP("stub")
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_sequential(model, chunks, input_var)
    """
    # Hack for keyword-only parameter in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    def run_function(start, end, functions):
        def forward(input):
            for j in range(start, end + 1):
                input = functions[j](input)
            return input
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    segment_size = len(functions) // segments
    # the last chunk has to be non-volatile
    end = -1
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        input = checkpoint(
            run_function(start, end, functions),
            input,
            use_reentrant=use_reentrant,
            preserve_rng_state=preserve
        )
    return run_function(end + 1, len(functions) - 1, functions)(input)


def _checkpoint_without_reentrant(function, preserve_rng_state=True, *args, **kwargs):
    """Checkpointining without re-entrant autograd
    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        *args: Arguments to pass in to the given ``function``.
        **kwargs: Keyword arguments to pass into the given ``function``.
    """
    # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
    gpu_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs()

    if preserve_rng_state:
        fwd_cpu_state = torch.get_rng_state()
        # Don't eagerly initialize the cuda context by accident.
        # (If the user intends that the context is initialized later, within their
        # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
        # we have no way to anticipate this will happen before we run the function.
        # If they do so, we raise an error.)
        had_cuda_in_fwd = False
        if torch.cuda._initialized:
            had_cuda_in_fwd = True
            fwd_gpu_devices, fwd_gpu_states = get_device_states(*args)

    # Custom class to be able to take weak references
    class Holder():
        pass
    # The Holder object for each of the saved object is saved directly on the
    # SavedVariable and is cleared when reset_data() is called on it. We MUST make
    # sure that this is the only object having an owning reference to ensure that
    # the Tensor stored in storage is deleted as soon as the corresponding SavedVariable
    # data is cleared.
    storage: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
    weak_holder_list = []

    def pack(x):
        # TODO(varal7): Instead of returning abstract object, we can return things metadata (such as
        # size, device, ...) to catch certain cases of undeterministic behavior of the forward
        res = Holder()
        weak_holder_list.append(weakref.ref(res))
        return res


    def unpack(x):
        unpack_counter = 0
        if len(storage) == 0:
            def inner_pack(inner):
                nonlocal unpack_counter
                unpack_counter += 1
                # If the holder went out of scope, the SavedVariable is dead and so
                # the value will never be read from the storage. Skip filling it.
                if weak_holder_list[unpack_counter - 1]() is None:
                    return
                # Use detach here to ensure we don't keep the temporary autograd
                # graph created during the second forward
                storage[weak_holder_list[unpack_counter - 1]()] = inner.detach()
                return

            def inner_unpack(packed):
                raise RuntimeError("You are calling backwards on a tensor that is never exposed. Please open an issue.")

            # Stash the surrounding rng state, and mimic the state that was
            # present at this time during forward.  Restore the surrounding state
            # when we're done.
            rng_devices = []
            if preserve_rng_state and had_cuda_in_fwd:
                rng_devices = fwd_gpu_devices
            with torch.random.fork_rng(devices=rng_devices, enabled=preserve_rng_state):
                if preserve_rng_state:
                    torch.set_rng_state(fwd_cpu_state)
                    if had_cuda_in_fwd:
                        set_device_states(fwd_gpu_devices, fwd_gpu_states)

                with torch.enable_grad(), \
                     torch.cuda.amp.autocast(**gpu_autocast_kwargs), \
                     torch.cpu.amp.autocast(**cpu_autocast_kwargs), \
                     torch.autograd.graph.saved_tensors_hooks(inner_pack, inner_unpack):
                    _unused = function(*args, **kwargs)

        if x not in storage:
            raise RuntimeError(
                "Attempt to retrieve a tensor saved by autograd multiple times without checkpoint"
                " recomputation being triggered in between, this is not currently supported. Please"
                " open an issue with details on your use case so that we can prioritize adding this."
            )

        return storage[x]

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = function(*args, **kwargs)
        if torch.cuda._initialized and preserve_rng_state and not had_cuda_in_fwd:
            # Cuda was not initialized before running the forward, so we didn't
            # stash the CUDA state.
            raise RuntimeError(
                "PyTorch's CUDA state was initialized in the forward pass "
                "of a Checkpoint, which is not allowed. Please open an issue "
                "if you need this feature.")

    return output
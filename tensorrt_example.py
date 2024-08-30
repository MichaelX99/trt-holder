import torch
from ultralytics.nn.tasks import RTDETRDetectionModel
import time
import numpy as np
import tensorrt as trt
import os
from cuda import cuda, cudart
from typing import Optional
import ctypes
from calibrator import EngineCalibrator, ImageBatcher

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
#TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

import onnx_graphsurgeon as gs
from onnx import shape_inference
import onnx
def sanitize(graph):
        """
        Sanitize the graph by cleaning any unconnected nodes, do a topological resort, and fold constant inputs values.
        When possible, run shape inference on the ONNX graph to determine tensor shapes.
        """
        for i in range(3):
            count_before = len(graph.nodes)
            print(count_before)

            graph.cleanup().toposort()
            try:
                for node in graph.nodes:
                    for o in node.outputs:
                        o.shape = None
                model = gs.export_onnx(graph)
                model = shape_inference.infer_shapes(model)
                graph = gs.import_onnx(model)
            except Exception as e:
                print(
                    "Shape inference could not be performed at this time:\n{}".format(e)
                )
            try:
                graph.fold_constants(fold_shapes=True)
            except TypeError as e:
                print(
                    "This version of ONNX GraphSurgeon does not support folding shapes, please upgrade your "
                    "onnx_graphsurgeon module. Error:\n{}".format(e)
                )
                raise

            count_after = len(graph.nodes)
            if count_before == count_after:
                # No new folding occurred in this iteration, so we can stop for now.
                break
        print(count_after)

        

#model = RTDETRDetectionModel(cfg='rtdetr-resnet50.yaml', verbose=False).to('cuda:0').eval()
model = RTDETRDetectionModel(cfg='rtdetr-l.yaml', verbose=False).to('cuda:0').eval()
#model = RTDETRDetectionModel(cfg='rtdetr-l.yaml', verbose=False).to('cuda:0').eval()

x = torch.zeros(4, 3, 1024, 1024, requires_grad=False).to('cuda:0')


for _ in range(5):
    with torch.no_grad():
        out = model(x)

with torch.autocast('cuda'):
    times = []
    for _ in range(10):
        t1 = time.perf_counter()
        with torch.no_grad():
            out = model(x)
        t2 = time.perf_counter()
        times.append(t2-t1)

print(np.median(times))


####################################################################################################################3
####################################################################################################################3
####################################################################################################################3
####################################################################################################################3

with torch.no_grad():
    out = model(x)

def prepare_onnx():
    if not os.path.exists('./rtdetr.onnx'):
        with torch.no_grad():
            torch.onnx.export(
                model.cpu(),
                x.cpu(),
                './rtdetr.onnx',
                export_params=True,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                verbose=False,
                opset_version=16,
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    #'output': {0, 'batch_size'}
                },
            )
            print('done')

def build_engine():
    trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")
    if not os.path.exists('./rtdetr-trt.engine'):
        # For more information on TRT basics, refer to the introductory samples.
        builder = trt.Builder(TRT_LOGGER)
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        config = builder.create_builder_config()

        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, 8 * (2**30)
        )  # 8 GB


        config.set_flag(trt.BuilderFlag.FP16)

        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open('./rtdetr.onnx', 'rb') as f:
        #with open('./rtdetr-opt.onnx', 'rb') as f:
            parser.parse(f.read())

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]

        print("Network Description")
        for input in inputs:
            print(
                "Input '{}' with shape {} and dtype {}".format(
                    input.name, input.shape, input.dtype
                )
            )
        for output in outputs:
            print(
                "Output '{}' with shape {} and dtype {}".format(
                    output.name, output.shape, output.dtype
                )
            )

        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        config.set_flag(trt.BuilderFlag.DIRECT_IO)
        config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)

        profile = builder.create_optimization_profile()
        min_shape = [1] + list(input.shape[1:])
        opt_shape = [4] + list(input.shape[1:])
        max_shape = [4] + list(input.shape[1:])
        profile.set_shape(input.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        """
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = EngineCalibrator('./int8_calib.cache')
        if not os.path.exists('./int8_calib.cache'):
            calib_shape = [8] + list(inputs[0].shape[1:])
            calib_dtype = trt.nptype(inputs[0].dtype)
            config.int8_calibrator.set_image_batcher(
                ImageBatcher(
                    './val2014/',
                    calib_shape,
                    calib_dtype,
                    max_num_images=500,
                    exact_batches=True,
                    shuffle_files=True,
                )
            )
        """
            
        engine = builder.build_serialized_network(network, config)

        if engine is None:
            raise RuntimeError('engine could not be created')
        
        with open('./rtdetr-trt.engine', 'wb') as f:
            f.write(engine)

    with open('./rtdetr-trt.engine', 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        assert runtime
        engine = runtime.deserialize_cuda_engine(f.read())
        assert engine

    context = engine.create_execution_context()

    return engine, context


prepare_onnx()
engine, context = build_engine()

def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res

class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""

    def __init__(self,
                 size: int,
                 dtype: np.dtype,
                 name: Optional[str] = None,
                 shape: Optional[trt.Dims] = None,
                 format: Optional[trt.TensorFormat] = None):
        nbytes = size * dtype.itemsize
        print(size, dtype)
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type),
                                           (size, ))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes
        self._name = name
        self._shape = shape
        self._format = format
        self._dtype = dtype

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, arr: np.ndarray):
        if arr.size > self.host.size:
            raise ValueError(
                f"Tried to fit an array of size {arr.size} into host memory of size {self.host.size}"
            )
        np.copyto(self.host[:arr.size], arr.flat, casting='safe')

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def shape(self) -> Optional[trt.Dims]:
        return self._shape

    @property
    def format(self) -> Optional[trt.TensorFormat]:
        return self._format

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))

# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))

class TRTInfer():
    def __init__(self, engine, context):
        self.engine = engine
        self.context = context

        stream = cuda_call(cudart.cudaStreamCreate())

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            shape = self.context.get_tensor_shape(name)
            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_tensor_profile_shape(name, 0)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                self.context.set_input_shape(name, profile_shape[2])
                shape = self.context.get_tensor_shape(name)
            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = cuda_call(cudart.cudaMalloc(size))
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            print(
                "{} '{}' with shape {} and dtype {}".format(
                    "Input" if is_input else "Output",
                    binding["name"],
                    binding["shape"],
                    binding["dtype"],
                )
            )

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

        self.stream = stream

    def infer(self, x):
        memcpy_host_to_device(self.inputs[0]["allocation"], x)
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            memcpy_device_to_host(
                self.outputs[o]["host_allocation"], self.outputs[o]["allocation"]
            )
        return [o["host_allocation"] for o in self.outputs]


trt_infer = TRTInfer(engine, context)

trt_x = x.cpu().numpy()

for _ in range(5):
    trt_infer.infer(trt_x)

times = []
for _ in range(10):
    t1 = time.perf_counter()
    trt_infer.infer(trt_x)
    t2 = time.perf_counter()
    times.append(t2-t1)

print(np.median(times))

# NOTE
# first run this script to generate onnx file
# next run onnxoptimizer cli app to optimize the onnx file
# third run this script again
import tensorrt as trt
import numpy as np
import os
import sys
import random

import numpy as np
from PIL import Image
from cuda import cuda, cudart

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

def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    def set_image_batcher(self, image_batcher):
        """
        Define the image batcher to use, if any. If using only the cache file, an image batcher doesn't need
        to be defined.
        :param image_batcher: The ImageBatcher object
        """
        self.image_batcher = image_batcher
        size = int(
            np.dtype(self.image_batcher.dtype).itemsize
            * np.prod(self.image_batcher.shape)
        )
        self.batch_allocation = cuda_call(cudart.cudaMalloc(size))
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _, _ = next(self.batch_generator)
            print(
                "Calibrating image {} / {}".format(
                    self.image_batcher.image_index, self.image_batcher.num_images
                )
            )
            memcpy_host_to_device(
                self.batch_allocation, np.ascontiguousarray(batch)
            )
            return [int(self.batch_allocation)]
        except StopIteration:
            print("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if self.cache_file is not None and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                print("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        if self.cache_file is None:
            return
        with open(self.cache_file, "wb") as f:
            print("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)


class ImageBatcher:
    """
    Creates batches of pre-processed images.
    """

    def __init__(
        self,
        input,
        shape,
        dtype,
        max_num_images=None,
        exact_batches=False,
        preprocessor="EfficientDet",
        shuffle_files=False,
    ):
        """
        :param input: The input directory to read images from.
        :param shape: The tensor shape of the batch to prepare, either in NCHW or NHWC format.
        :param dtype: The (numpy) datatype to cast the batched data to.
        :param max_num_images: The maximum number of images to read from the directory.
        :param exact_batches: This defines how to handle a number of images that is not an exact multiple of the batch
        size. If false, it will pad the final batch with zeros to reach the batch size. If true, it will *remove* the
        last few images in excess of a batch size multiple, to guarantee batches are exact (useful for calibration).
        :param preprocessor: Set the preprocessor to use, depending on which network is being used.
        :param shuffle_files: Shuffle the list of files before batching.
        """
        # Find images in the given input path
        input = os.path.realpath(input)
        self.images = []

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        def is_image(path):
            return (
                os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions
            )

        if os.path.isdir(input):
            self.images = [
                os.path.join(input, f)
                for f in os.listdir(input)
                if is_image(os.path.join(input, f))
            ]
            self.images.sort()
            if shuffle_files:
                random.seed(47)
                random.shuffle(self.images)
        elif os.path.isfile(input):
            if is_image(input):
                self.images.append(input)
        self.num_images = len(self.images)
        if self.num_images < 1:
            print("No valid {} images found in {}".format("/".join(extensions), input))
            sys.exit(1)

        # Handle Tensor Shape
        self.dtype = dtype
        self.shape = shape
        assert len(self.shape) == 4
        self.batch_size = shape[0]
        assert self.batch_size > 0
        self.format = None
        self.width = -1
        self.height = -1
        if self.shape[1] == 3:
            self.format = "NCHW"
            self.height = self.shape[2]
            self.width = self.shape[3]
        elif self.shape[3] == 3:
            self.format = "NHWC"
            self.height = self.shape[1]
            self.width = self.shape[2]
        assert all([self.format, self.width > 0, self.height > 0])

        # Adapt the number of images as needed
        if max_num_images and 0 < max_num_images < len(self.images):
            self.num_images = max_num_images
        if exact_batches:
            self.num_images = self.batch_size * (self.num_images // self.batch_size)
        if self.num_images < 1:
            print("Not enough images to create batches")
            sys.exit(1)
        self.images = self.images[0 : self.num_images]

        # Subdivide the list of images into batches
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            self.batches.append(self.images[start:end])

        # Indices
        self.image_index = 0
        self.batch_index = 0

        self.preprocessor = preprocessor

    def preprocess_image(self, image_path):
        """
        The image preprocessor loads an image from disk and prepares it as needed for batching. This includes padding,
        resizing, normalization, data type casting, and transposing.
        This Image Batcher implements one algorithm for now:
        * EfficientDet: Resizes and pads the image to fit the input size.
        :param image_path: The path to the image on disk to load.
        :return: Two values: A numpy array holding the image sample, ready to be contacatenated into the rest of the
        batch, and the resize scale used, if any.
        """

        def resize_pad(image, pad_color=(0, 0, 0)):
            """
            A subroutine to implement padding and resizing. This will resize the image to fit fully within the input
            size, and pads the remaining bottom-right portions with the value provided.
            :param image: The PIL image object
            :pad_color: The RGB values to use for the padded area. Default: Black/Zeros.
            :return: Two values: The PIL image object already padded and cropped, and the resize scale used.
            """
            width, height = image.size
            width_scale = width / self.width
            height_scale = height / self.height
            scale = 1.0 / max(width_scale, height_scale)
            image = image.resize(
                (round(width * scale), round(height * scale)), resample=Image.BILINEAR
            )
            pad = Image.new("RGB", (self.width, self.height))
            pad.paste(pad_color, [0, 0, self.width, self.height])
            pad.paste(image)
            return pad, scale

        scale = None
        image = Image.open(image_path)
        image = image.convert(mode="RGB")
        if self.preprocessor == "EfficientDet":
            # For EfficientNet V2: Resize & Pad with ImageNet mean values and keep as [0,255] Normalization
            image, scale = resize_pad(image, (124, 116, 104))
            image = np.asarray(image, dtype=self.dtype)
            # [0-1] Normalization, Mean subtraction and Std Dev scaling are part of the EfficientDet graph, so
            # no need to do it during preprocessing here
        else:
            print("Preprocessing method {} not supported".format(self.preprocessor))
            sys.exit(1)
        if self.format == "NCHW":
            image = np.transpose(image, (2, 0, 1))
        return image, scale

    def get_batch(self):
        """
        Retrieve the batches. This is a generator object, so you can use it within a loop as:
        for batch, images in batcher.get_batch():
           ...
        Or outside of a batch with the next() function.
        :return: A generator yielding three items per iteration: a numpy array holding a batch of images, the list of
        paths to the images loaded within this batch, and the list of resize scales for each image in the batch.
        """
        for i, batch_images in enumerate(self.batches):
            batch_data = np.zeros(self.shape, dtype=self.dtype)
            batch_scales = [None] * len(batch_images)
            for i, image in enumerate(batch_images):
                self.image_index += 1
                batch_data[i], batch_scales[i] = self.preprocess_image(image)
            self.batch_index += 1
            yield batch_data, batch_images, batch_scales




#####################################
from onnxruntime.quantization import CalibrationDataReader
import onnxruntime
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

def _preprocess_images(images_folder: str, height: int, width: int, size_limit=0):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + "/" + image_name
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))
        input_data = np.float32(pillow_img) - np.array(
            [123.68, 116.78, 103.94], dtype=np.float32
        )
        nhwc_data = np.expand_dims(input_data, axis=0)
        nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard
        unconcatenated_batch_data.append(nchw_data)
    batch_data = np.concatenate(
        np.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    )
    return batch_data


class ResNet50DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(
            calibration_image_folder, height, width, size_limit=16
            #calibration_image_folder, height, width, size_limit=0
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None
import onnx
def convert_model_batch_to_dynamic(model_path):
    model = onnx.load(model_path)
    initializers =  [node.name for node in model.graph.initializer]
    inputs = []
    for node in model.graph.input:
        if node.name not in initializers:
            inputs.append(node)
    input_name = inputs[0].name
    shape = inputs[0].type.tensor_type.shape
    dim = shape.dim
    if not dim[0].dim_param:
        dim[0].dim_param = 'N'
        model = onnx.shape_inference.infer_shapes(model)
        model_name = model_path.split(".")
        model_path = model_name[0] + "_dynamic.onnx"
        onnx.save(model, model_path)
    return [model_path, input_name]

from onnxruntime.quantization import CalibrationDataReader, create_calibrator, write_calibration_table
if __name__ == "__main__":
    # INT8 calibration setting
    calibration_table_generation_enable = True  # Enable/Disable INT8 calibration

    # TensorRT EP INT8 settings
    os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"  # Enable FP16 precision
    os.environ["ORT_TENSORRT_INT8_ENABLE"] = "1"  # Enable INT8 precision
    os.environ["ORT_TENSORRT_INT8_CALIBRATION_TABLE_NAME"] = "calibration.flatbuffers"  # Calibration table name
    os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"  # Enable engine caching
    execution_provider = ["TensorrtExecutionProvider"]

    """
    # Convert static batch to dynamic batch
    model_path = 'rtdetr-opt.onnx'
    augmented_model_path='rtdetr-opt_augmented.onnx'
    [new_model_path, input_name] = convert_model_batch_to_dynamic(model_path)

    calibrator = create_calibrator(new_model_path, [], augmented_model_path=augmented_model_path)
    #calibrator = create_calibrator(new_model_path, [], augmented_model_path=augmented_model_path)
    calibrator.set_execution_providers(["CUDAExecutionProvider"]) 

    dr = ResNet50DataReader('./val2014/', augmented_model_path)

    #sess = onnxruntime.InferenceSession(augmented_model_path, providers=['CUDAExecutionProvider'])
    #breakpoint()
    #exit()

    calibrator.collect_data(dr)
    write_calibration_table(calibrator.compute_range())
    """

    augmented_model_path='rtdetr.onnx'
    #augmented_model_path='rtdetr-opt-infer.onnx'
    dr = ResNet50DataReader('./val2014/', augmented_model_path)
    quantize_static(augmented_model_path, 'quant.onnx', dr, weight_type=QuantType.QInt8)
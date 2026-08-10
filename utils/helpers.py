# Standard Libraries
import numpy.random 
import tensorflow as tf
import random
import os

# 3rd Party Libraries
import cv2

def makedirs(path):
    """ Try to create the directory, pass if the directory exists already, fails otherwise.
    :param path:            <string>            directory path, that should be created

    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        if not os.path.isdir(path):
            raise

def layer_to_index(target_layer, model):
    """
    Turns the identifier of an layer into a 
    layer index, regardless if it is a str or already index

    :param target_layer: <str or int> identifier for the target layer. None will return None
    :param model: <keras.Model> The model containing the target layer

    :returns: <int> index of target layer
    """
    if target_layer is None:
        return None
    if isinstance(target_layer, str):
        for idx, layer in enumerate(model.layers):
            if layer.name == target_layer:
                return idx
    elif isinstance(target_layer, int):
        return target_layer
    else:
        raise TypeError("layer has to be int or str")

    raise Exception("Layer %s could not be found"%(str(target_layer)))

def tb_write_images(callback, names, imgs):
    """
    Write images to TensorBoard using the TensorFlow 2 summary API.

    :param callback: keras.callbacks.TensorBoard instance
    :param names: list of summary names
    :param imgs: list of image arrays
    """
    log_dir = getattr(callback, "log_dir", None)
    if not log_dir:
        return

    writer = tf.summary.create_file_writer(log_dir)

    with writer.as_default():
        for name, img in zip(names, imgs):
            img = numpy.asarray(img)

            # TensorBoard expects [N, H, W, C].
            if img.ndim == 2:
                img = numpy.expand_dims(img, axis=-1)
            if img.ndim == 3:
                img = numpy.expand_dims(img, axis=0)

            # Convert typical uint8 images to float in [0, 1].
            if img.dtype == numpy.uint8:
                img = img.astype(numpy.float32) / 255.0

            tf.summary.image(name, img, step=0, max_outputs=3)

        writer.flush()

def tb_write_texts(callback, names, texts):
    """
    Write text summaries to TensorBoard using the TensorFlow 2 summary API.

    :param callback: keras.callbacks.TensorBoard instance
    :param names: list of summary names
    :param texts: list of strings
    """
    log_dir = getattr(callback, "log_dir", None)
    if not log_dir:
        return

    writer = tf.summary.create_file_writer(log_dir)

    with writer.as_default():
        for name, text in zip(names, texts):
            tf.summary.text(name, tf.convert_to_tensor(str(text)), step=0)

        writer.flush()

def initialize_seed(seed=0):
    """
    Make experiments more reproducible across Python, NumPy and TensorFlow.
    """
    random.seed(a=seed)
    numpy.random.seed(seed)
    tf.random.set_seed(seed)

def get_session(gpu_usage=None):
    """
    Configure TensorFlow 2 GPU memory handling.

    CRF-Net originally returned a TensorFlow 1.x Session configured with
    tf.GPUOptions / tf.ConfigProto. TensorFlow 2.11 no longer uses that API.

    This compatibility implementation enables memory growth on all GPUs that
    TensorFlow can see. The legacy gpu_usage fraction is retained as an
    argument for configuration compatibility, but dynamic growth is used
    because it is the portable TensorFlow 2 behaviour.

    :param gpu_usage: legacy CRF-Net GPU-memory fraction (kept for compatibility)
    :return: None
    """
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        print("No GPU detected by TensorFlow. Training will use CPU unless the "
              "batch job exposes a GPU.")
        return None

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("TensorFlow GPU memory growth enabled.")
        print("Visible GPU(s): {}".format(gpus))

        if gpu_usage is not None:
            print(
                "Note: CRF-Net gpu_mem_usage={} is a TensorFlow 1.x setting; "
                "TensorFlow 2 memory growth is being used instead.".format(
                    gpu_usage
                )
            )

    except RuntimeError as exc:
        # This happens if GPU initialization has already occurred.
        print("GPU configuration warning: {}".format(exc))

    return None

def output_index_by_name(model, output_name):
    """
    :param model: the keras model
    :param output_name: the string name for a specific output

    :returns: <int> specifying the index of the requested output
    """
    name_to_index = {name:i for i,name in enumerate(model.output_names)}
    return name_to_index[output_name]


def input_index_by_name(model, input_name):
    """
    :param model: the keras model
    :param output_name: the string name for a specific output

    :returns: <int> specifying the index of the requested output
    """
    name_to_index = {name:i for i,name in enumerate(model.input_names)}
    return name_to_index[input_name]
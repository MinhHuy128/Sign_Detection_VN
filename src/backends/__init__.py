from .base import InferenceBackend
from ..utils.logger import get_logger
import os
logger = get_logger(__name__)

def load_best_backend(config: dict) -> InferenceBackend:
    backend_pref = config.get('model', {}).get('backend', 'auto')
    from .tensorrt_backend import TensorRTBackend
    from .onnx_backend import ONNXBackend
    from .pytorch_backend import PyTorchBackend
    paths = {'trt': config.get('model', {}).get('sign_model_trt'), 'onnx': config.get('model', {}).get('sign_model_onnx'), 'pt': config.get('model', {}).get('sign_model_pt')}
    backends_to_try = []
    if backend_pref == 'auto':
        backends_to_try = [(TensorRTBackend, paths['trt'], 'cuda'), (ONNXBackend, paths['onnx'], 'cuda'), (PyTorchBackend, paths['pt'], 'cuda')]
    else:
        backends_to_try = [(PyTorchBackend, paths['pt'], 'cuda')]
    for BackendClass, path, device in backends_to_try:
        if path and os.path.exists(path):
            try:
                backend = BackendClass()
                backend.load(path, device)
                logger.info(f'Successfully loaded backend: {backend.backend_name}')
                return backend
            except Exception as e:
                logger.warning(f'Failed to load {BackendClass.__name__}: {e}')
    logger.warning('All backends failed to load proper weights. Creating empty PyTorch backend.')
    return PyTorchBackend()
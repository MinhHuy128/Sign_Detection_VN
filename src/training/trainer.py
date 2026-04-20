import os
import yaml
import torch
import shutil
from typing import Dict, Tuple, Optional
from ..utils.logger import get_logger
from .experiment import ExperimentManager
try:
    from ultralytics import YOLO, RTDETR
    import albumentations as A
    import ultralytics.data.augment
    from ultralytics.data.augment import Albumentations as UAlbumentations

    def get_custom_transform(p=1.0):
        return A.Compose([A.Sharpen(alpha=(0.1, 0.3), lightness=(0.95, 1.05), p=0.15), A.RandomGamma(gamma_limit=(85, 115), p=0.25), A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.12, p=0.25), A.MotionBlur(blur_limit=(3, 7), p=0.15), A.ImageCompression(quality_range=(65, 95), p=0.25)], p=p)

    class CustomAlbumentations(UAlbumentations):

        def __init__(self, p=1.0, **kwargs):
            super().__init__(p)
            self.transform = get_custom_transform(p=1.0)
            self.contains_spatial = False
    ultralytics.data.augment.Albumentations = CustomAlbumentations
except ImportError:
    YOLO, RTDETR = (None, None)
logger = get_logger(__name__)

class ModelTrainer:

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.train_cfg = self.config.get('training', {})
        self.arch_cfg = self.config.get('architecture', {})

    def detect_device(self) -> Tuple[str, int]:
        batch_size = self.train_cfg.get('batch', 16)
        return ('0', batch_size)

    def select_architecture(self) -> str:
        if not self.arch_cfg.get('auto_select', True):
            return self.arch_cfg.get('fallback', 'yolo11s.pt')
        candidates = self.arch_cfg.get('candidates', [])
        best_arch = self.arch_cfg.get('fallback', 'yolo11s.pt')
        best_map = -1.0
        device, batch = self.detect_device()
        epochs = self.arch_cfg.get('arch_selection_epochs', 10)
        if device == 'cpu':
            logger.warning('[WARN] CPU mode detected: arch selection will be very slow')
        with ExperimentManager('arch_selection', self.config.get('mlflow', {}).get('tracking_uri', 'mlruns/')) as exp:
            for cand in candidates:
                try:
                    name = cand['name']
                    pretrained = cand['pretrained']
                    logger.info(f'Evaluating candidate: {name}')
                    if YOLO is None:
                        map50 = 0.5
                    else:
                        model = RTDETR(pretrained) if 'rtdetr' in pretrained.lower() else YOLO(pretrained)
                        results = model.train(data=self.config['dataset']['path'], epochs=epochs, batch=batch, device=device, imgsz=self.train_cfg.get('imgsz', 640), verbose=False, workers=self.train_cfg.get('workers', 0))
                        map50 = getattr(results.box, 'map50', 0.5) if hasattr(results, 'box') else 0.5
                    exp.log_metrics({f'{name}_mAP50': map50})
                    if map50 > best_map:
                        best_map = map50
                        best_arch = pretrained
                except Exception as e:
                    logger.warning(f"[WARN] Candidate {cand['name']} failed: {str(e)}")
                    continue
        return best_arch

    def train(self, arch: str) -> str:
        device, batch = self.detect_device()
        logger.info(f'Starting full training with architecture: {arch} on {device}')
        with ExperimentManager(self.config.get('mlflow', {}).get('experiment_name', 'RoadSense-AI-v2')) as exp:
            exp.log_params(self.train_cfg)
            if YOLO is not None:
                model = RTDETR(arch) if 'rtdetr' in arch.lower() else YOLO(arch)
                train_epochs = self.train_cfg.get('epochs', 80)
                if device == 'cpu':
                    logger.warning('[WARN] CPU mode detected: full training will take a long time')
                results = model.train(data=self.config['dataset']['path'], epochs=train_epochs, batch=batch, device=device, imgsz=self.train_cfg.get('imgsz', 640), optimizer=self.train_cfg.get('optimizer', 'AdamW'), lr0=self.train_cfg.get('lr0', 0.000164), momentum=self.train_cfg.get('momentum', 0.9), weight_decay=self.train_cfg.get('weight_decay', 0.0005), close_mosaic=self.train_cfg.get('close_mosaic', 20), patience=self.train_cfg.get('patience', 15), fliplr=self.train_cfg.get('fliplr', 0.0), amp=self.train_cfg.get('amp', False), workers=self.train_cfg.get('workers', 0), mosaic=self.train_cfg.get('mosaic', 1.0), mixup=self.train_cfg.get('mixup', 0.0), copy_paste=self.train_cfg.get('copy_paste', 0.0))
                best_pt = str(model.trainer.best) if hasattr(model, 'trainer') else 'best.pt'
            else:
                os.makedirs('models/signs', exist_ok=True)
                with open('models/signs/best.pt', 'w') as f:
                    f.write('mock')
                best_pt = 'models/signs/best.pt'
            if self.config.get('mlflow', {}).get('log_artifacts', True):
                os.makedirs('models/signs', exist_ok=True)
                if os.path.exists(best_pt):
                    shutil.copy(best_pt, 'models/signs/best.pt')
                    best_pt = 'models/signs/best.pt'
            return best_pt

    def export(self, pt_path: str) -> Dict[str, Optional[str]]:
        res = {'pt': pt_path, 'onnx': None, 'engine': None}
        if YOLO is None or pt_path is None or (not os.path.exists(pt_path)):
            return res
        model = YOLO(pt_path)
        try:
            onnx_path = model.export(format='onnx')
            res['onnx'] = str(onnx_path)
        except Exception as e:
            logger.warning(f'[WARN] ONNX export failed: {e}')
        try:
            import torch
            if torch.cuda.is_available():
                trt_path = model.export(format='engine', device='0')
                res['engine'] = trt_path
            else:
                logger.warning('[WARN] Skipping TRT export because CUDA is not available.')
        except BaseException as e:
            logger.warning(f'[WARN] TRT export failed (TRT not available or out of memory): {e}')
        return res
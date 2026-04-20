try:
    import albumentations as A
except ImportError:
    A = None

class DashcamAugmentationPipeline:

    def __init__(self):
        if A is None:
            raise ImportError('albumentations is not installed.')
        self.transform = A.Compose([A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5), A.RandomGamma(gamma_limit=(80, 120), p=0.5), A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), A.GaussNoise(var_limit=(10.0, 50.0), p=0.5), A.MotionBlur(blur_limit=5, p=0.5), A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5)], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.35))

    def __call__(self, image, bboxes, class_labels):
        return self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
import os
import cv2
import numpy as np
import yaml
from tqdm import tqdm
import random

def generate_dataset(base_path='datasets/VietNamSigns', num_train=100, num_val=20):
    with open(os.path.join(base_path, 'data.yaml'), 'r') as f:
        data = yaml.safe_load(f)
    classes = list(data['names'].keys())
    for split, count in [('train', num_train), ('val', num_val)]:
        img_dir = os.path.join(base_path, split, 'images')
        lbl_dir = os.path.join(base_path, split, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        print(f'Generating {count} synthetic images for {split}...')
        for i in tqdm(range(count)):
            bg_color = [random.randint(50, 200) for _ in range(3)]
            img = np.ones((640, 640, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
            noise = np.random.randint(0, 50, (640, 640, 3), dtype=np.uint8)
            img = cv2.add(img, noise)
            num_signs = random.randint(1, 3)
            labels = []
            for _ in range(num_signs):
                cls = random.choice(classes)
                w = random.randint(50, 150)
                h = w
                x1 = random.randint(0, 640 - w)
                y1 = random.randint(0, 640 - h)
                x2, y2 = (x1 + w, y1 + h)
                sign_type = random.choice(['circle', 'triangle', 'square'])
                if sign_type == 'circle':
                    cv2.circle(img, (x1 + w // 2, y1 + h // 2), w // 2, (0, 0, 255), -1)
                    cv2.circle(img, (x1 + w // 2, y1 + h // 2), w // 2 - 5, (255, 255, 255), -1)
                elif sign_type == 'triangle':
                    pts = np.array([[x1 + w // 2, y1], [x1, y2], [x2, y2]], np.int32)
                    cv2.fillPoly(img, [pts], (0, 0, 255))
                    pts_inner = np.array([[x1 + w // 2, y1 + 10], [x1 + 10, y2 - 5], [x2 - 10, y2 - 5]], np.int32)
                    cv2.fillPoly(img, [pts_inner], (0, 255, 255))
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), -1)
                cv2.putText(img, str(cls), (x1 + w // 4, y1 + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cx_norm = (x1 + w / 2) / 640.0
                cy_norm = (y1 + h / 2) / 640.0
                w_norm = w / 640.0
                h_norm = h / 640.0
                labels.append(f'{cls} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}')
            img_name = f'synth_{i:04d}.jpg'
            lbl_name = f'synth_{i:04d}.txt'
            cv2.imwrite(os.path.join(img_dir, img_name), img)
            with open(os.path.join(lbl_dir, lbl_name), 'w') as f:
                f.write('\n'.join(labels))
if __name__ == '__main__':
    generate_dataset()
    print('Synthetic dataset successfully generated!')
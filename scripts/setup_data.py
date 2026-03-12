import os
import shutil

def setup_data():
    """"""
    config_path = 'configs/train_config.yaml'
    dataset_path = ''
    if os.path.exists(config_path):
        import yaml
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            dataset_path = config.get('dataset', {}).get('path', '')
            if dataset_path and os.path.exists(dataset_path):
                print(f'[✓] Configured dataset exists at: {dataset_path}')
                return
        except Exception as e:
            print(f'[WARN] Failed to read train_config.yaml: {e}')
    dataset_dir = 'datasets/vnts-merge-2'
    if os.path.exists(dataset_dir) and os.path.exists(os.path.join(dataset_dir, 'data.yaml')):
        print('[✓] Dataset already exists.')
        return
    print('Checking dataset directory...')
    os.makedirs(os.path.join(dataset_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'valid', 'images'), exist_ok=True)
    if not os.path.exists(os.path.join(dataset_dir, 'data.yaml')):
        print("[WARN] Dataset not fully populated. Please download and extract dataset to 'datasets/vnts-merge-2'.")
        valid_yaml = f'path: {os.path.abspath(dataset_dir)}\ntrain: train/images\nval: valid/images\nnames:\n  0: Sign'
        with open(os.path.join(dataset_dir, 'data.yaml'), 'w') as f:
            f.write(valid_yaml)
    else:
        print('[✓] Dataset setup complete.')
if __name__ == '__main__':
    setup_data()
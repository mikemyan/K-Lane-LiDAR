import os
GPUS_EN = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS_EN
import torch
from baseline.utils.config import Config
import torch.backends.cudnn as cudnn
import time
import shutil

time_now = time.localtime()
time_log = '%04d-%02d-%02d-%02d-%02d-%02d' % (time_now.tm_year, time_now.tm_mon, time_now.tm_mday, time_now.tm_hour, time_now.tm_min, time_now.tm_sec)

from baseline.utils.config import Config
from baseline.engine.runner import Runner

def main():
    path_config = './configs/Proj28_GFC-T3_RowRef_82_73.py' # ml_proj config
    ckpt_path = './configs/ml_curr_best.pth' # ml_proj checkpoint
    
    # path_config = './configs/baseline_config.py' # baseline config    
    # ckpt_path = './configs/ml_base.pth' # baseline checkpoint
    
    path_split = path_config.split('/')
    time_now = time.localtime()
    time_log = '%04d-%02d-%02d-%02d-%02d-%02d' % (time_now.tm_year, time_now.tm_mon, time_now.tm_mday, time_now.tm_hour, time_now.tm_min, time_now.tm_sec)
    
    cfg = Config.fromfile(path_config)
    cfg.log_dir = cfg.log_dir + '/' + time_log
    cfg.time_log = time_log
    os.makedirs(cfg.log_dir, exist_ok=True)
    shutil.copyfile(path_config, cfg.log_dir + '/' + path_split[-2] + '_' + path_split[-1].split('.')[0] + '.txt')
    cfg.work_dirs = cfg.log_dir + '/' + cfg.dataset.train.type
    cfg.gpus = len(GPUS_EN.split(','))

    cudnn.benchmark = True

    runner = Runner(cfg)

    if os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
        runner.load_ckpt(ckpt_path)
    else:
        print(f"Checkpoint not found at {ckpt_path}. Exiting.")
        return

    runner.train()

if __name__ == '__main__':
    main()

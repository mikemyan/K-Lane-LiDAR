import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..registry import HEADS

@HEADS.register_module
class LightweightConv2dHead(nn.Module):
    def __init__(self, in_channels=32, num_cls=6, **kwargs):
        super().__init__()
        self.num_cls = num_cls
        self.conf_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.cls_head = nn.Conv2d(in_channels, num_cls, kernel_size=1)

    def forward(self, x):
        conf = self.conf_head(x)
        cls = self.cls_head(x)
        return {'conf': conf, 'cls': cls}


    def loss(self, out, batch, loss_type=None):
        label = batch['label']
        conf_gt = (label != 255).float().unsqueeze(1)
        cls_gt = label.clone()
        cls_gt[cls_gt == 255] = 0
    
        conf_pred = out['conf']
        cls_pred = out['cls']
    
        conf_loss = F.binary_cross_entropy_with_logits(conf_pred, conf_gt)
        mask = (label != 255)
        cls_loss = F.cross_entropy(cls_pred, cls_gt.long(), reduction='none')
        cls_loss = (cls_loss * mask.float()).sum() / mask.float().sum()
    
        return {
            'conf_loss': conf_loss,
            'cls_loss': cls_loss,
            'loss': conf_loss + cls_loss
        }

    def get_conf_and_cls_dict(self, out, batch=None, **kwargs):
        conf = torch.sigmoid(out['conf'])
        cls = torch.softmax(out['cls'], dim=1)
    
        cls_idx = torch.argmax(cls, dim=1)
    
        return {
            'conf': conf,
            'cls': cls,
            'cls_idx': cls_idx
        }

    def get_lane_map_numpy_with_label(self, output, data=None, is_flip=True, is_img=False, is_get_1_stage_result=True):
        conf = output['conf']
        cls = output['cls']
        if conf.dim() == 4:
            conf = conf[:, 0]
        batch_size = conf.shape[0]
        num_cls = cls.shape[1]
    
        conf_pred_raw_list = []
        conf_pred_list = []
        cls_pred_raw_list = []
        cls_idx_list = []
        conf_label_list = []
        cls_label_list = []
        conf_by_cls_list = []
        conf_cls_idx_list = []
    
        for i in range(batch_size):
            # Per-sample processing
            conf_i = conf[i].cpu().numpy()
            cls_i = cls[i].cpu().numpy()
            cls_softmax_i = torch.softmax(cls[i], dim=0).cpu().numpy()
            conf_pred_raw = conf_i
            conf_pred = (conf_i > 0.5).astype('uint8')
            cls_pred_raw = cls_softmax_i
            cls_idx = np.argmax(cls_softmax_i, axis=0)
            conf_label = conf_pred
            cls_label = cls_idx
            conf_by_cls = (cls_idx != 6).astype('uint8')
            conf_cls_idx = cls_idx.copy()
    
            if is_flip:
                conf_pred_raw = np.flip(np.flip(conf_pred_raw, 0), 1)
                cls_pred_raw = np.flip(np.flip(cls_pred_raw, 1), 2)
    
            conf_pred_raw_list.append(conf_pred_raw)
            conf_pred_list.append(conf_pred)
            cls_pred_raw_list.append(cls_pred_raw)
            cls_idx_list.append(cls_idx)
            conf_label_list.append(conf_label)
            cls_label_list.append(cls_label)
            conf_by_cls_list.append(conf_by_cls)
            conf_cls_idx_list.append(conf_cls_idx)
    
        lane_maps = {
            'conf_label': conf_label_list,
            'cls_label': cls_label_list,
            'conf_pred_raw': conf_pred_raw_list,
            'cls_pred_raw': cls_pred_raw_list,
            'conf_pred': conf_pred_list,
            'conf_by_cls': conf_by_cls_list,
            'cls_idx': cls_idx_list,
            'conf_cls_idx': conf_cls_idx_list,
        }
        return lane_maps
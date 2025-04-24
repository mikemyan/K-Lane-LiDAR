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
        # x: [B, C, H, W]
        conf = self.conf_head(x)  # [B, 1, H, W]
        cls = self.cls_head(x)    # [B, num_cls, H, W]
        return {'conf': conf, 'cls': cls}

    # def loss(self, out, batch, loss_type=None):
    #     # Example: binary cross-entropy for conf, cross-entropy for cls
    #     conf_pred = out['conf']
    #     cls_pred = out['cls']
    #     conf_gt = batch['label'][:, 1:2, :, :]  # Adjust to your label format
    #     cls_gt = batch['label'][:, 0, :, :]     # Adjust to your label format

    #     conf_loss = F.binary_cross_entropy_with_logits(conf_pred, conf_gt.float())
    #     cls_loss = F.cross_entropy(cls_pred, cls_gt.long())
    #     return {'conf_loss': conf_loss, 'cls_loss': cls_loss, 'loss': conf_loss + cls_loss}

    def loss(self, out, batch, loss_type=None):
        # batch['label']: [B, H, W], values in [0, num_cls-1] or 255 for background
        label = batch['label']  # [B, H, W]
    
        # Confidence: 1 if not background, 0 if background
        conf_gt = (label != 255).float().unsqueeze(1)  # [B, 1, H, W]
        # Class: set background pixels to 0 (or any valid class, will be masked out)
        cls_gt = label.clone()
        cls_gt[cls_gt == 255] = 0  # [B, H, W]
    
        conf_pred = out['conf']  # [B, 1, H, W]
        cls_pred = out['cls']    # [B, num_cls, H, W]
    
        conf_loss = F.binary_cross_entropy_with_logits(conf_pred, conf_gt)
        # Mask out background pixels for class loss
        mask = (label != 255)
        cls_loss = F.cross_entropy(cls_pred, cls_gt.long(), reduction='none')  # [B, H, W]
        cls_loss = (cls_loss * mask.float()).sum() / mask.float().sum()
    
        return {
            'conf_loss': conf_loss,
            'cls_loss': cls_loss,
            'loss': conf_loss + cls_loss
        }

    def get_conf_and_cls_dict(self, out, batch=None, **kwargs):
        """
        Post-process the output dictionary to match expected format.
        Returns:
            dict with keys 'conf' and 'cls', both as torch tensors.
        """
        # Apply sigmoid to confidence and softmax to class logits
        conf = torch.sigmoid(out['conf'])  # [B, 1, H, W]
        cls = torch.softmax(out['cls'], dim=1)  # [B, num_cls, H, W]
    
        # Optionally, get predicted class index map
        cls_idx = torch.argmax(cls, dim=1)  # [B, H, W]
    
        # You can return whatever is expected by the rest of your pipeline
        return {
            'conf': conf,
            'cls': cls,
            'cls_idx': cls_idx
        }

    # def get_lane_map_numpy_with_label(self, output, data=None, is_flip=True, is_img=False, is_get_1_stage_result=True):
    #     conf = output['conf']  # [B, 1, H, W] or [B, H, W]
    #     cls = output['cls']    # [B, num_cls, H, W]
    #     if conf.dim() == 4:
    #         conf = conf[:, 0]  # [B, H, W]
    #     batch_size = conf.shape[0]
    #     num_cls = cls.shape[1]

    #     # Apply activations
    #     conf_pred_raw = conf.cpu().numpy()  # [B, H, W]
    #     conf_pred = (conf > 0.5).cpu().numpy().astype('uint8')  # [B, H, W]
    #     cls_softmax = torch.softmax(cls, dim=1).cpu().numpy()   # [B, num_cls, H, W]
    #     cls_pred_raw = cls_softmax  # [B, num_cls, H, W]
    #     cls_idx = np.argmax(cls_softmax, axis=1)  # [B, H, W]

    #     conf_label = conf_pred  # [B, H, W]
    #     cls_label = cls_idx     # [B, H, W]

    #     conf_by_cls = (cls_idx != (num_cls - 1)).astype('uint8')
    #     conf_cls_idx = cls_idx.copy()

    #     if is_flip:
    #         #conf_label = np.flip(np.flip(conf_label, axis=0),1)
    #         conf_pred_raw = np.flip(np.flip(conf_pred_raw, axis=0),1)
    #         #conf_pred = np.flip(np.flip(conf_pred, axis=0),1)
    #         cls_pred_raw = np.flip(np.flip(cls_pred_raw, axis=1),2)
    #         #cls_label = np.flip(np.flip(cls_label, axis=0),1)
    #         #cls_idx = np.flip(np.flip(cls_idx, axis=0),1)
    #         #conf_by_cls = np.flip(np.flip(conf_by_cls, axis=0),1)
    #         #conf_cls_idx = np.flip(np.flip(conf_cls_idx, axis=0),1)

    #     lane_maps = {
    #         'conf_label': [conf_label[i] for i in range(batch_size)],
    #         'cls_label': [cls_label[i] for i in range(batch_size)],
    #         'conf_pred_raw': [conf_pred_raw[i] for i in range(batch_size)],
    #         'cls_pred_raw': [cls_pred_raw[i] for i in range(batch_size)],
    #         'conf_pred': [conf_pred[i] for i in range(batch_size)],
    #         'conf_by_cls': [conf_by_cls[i] for i in range(batch_size)],
    #         'cls_idx': [cls_idx[i] for i in range(batch_size)],
    #         'conf_cls_idx': [conf_cls_idx[i] for i in range(batch_size)],
    #     }
    #     return lane_maps

    def get_lane_map_numpy_with_label(self, output, data=None, is_flip=True, is_img=False, is_get_1_stage_result=True):
        conf = output['conf']  # [B, 1, H, W] or [B, H, W]
        cls = output['cls']    # [B, num_cls, H, W]
        if conf.dim() == 4:
            conf = conf[:, 0]  # [B, H, W]
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
            conf_i = conf[i].cpu().numpy()  # [H, W]
            cls_i = cls[i].cpu().numpy()    # [num_cls, H, W]
            cls_softmax_i = torch.softmax(cls[i], dim=0).cpu().numpy()  # [num_cls, H, W]
            conf_pred_raw = conf_i
            conf_pred = (conf_i > 0.5).astype('uint8')
            cls_pred_raw = cls_softmax_i
            cls_idx = np.argmax(cls_softmax_i, axis=0)  # [H, W]
            conf_label = conf_pred
            cls_label = cls_idx
            conf_by_cls = (cls_idx != 6).astype('uint8')
            conf_cls_idx = cls_idx.copy()
    
            if is_flip:
                conf_pred_raw = np.flip(np.flip(conf_pred_raw, 0), 1)
                # conf_pred = np.flip(np.flip(conf_pred, 0), 1)
                cls_pred_raw = np.flip(np.flip(cls_pred_raw, 1), 2)
                # cls_idx = np.flip(np.flip(cls_idx, 0), 1)
                # conf_label = np.flip(np.flip(conf_label, 0), 1)
                # cls_label = np.flip(np.flip(cls_label, 0), 1)
                # conf_by_cls = np.flip(np.flip(conf_by_cls, 0), 1)
                # conf_cls_idx = np.flip(np.flip(conf_cls_idx, 0), 1)
    
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
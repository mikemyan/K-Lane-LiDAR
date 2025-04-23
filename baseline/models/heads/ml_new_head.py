import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def get_lane_map_numpy_with_label(self, output, data=None, is_flip=True, is_img=False, is_get_1_stage_result=True):
        """
        Minimal post-processing for lane map visualization.
        Args:
            output: dict containing 'conf' ([B, 1, H, W]) and 'cls' ([B, num_cls, H, W])
        Returns:
            dict with keys:
                'conf_label': (H, W) binary confidence mask
                'cls_label': (H, W) predicted class index map
                'conf_pred': (H, W) binary confidence mask
                'cls_pred': (H, W) predicted class index map
        """
        conf = output['conf']  # [B, 1, H, W] or [B, H, W]
        cls = output['cls']    # [B, num_cls, H, W]
        if conf.dim() == 4:
            conf = conf[:, 0]  # [B, H, W]
        conf_label = (conf > 0.5).cpu().numpy().astype('uint8')  # thresholded
    
        cls_label = torch.argmax(cls, dim=1).cpu().numpy()  # [B, H, W]
    
        # For simplicity, return the first batch item
        lane_maps = {
            'conf_label': conf_label[0],
            'cls_label': cls_label[0],
            'conf_pred': conf_label[0],
            'cls_pred': cls_label[0]
        }
        return lane_maps
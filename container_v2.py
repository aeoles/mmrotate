# Check MMRotate installation
import mmrotate
print(mmrotate.__version__)

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.dota import DOTADataset


@ROTATED_DATASETS.register_module()
class ContainerDataset2(DOTADataset):
    """container-crane and container for detection."""
    CLASSES = ('container-crane','container')

import mmcv
from mmcv import Config
cfg = Config.fromfile('/Users/bytedance/Documents/GitHub/mmrotate/configs/container_rotated_rcnn/container_v2_freeze_rpn_finetune_oriented_rcnn_r50_fpn_1x_dota_le90.py')


from mmdet.apis import set_random_seed

# Modify dataset type and path
cfg.dataset_type = 'ContainerDataset2'
cfg.data_root = '/Users/bytedance/Documents/GitHub/mmrotate/dataset/Dota_container_crane/'

cfg.data.test.type = 'ContainerDataset2'
cfg.data.test.data_root = '/Users/bytedance/Documents/GitHub/mmrotate/dataset/Dota_container_crane/val/'
cfg.data.test.ann_file = 'labelTxt'
cfg.data.test.img_prefix = 'image'

cfg.data.train.type = 'ContainerDataset2'
cfg.data.train.data_root = '/Users/bytedance/Documents/GitHub/mmrotate/dataset/Dota_container_crane/split_train'
cfg.data.train.ann_file = 'labelTxt'
cfg.data.train.img_prefix = 'image'

cfg.data.val.type = 'ContainerDataset2'
cfg.data.val.data_root = '/Users/bytedance/Documents/GitHub/mmrotate/dataset/Dota_container_crane/val'
cfg.data.val.ann_file = 'labelTxt'
cfg.data.val.img_prefix = 'image'

# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = 2
# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
cfg.load_from = 'oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'

# Set up working dir to save files and logs.
cfg.work_dir = '/Users/bytedance/Documents/GitHub/mmrotate/dataset/Dota_container_crane'

cfg.optimizer.lr = 0.001
cfg.lr_config.warmup = None
cfg.runner.max_epochs = 5
cfg.log_config.interval = 10

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'mAP'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 3
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 3

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
# cfg.device='cuda'
cfg.device ='cpu'

# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')


import os.path as osp

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

# Build dataset

datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)
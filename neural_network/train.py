import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from suctionnet.suctionnet_dataset import SuctionNetDataset
import ConvNet
import DeepLabV3Plus.network as network
from utils.avgmeter import AverageMeter


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='deeplabv3plus_resnet101', help='Model file name [default: votenet]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
parser.add_argument('--camera', default='realsense', help='Camera name. kinect or realsense. [default: realsense]')
parser.add_argument('--log_dir', default='/DATA2/Benchmark/suction/models/log_kinectV6', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--data_root', default='/DATA2/Benchmark/graspnet', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--label_root', default='/ssd1/hanwen/grasping/graspnet_label', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=10, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='20,40,60', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.7,0.7,0.7', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--finetune', action='store_true', help='finetune backbone network.')
FLAGS = parser.parse_args()


DATA_ROOT = FLAGS.data_root
LABEL_ROOT = FLAGS.label_root
BATCH_SIZE = FLAGS.batch_size
CAMERA = FLAGS.camera
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
LOG_DIR = FLAGS.log_dir
CHECKPOINT_PATH = FLAGS.checkpoint_path

BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]

# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(LOG_DIR) and FLAGS.overwrite:
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)'%(LOG_DIR))
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')
        os.system('rm -r %s'%(LOG_DIR))

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


TRAIN_DATASET = SuctionNetDataset(DATA_ROOT, LABEL_ROOT, camera=CAMERA, split='train', input_size=(480, 480))

print(len(TRAIN_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, drop_last=True, worker_init_fn=my_worker_init_fn)
print(len(TRAIN_DATALOADER))

# MODEL = importlib.import_module(FLAGS.model)
# TODO
model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
        'convnet_resnet101': ConvNet.convnet_resnet101,
        'deeplabv3plus_resnet101_depth': network.deeplabv3plus_resnet101_depth
    }
net = model_map[FLAGS.model](num_classes=FLAGS.num_classes, output_stride=FLAGS.output_stride, pretrained_backbone=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

EPOCH_CNT = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    print('Loading model from:')
    print(CHECKPOINT_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    EPOCH_CNT = checkpoint['epoch']

net.to(device)

criterion = nn.MSELoss()


if FLAGS.finetune:
    if 'deeplabv3' in FLAGS.model:
        conv1_params = list(map(id, net.module.backbone.conv1.parameters()))
        classifier_params = list(map(id, net.module.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in conv1_params + classifier_params,
                            net.parameters())
        optimizer = optim.Adam([
                {'params': base_params, 'lr': 0},
                {'params': net.module.backbone.conv1.parameters(), 'lr': BASE_LEARNING_RATE},
                {'params': net.module.classifier.parameters(), 'lr': BASE_LEARNING_RATE}], lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)
    elif 'convnet' in FLAGS.model:
        fuse_layer_params = list(map(id, net.module.fuselayers.parameters()))
        classifier_params = list(map(id, net.module.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in fuse_layer_params + classifier_params,
                            net.parameters())
        optimizer = optim.Adam([
                {'params': base_params, 'lr': 0},
                {'params': net.module.fuselayers.parameters(), 'lr': BASE_LEARNING_RATE},
                {'params': net.module.classifier.parameters(), 'lr': BASE_LEARNING_RATE}], lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)
    else:
        raise NotImplementedError('unrecognized model name')
else:
    optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)

# bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_one_epoch():
    losses = AverageMeter()
    score_losses = AverageMeter()
    center_losses = AverageMeter()

    adjust_learning_rate(optimizer, EPOCH_CNT)
    
    net.train()
    for batch_idx, (rgbs, depths, scores, wrenches, _) in enumerate(TRAIN_DATALOADER):
        optimizer.zero_grad()
        depths = torch.clamp(depths, 0, 1)
        if FLAGS.model == 'convnet_resnet101':
            depths = depths.unsqueeze(-1).repeat([1, 1, 1, 3])
            rgbds = torch.cat([rgbs, depths], dim=-1)
        elif 'depth' in FLAGS.model:
            rgbds = depths.unsqueeze(-1)
        else:
            rgbds = torch.cat([rgbs, depths.unsqueeze(-1)], dim=-1)
        
        rgbds = rgbds.permute(0, 3, 1, 2)
        rgbds = rgbds.to(device)
        scores = scores.to(device)
        wrenches = wrenches.to(device)
        
        pred = net(rgbds)
        
        score_loss = criterion(pred[:, 0, ...], scores)
        wrench_loss = criterion(pred[:, 1, ...], wrenches)
        loss = score_loss + wrench_loss

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), rgbs.size(0))
        score_losses.update(score_loss.item(), rgbs.size(0))
        center_losses.update(wrench_loss.item(), rgbs.size(0))
        
        if (batch_idx+1) % 10 == 0:
            log_string('Lr: {lr:.3e} | '
                        'Epoch: [{0}][{1}/{2}] | '
                        'Score Loss: {score_loss.val:.4f} ({score_loss.avg:.4f}) | '
                        'Center Loss: {center_loss.val:.4f} ({center_loss.avg:.4f}) | '
                        'Total Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                        EPOCH_CNT, batch_idx+1, len(TRAIN_DATALOADER), lr=get_current_lr(EPOCH_CNT),
                        score_loss=score_losses, center_loss=center_losses, loss=losses))
    


def train():
    global EPOCH_CNT

    for epoch in range(EPOCH_CNT, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** TRAIN EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f'%(get_current_lr(epoch)))
        train_one_epoch()

        if EPOCH_CNT % 10 == 0: # save every 10 epochs

            save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                        'optimizer_state_dict': optimizer.state_dict()}
            try: # with nn.DataParallel() the net is added as a submodule of DataParallel
                save_dict['model_state_dict'] = net.state_dict()
            except:
                save_dict['model_state_dict'] = net.state_dict()
            torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint_'+str(epoch)))
            


if __name__ == "__main__":
    train()
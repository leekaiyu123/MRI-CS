import math
import os
import pdb
import cv2
import time
import glob
import random
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.autograd import Variable
import gc
import numpy as np
import torch.nn.functional as F
import torch # PyTorch
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp # https://pytorch.org/docs/stable/notes/amp_examples.html
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, KFold # Sklearn
import albumentations as A # Augmentations
import timm
# import segmentation_models_pytorch as smp # smp
from timm.data.mixup import Mixup
from sklearn.metrics import confusion_matrix
import logging

os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'

# 数据增强：通过使用 Mixup 和 cutmix_alpha 参数，可以实现数据的混合增强。这可以提高模型的泛化能力和鲁棒性，减少过拟合的风险。
unique_labels = ["0", "1", "2"]
mixup_args = {'mixup_alpha': 1.,
             'cutmix_alpha': 1.,
             'prob': 0.2,
             'switch_prob': 0.2,
             'mode': 'batch',
             'label_smoothing': 0.05,  # 0.05 改为0.1
             'num_classes': 3}
mixup_fn = Mixup(**mixup_args)

# 定义了一个交叉熵损失函数 cross_entropy，使用 KLDivLoss 作为损失函数，计算输出和平滑标签之间的交叉熵损失。
def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)

# 日志记录：get_logger 函数用于创建日志记录器对象，并将日志同时输出到文件和控制台。日志记录对于深度学习实验的调试和结果分析非常重要，可以记录模型训练过程中的各种信息和指标。
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

# 定义了一个设置随机数种子的函数 set_seed，用于设置随机数种子，以确保实验的可复现性。
def set_seed(seed=42):
    random.seed(seed) # python
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 定义了一个名为 LabelSmoothingLoss 的自定义损失函数类。这个类用于实现标签平滑损失函数，可以在深度学习模型中用作损失函数进行训练。
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# 定义了一个名为 Focus 的自定义模块类。这个类用于实现特征聚焦操作，将输入的宽高信息聚焦到通道维度上，可以用于提取更具有代表性的特征。
class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1):
        super(Focus, self).__init__()
        self.conv = nn.Conv2d(c1 * 4, c2, k, 1)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


# 定义了一个名为 Net5 的神经网络模型类。该模型类包含一个预训练的backbone网络和一些附加的层。
class Net5(nn.Module):
    def __init__(self, backbone_name,n_classes = 3):
        super(Net5, self).__init__()

        self.n_classes = n_classes
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(self.backbone_name,pretrained=True,in_chans=3,global_pool = "avg",num_classes = 256)
        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.2)
        self.dropout_3 = nn.Dropout(0.3)
        self.dropout_4 = nn.Dropout(0.4)
        self.dropout_5 = nn.Dropout(0.5)
        self.head = nn.Linear(256, self.n_classes)

    def forward(self,x):
        x = self.backbone(x)
        x = (self.dropout_1(x) +
             self.dropout_2(x) +
             self.dropout_3(x) +
             self.dropout_4(x) +
             self.dropout_5(x)
             ) / 5
        y = self.head(x)

        return y

# 定义了训练数据和验证数据的数据增强操作
train_aug = A.Compose([
    # A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.2, rotate_limit=10, p=0.2),  #0.5
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(contrast_limit=0.4, p=0.5),  # 随机调整亮度和对比度

    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),   #mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)
    ToTensorV2(),
])

val_aug = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
    ToTensorV2(),
])

class lwhdataset(Dataset):
    def __init__(self,data_dir,train_transform,size,pad):
        self.pad = pad
        self.size = size
        self.label_dict = {"0": 0, "1": 1, "2": 2}
        self.c_paths = sorted(data_dir)
        self.transforms = train_transform

    def __getitem__(self, index):
        label = self.label_dict[self.c_paths[index].split('\\')[-2]]
        image = Image.open(self.c_paths[index]).convert("RGB")
        if self.pad:
            image = self.pading(self.size,image)
            image = np.array(image)
        else:
            image = np.array(image)
        image = self.transforms(image=image)['image']
        return image, label

    def __len__(self):
        if len(self.c_paths) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.c_paths)

    @staticmethod
    def get_id_labels(data_dir):
        image_fns = glob.glob(os.path.join(data_dir, '*'))
        label_names = [os.path.split(s)[-1] for s in image_fns]
        unique_labels = list(set(label_names))
        unique_labels.sort()
        id_labels = {_id: name for name, _id in enumerate(unique_labels)}
        return id_labels

    @staticmethod
    def pading(size,img):
        padding_v = tuple([125, 125, 125])
        w, h = img.size
        target_size = size
        interpolation = Image.BILINEAR
        if w > h:
            img = img.resize((int(target_size), int(h * target_size * 1.0 / w)), interpolation)
        else:
            img = img.resize((int(w * target_size * 1.0 / h), int(target_size)), interpolation)

        ret_img = Image.new("RGB", (target_size, target_size), padding_v)
        w, h = img.size
        st_w = int((ret_img.size[0] - w) / 2.0)
        st_h = int((ret_img.size[1] - h) / 2.0)
        ret_img.paste(img, (st_w, st_h))
        # ret_img = np.array(ret_img)
        return ret_img

class lwhdataset_swa(Dataset):
    def __init__(self,data_dir,train_transform,size,pad):
        self.pad = pad
        self.size = size
        self.label_dict = {"0": 0, "1": 1, "2": 2}  # 在 lwhdataset 中
        self.c_paths = sorted(data_dir)
        # print(f"Total samples in dataset: {len(self.c_paths)}")  # Check dataset size
        self.transforms = train_transform

    def __getitem__(self, index):
        label = self.label_dict[self.c_paths[index].split('\\')[-2]]
        image = Image.open(self.c_paths[index]).convert("RGB")
        if self.pad:
            image = self.pading(self.size,image)
            image = np.array(image)

        else:
            image = np.array(image)

        image = self.transforms(image=image)['image']

        return image.cuda()

    def __len__(self):
        if len(self.c_paths) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.c_paths)

    @staticmethod
    def get_id_labels(data_dir):
        image_fns = glob.glob(os.path.join(data_dir, '*'))
        label_names = [os.path.split(s)[-1] for s in image_fns]
        unique_labels = list(set(label_names))
        unique_labels.sort()
        id_labels = {_id: name for name, _id in enumerate(unique_labels)}
        return id_labels

    @staticmethod
    def pading(size,img):
        padding_v = tuple([125, 125, 125])
        w, h = img.size
        target_size = size
        interpolation = Image.BILINEAR
        if w > h:
            img = img.resize((int(target_size), int(h * target_size * 1.0 / w)), interpolation)
        else:
            img = img.resize((int(w * target_size * 1.0 / h), int(target_size)), interpolation)
        ret_img = Image.new("RGB", (target_size, target_size), padding_v)
        w, h = img.size
        st_w = int((ret_img.size[0] - w) / 2.0)
        st_h = int((ret_img.size[1] - h) / 2.0)
        ret_img.paste(img, (st_w, st_h))
        # ret_img = np.array(ret_img)
        return ret_img

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class CFG:
        # step1: hyper-parameter
        seed = 42
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #选择使用设备
        ckpt_fold = "model_repvgg_a0"   #保存模型路径
        ckpt_name = "repvgg_a0"  # for submit.
        train_dir = r"D:\\CS1"
        val_dir = r"D:\\CS2"
        # step2: data
        n_fold = 5    #交叉验证得次数
        img_size = [224, 224]   # 图片尺寸
        train_bs = 16 # bachsize
        valid_bs = train_bs
        log_interval = 20
        # step3: model
        backbone = 'repvgg_a0'
        num_classes = 3
        # step4: optimizer
        epoch = 60
        lr = 1e-3
        wd = 5e-2
        lr_drop = 8
        # step5: infer
        thr = 0.5

    set_seed(CFG.seed)
    # 创建保存模型得路径
    ckpt_path = f"./{CFG.ckpt_fold}/{CFG.ckpt_name}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # 加载未篡改得图片路径和篡改图片得路径并合并
    train_path = sorted(glob.glob(CFG.train_dir +"/*/*"))
    val_path = sorted(glob.glob(CFG.val_dir +"/*/*"))
    random.seed(CFG.seed)

    kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

    logger = get_logger(os.path.join(CFG.ckpt_fold, CFG.ckpt_name + '.log'))
    logger.info('Using: {}'.format(CFG.ckpt_name))

    model = timm.create_model(CFG.backbone,
                              pretrained=True,
                              num_classes=CFG.num_classes,global_pool = 'catevgmax' )

    model.to(CFG.device)
    swa_model = AveragedModel(model)

    train_data = lwhdataset(data_dir=train_path, train_transform=train_aug, size=CFG.img_size[0], pad=True)
    valid_data = lwhdataset(data_dir=val_path ,train_transform=val_aug, size=CFG.img_size[0], pad=True)
    valid_data_swa = lwhdataset_swa(data_dir=val_path ,train_transform=val_aug, size=CFG.img_size[0], pad=True)
    print(f"Total samples in training dataset: {len(train_data)}")
    print(f"Total samples in validation dataset: {len(valid_data)}")

    train_loader = DataLoader(dataset=train_data, batch_size=CFG.train_bs, shuffle=True, num_workers=10,drop_last = True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=CFG.valid_bs, shuffle=False, num_workers=8)
    print(f"Number of batches per epoch: {len(train_loader)}")  # Should be greater than 1
    valid_swa_loader = DataLoader(dataset=valid_data_swa, batch_size=CFG.train_bs, shuffle=False, num_workers=8)
    x = [1.0, 1.0, 1.0]
    weight = torch.Tensor(x).to("cuda:0")
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=CFG.wd)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    swa_start = 10
    swa_scheduler = SWALR(optimizer, anneal_strategy="linear", anneal_epochs=3, swa_lr=0.00001)
    scaler = amp.GradScaler(enabled=True)

    best_val_acc = 0
    best_trp_score = 0
    step = 0

    # for epoch in range(1+6, CFG.epoch + 1):
    for epoch in range(1, CFG.epoch + 1):
        os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'
        model.train()
        loss_mean = 0.
        correct = 0.
        total = 0.
        for i, (images, labels) in enumerate(train_loader):
            # print(f"Epoch {epoch}, Batch {i + 1}/{len(train_loader)}")  # Debug: Check batch processing
            images = images.to(CFG.device, dtype=torch.float)  # [b, c, w, h]
            labels = labels.to(CFG.device)

            optimizer.zero_grad()

            y_preds = model(images)

            loss = criterion(y_preds, labels)
            loss.backward()
            optimizer.step()
            # print(f"Processed Batch {i + 1}, Loss: {loss.item()}")  # Debug: Track loss and batch progress

            _, predicted = torch.max(y_preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().cpu().sum().numpy()

            loss_mean += loss.item()
            current_lr = optimizer.param_groups[0]['lr']

            if (i + 1) % CFG.log_interval == 0:
                loss_mean = loss_mean / CFG.log_interval
                logger.info("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} current_lr:{:.5f}".format(
                    epoch, CFG.epoch + 1, i + 1, len(train_loader), loss_mean, correct / total,current_lr))

                step += 1
                loss_mean = 0.
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        gc.collect()
        pres_list = []
        labels_list = []
        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        model.eval()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

                loss_val += loss.item()
                pres_list += predicted.cpu().numpy().tolist()
                labels_list += labels.data.cpu().numpy().tolist()

            loss_val_mean = loss_val / len(valid_loader)

            _, _, f_class, _ = precision_recall_fscore_support(y_true=labels_list, y_pred=pres_list,
                                                               labels=[id for id, name in enumerate(unique_labels)],
                                                               average=None)
            confusion = confusion_matrix(labels_list, pres_list, labels=None, sample_weight=None)
            # logger.info("confusion_matrix: ",confusion)
            fper_class = {name: "{:.2%}".format(f_class[_id]) for _id, name in enumerate(unique_labels)}
            logger.info('class_F1:{}  class_F1_average:{:.2%}'.format(fper_class, f_class.mean()))
            logger.info("Valid_acc:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} ".format(
                epoch, CFG.epoch + 1, j + 1, len(valid_loader), loss_val_mean, correct_val / total_val))

            if correct_val / total_val > best_val_acc:
                best_val_acc = correct_val / total_val
                save_path = f"{ckpt_path}/best_fold.pth"
                torch.save(model, save_path)

            logger.info("best_acc_score:\t best_acc:{:.2%}".format(best_val_acc))
        gc.collect()

    gc.collect()

    logger.info('stop training...')
import argparse          # 命令行参数解析
import datetime          # 时间计算（ETA/耗时）
import logging           # 日志记录
import os                # 路径/文件操作
import random            # 随机数
import sys

sys.path.append(".")     # 将项目根目录加入Python路径，确保能导入本地模块
import matplotlib.pyplot as plt  # 绘图（代码中未实际使用，可能是预留）
import numpy as np       # 数值计算
import torch             # PyTorch核心
import torch.distributed as dist  # 分布式训练核心模块（单卡需删除）
import torch.nn.functional as F  # 神经网络常用函数（损失/插值等）
from omegaconf import OmegaConf  # 配置文件解析（yaml→字典）
from torch.nn.parallel import DistributedDataParallel  # 分布式模型封装（单卡需删除）
from torch.utils.data import DataLoader  # 数据加载器
from torch.utils.data.distributed import DistributedSampler  # 分布式采样器（单卡需替换）
from torch.utils.tensorboard import SummaryWriter  # TensorBoard可视化
from tqdm import tqdm     # 进度条
# 本地自定义模块
from datasets import voc  # VOC数据集加载类
from utils.losses import DenseEnergyLoss, get_aff_loss, get_energy_loss  # 损失函数
from wetr.PAR import PAR  # 后处理模块（PAR迭代优化）
from utils import evaluate, imutils  # 评估指标/图像工具
from utils.AverageMeter import AverageMeter  # 指标平均值计算（如loss均值）
from utils.camutils import (cam_to_label, cams_to_affinity_label, ignore_img_box,
                            multi_scale_cam, multi_scale_cam_with_aff_mat,
                            propagte_aff_cam_with_bkg, refine_cams_with_bkg_v2,
                            refine_cams_with_cls_label)  # CAM相关工具（核心）
from utils.optimizer import PolyWarmupAdamW  # 自定义优化器（带warmup的AdamW）
from wetr.model_attn_aff import WeTr  # 核心模型WeTr

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/voc.yaml',
                    type=str,
                    help="config")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method") #最大池化
parser.add_argument("--seg_detach", action="store_true", help="detach seg") # 是否分离seg分支梯度 
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")  # 工作目录（保存日志/模型）
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")   # 分布式进程编号（单卡需删除）
parser.add_argument("--radius", default=8, type=int, help="radius")  # 注意力掩码半径
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")  # 图像裁剪尺寸

parser.add_argument("--high_thre", default=0.55, type=float, help="high_bkg_score") # CAM高阈值
parser.add_argument("--low_thre", default=0.35, type=float, help="low_bkg_score")  # CAM低阈值

parser.add_argument('--backend', default='nccl') # 分布式通信后端（单卡需删除）

#固定随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
#配置日志系统
def setup_logger(filename='test.log'):
    # 定义日志格式：时间 - 文件名 - 日志级别: 内容
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s') 
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 文件处理器：将日志写入指定文件
    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)
    # 控制台处理器：将日志打印到终端
    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)
    
#计算训练耗时和剩余时间
def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    #time_now = datetime.datetime.strptime(time_now.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')
    # 计算剩余迭代数/已完成迭代数 → 时间缩放比例
    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)# 已耗时
    eta = (delta*scale)# 剩余时间（预估）
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)  # 返回已耗时、剩余时间（字符串格式）

#计算下采样后的尺寸
def get_down_size(ori_shape=(512,512), stride=16):
    h, w = ori_shape
    # 计算经过stride=16下采样后的尺寸（向上取整逻辑）
    _h = h // stride + 1 - ((h % stride) == 0)
    _w = w // stride + 1 - ((w % stride) == 0)
    return _h, _w

#验证函数（评估模型精度）
def validate(model=None, data_loader=None, cfg=None):
 # 初始化存储列表：预测结果、真实标签、CAM结果、Affinity CAM结果
    preds, gts, cams, aff_gts = [], [], [], []
    model.eval() # 模型切换到评估模式（关闭Dropout/BatchNorm更新）
    avg_meter = AverageMeter()  # 计算分类F1分数均值
    with torch.no_grad(): # 禁用梯度计算，节省显存/加速
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            # 加载验证数据：图像名、输入图像、标签、分类标签                
            name, inputs, labels, cls_label = data
            # 数据移到GPU
            inputs = inputs.cuda()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda()
            # 前向推理：模型输出分类结果、分割结果、占位符、注意力预测
            cls, segs, _, attn_pred = model(inputs,)
            # 计算分类F1分数（多标签分类）
            cls_pred = (cls>0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})
            # 分割结果上采样到标签尺寸（双线性插值）
            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            #### 多尺度CAM计算（提升CAM精度）
            _cams = multi_scale_cam(model, inputs, cfg.cam.scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_to_label(resized_cam, cls_label, cfg=cfg)
            #计算下采样尺寸，生成注意力掩码
            H, W = get_down_size(ori_shape=(h, w))
            infer_mask = get_mask_by_radius(h=H, w=W, radius=args.radius)
            #CAM下采样到注意力图尺寸
            valid_cam_resized = F.interpolate(resized_cam, size=(H,W), mode='bilinear', align_corners=False)
            #基于Affinity传播CAM（带背景分数）
            aff_cam = propagte_aff_cam_with_bkg(valid_cam_resized, aff=attn_pred, mask=infer_mask, cls_labels=cls_label, bkg_score=0.35)
            aff_cam = F.interpolate(aff_cam, size=labels.shape[1:], mode="bilinear", align_corners=False)
            aff_label = aff_cam.argmax(dim=1)
            #infer_path_index = irnutils.PathIndex(radius=5, default_size=(edge.shape[2], edge.shape[3]))
            #irn_cams = irnutils.batch_propagate_edge(cams=resized_cam, edge=edge.detach(), cls_label=cls_label, path_index=infer_path_index)
            #irn_label = cam_to_label_irn(irn_cams.detach(), cls_label=cls_label, ignore_mid=False, cfg=cfg)
            ###
            # 收集结果（转CPU+numpy，避免GPU显存占用）
            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            aff_gts += list(aff_label.cpu().numpy().astype(np.int16))
            # （可选）保存CAM结果
            valid_label = torch.nonzero(cls_label[0])[:,0]
            out_cam = torch.squeeze(resized_cam)[valid_label]
            #np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

    # 计算各类精度指标
    cls_score = avg_meter.pop('cls_score')  # 分类F1分数
    seg_score = evaluate.scores(gts, preds, num_classes=cfg.dataset.num_classes)  # 分割精度
    cam_score = evaluate.scores(gts, cams, num_classes=cfg.dataset.num_classes)  # CAM精度
    aff_score = evaluate.scores(gts, aff_gts, num_classes=cfg.dataset.num_classes)  # Affinity CAM精度
    model.train()  # 模型切回训练模式
    return cls_score, seg_score, cam_score, aff_score

def get_seg_loss(pred, label, ignore_index=255):
    # 背景损失：只计算背景（label=0），其他设为ignore_index
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    #前景损失：只计算前景（label≠0），背景设为ignore_index
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5# 背景+前景损失平均

def get_mask_by_radius(h=20, w=20, radius=8):
    #生成注意力掩码（半径限制）
    hw = h * w # 特征图总像素数
    #_hw = (h + max(dilations)) * (w + max(dilations)) 
    mask  = np.zeros((hw, hw))
    for i in range(hw):
       # 计算像素i的坐标
        _h = i // w
        _w = i % w
        
        # 计算以(i,j)为中心、radius为半径的邻域范围
        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        # 邻域内的像素标记为1（表示可通信/计算）
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1  # 对称矩阵

    return mask

def train(cfg):

    num_workers = 10 # 数据加载线程数

    # 分布式核心：设置当前进程的GPU设备 + 初始化分布式进程组
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend,)
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0) # 记录训练开始时间（用于计算ETA）
    
    # 训练数据集：VOC12ClsDataset（分类任务数据集，带数据增强）
    train_dataset = voc.VOC12ClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split, # 训练集划分（如train_aug）
        stage='train',
        aug=True, # 开启数据增强
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,  # 水平翻转
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    # 验证数据集：VOC12SegDataset（分割任务数据集，无增强）
    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='val',
        aug=False, # 关闭数据增强
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    # 分布式采样器：保证多卡间数据不重复
    train_sampler = DistributedSampler(train_dataset,shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,# 单卡批次大小
                              #shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,# 丢弃最后一个不完整批次
                              sampler=train_sampler,# 丢弃最后一个不完整批次
                              prefetch_factor=4)# 丢弃最后一个不完整批次
    # 验证加载器：无分布式，batch_size=1
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)
    # 设置当前设备（分布式进程对应的GPU）
    device = torch.device(args.local_rank)
    # 初始化WeTr模型（核心模型）
    wetr = WeTr(backbone=cfg.backbone.config,# 骨干网络（如mit_b1）
                stride=cfg.backbone.stride,  # 下采样步长
                num_classes=cfg.dataset.num_classes,
                embedding_dim=256,              # 嵌入维度
                pretrained=True,                # 加载预训练权重
                pooling=args.pooling,)           # 加载预训练权重
    logging.info('\nNetwork config: \n%s'%(wetr))# 打印模型结构
    param_groups = wetr.get_param_groups()       # 获取模型参数组（用于不同学习率）
    # 初始化PAR模块（后处理优化CAM）
    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24])
    # 模型移到指定GPU
    wetr.to(device)
    par.to(device)
    
    # 计算模型输出特征图尺寸（基于裁剪尺寸和步长）
    mask_size = int(cfg.dataset.crop_size // 16)
    infer_size = int((cfg.dataset.crop_size * max(cfg.cam.scales)) // 16)
    # 生成训练/推理用的注意力掩码
    attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=args.radius)
    attn_mask_infer = get_mask_by_radius(h=infer_size, w=infer_size, radius=args.radius)
    # 仅主进程（local_rank=0）初始化TensorBoard（避免多进程重复写入）
    if args.local_rank==0:
        writer = SummaryWriter(cfg.work_dir.tb_logger_dir)
        dummy_input = torch.rand(1, 3, 384, 384).cuda(0)
        #writer.add_graph(wetr, dummy_input)
    
    optimizer = PolyWarmupAdamW(
        params=[
            # 参数组1：普通参数（基础学习率）
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            # 参数组2：归一化层（冻结，lr=0）
            {
                "params": param_groups[1],
                "lr": 0.0, ## freeze norm layers
                "weight_decay": 0.0,
            },
            # 参数组3：特定层（10倍学习率）
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
             # 参数组4：特定层（10倍学习率）
            {
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter, # warmup迭代数
        max_iter = cfg.train.max_iters,           # 最大迭代数
        warmup_ratio = cfg.scheduler.warmup_ratio, # warmup学习率比例
        power = cfg.scheduler.power               # 学习率衰减指数
    )
    logging.info('\nOptimizer: \n%s' % optimizer)
    #将模型封装为分布式模型（DDP），实现多卡梯度同步
    wetr = DistributedDataParallel(wetr, device_ids=[args.local_rank], find_unused_parameters=True)
    
    #训练循环初始化
    # loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
    # 分布式采样器设置随机种子（保证多卡shuffle一致）
    train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))
    train_loader_iter = iter(train_loader)# 数据迭代器
    #for n_iter in tqdm(range(cfg.train.max_iters), total=cfg.train.max_iters, dynamic_ncols=True):
    avg_meter = AverageMeter() # 损失均值计算
    bkg_cls = torch.ones(size=(cfg.train.samples_per_gpu, 1)) # 背景分类标签（全1）
    
    #核心训练循环
    for n_iter in range(cfg.train.max_iters):
         # ---------------------- 1. 加载批次数据 ----------------------
        try:
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        except:
            # 数据迭代完后，重新初始化迭代器（分布式需重置sampler种子）
            train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
         # 数据移到GPU（非阻塞传输，提升速度）
        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())# 图像反归一化（用于后处理）
        cls_labels = cls_labels.to(device, non_blocking=True)
        
        # ---------------------- 2. 模型前向推理 ----------------------
        cls, segs, attns, attn_pred = wetr(inputs, seg_detach=args.seg_detach)
        
        # ---------------------- 3. CAM生成与优化 ----------------------
        # 多尺度CAM + Affinity矩阵计算
        cams, aff_mat = multi_scale_cam_with_aff_mat(wetr, inputs=inputs, scales=cfg.cam.scales)
        # CAM转伪标签（忽略中间值，基于分类标签）
        valid_cam, pseudo_label = cam_to_label(cams.detach(), cls_label=cls_labels, img_box=img_box, ignore_mid=True, cfg=cfg)

        ## CAM下采样到推理尺寸
        valid_cam_resized = F.interpolate(valid_cam, size=(infer_size, infer_size), mode='bilinear', align_corners=False)
        # CAM下采样到推理尺寸
        aff_cam_l = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=cls_labels, bkg_score=cfg.cam.low_thre)
        aff_cam_l = F.interpolate(aff_cam_l, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        aff_cam_h = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=cls_labels, bkg_score=cfg.cam.high_thre)
        aff_cam_h = F.interpolate(aff_cam_h, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        # 拼接背景分类标签
        bkg_cls = bkg_cls.to(cams.device)
        _cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)
        # PAR优化Affinity CAM（低/高阈值）
        refined_aff_cam_l = refine_cams_with_cls_label(par, inputs_denorm, cams=aff_cam_l, labels=_cls_labels, img_box=img_box)
        refined_aff_label_l = refined_aff_cam_l.argmax(dim=1)
        refined_aff_cam_h = refine_cams_with_cls_label(par, inputs_denorm, cams=aff_cam_h, labels=_cls_labels, img_box=img_box)
        refined_aff_label_h = refined_aff_cam_h.argmax(dim=1)
        # 处理Affinity CAM（去掉背景通道）
        aff_cam = aff_cam_l[:,1:]
        refined_aff_cam = refined_aff_cam_l[:,1:,]
        refined_aff_label = refined_aff_label_h.clone()
        # 忽略背景/无效区域（设为ignore_index）
        refined_aff_label[refined_aff_label_h == 0] = cfg.dataset.ignore_index
        refined_aff_label[(refined_aff_label_h + refined_aff_label_l) == 0] = 0
        refined_aff_label = ignore_img_box(refined_aff_label, img_box=img_box, ignore_index=cfg.dataset.ignore_index)
        #PAR优化原始CAM（带背景）
        refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=cams, cls_labels=cls_labels, cfg=cfg, img_box=img_box)
        # ---------------------- 4. 损失计算 ----------------------
        # Affinity损失（注意力预测 vs Affinity标签）
        aff_label = cams_to_affinity_label(refined_pseudo_label, mask=attn_mask, ignore_index=cfg.dataset.ignore_index)
        aff_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label)

        # 分割结果上采样到伪标签尺寸
        segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        # 前6000迭代用原始伪标签，之后用优化后的Affinity标签
        if n_iter <= 6000:
            refined_aff_label = refined_pseudo_label

        # 分割损失（前景+背景）
        seg_loss = get_seg_loss(segs, refined_aff_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        # reg_loss = get_energy_loss(img=inputs, logit=segs, label=refined_aff_label, img_box=img_box, loss_layer=loss_layer)
        #seg_loss = F.cross_entropy(segs, pseudo_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        # 分类损失（多标签软边际损失）
        cls_loss = F.multilabel_soft_margin_loss(cls, cls_labels)
        # 损失加权（分阶段：前期只训分类，后期加分割+Affinity）
        if n_iter <= cfg.train.cam_iters:
            loss = 1.0 * cls_loss + 0.0 * seg_loss + 0.0 * aff_loss# + 0.0 * reg_loss
        else: 
            loss = 1.0 * cls_loss + 0.1 * seg_loss + 0.1 * aff_loss# + 0.01 * reg_loss

        # 记录损失均值
        avg_meter.add({'cls_loss': cls_loss.item(), 'seg_loss': seg_loss.item(), 'aff_loss': aff_loss.item()})

            # ---------------------- 5. 反向传播与优化 ----------------------
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数
    
        # ---------------------- 6. 日志记录（每log_iters迭代） ----------------------    
        if (n_iter+1) % cfg.train.log_iters == 0:
            # 计算耗时/剩余时间
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            # 计算分割准确率（伪标签 vs 预测）
            preds = torch.argmax(segs,dim=1,).cpu().numpy().astype(np.int16)
            gts = pseudo_label.cpu().numpy().astype(np.int16)
            refined_gts = refined_pseudo_label.cpu().numpy().astype(np.int16)
            aff_gts = refined_aff_label.cpu().numpy().astype(np.int16)
            seg_mAcc = (preds==gts).sum()/preds.size

            # 生成TensorBoard可视化图像（输入、CAM、Affinity CAM、注意力图等）
            grid_imgs, grid_cam = imutils.tensorboard_image(imgs=inputs.clone(), cam=valid_cam)
            _, grid_aff_cam = imutils.tensorboard_image(imgs=inputs.clone(), cam=aff_cam)
            _, grid_ref_aff_cam = imutils.tensorboard_image(imgs=inputs.clone(), cam=refined_aff_cam)

            _attns_detach = [a.detach() for a in attns]
            _attns_detach.append(attn_pred.detach())
            #_, grid_ref_cam = imutils.tensorboard_image(imgs=inputs.clone(), cam=refined_valid_cam)
            #grid_attns_0 = imutils.tensorboard_attn(attns=_attns_detach, n_pix=0, n_row=cfg.train.samples_per_gpu)
            #grid_attns_1 = imutils.tensorboard_attn(attns=_attns_detach, n_pix=0.3, n_row=cfg.train.samples_per_gpu)
            #grid_attns_2 = imutils.tensorboard_attn(attns=_attns_detach, n_pix=0.6, n_row=cfg.train.samples_per_gpu)
            #grid_attns_3 = imutils.tensorboard_attn(attns=_attns_detach, n_pix=0.9, n_row=cfg.train.samples_per_gpu)
            grid_attns = imutils.tensorboard_attn2(attns=_attns_detach, n_row=cfg.train.samples_per_gpu)

            grid_labels = imutils.tensorboard_label(labels=gts)
            grid_preds = imutils.tensorboard_label(labels=preds)
            grid_refined_gt = imutils.tensorboard_label(labels=refined_gts)
            grid_aff_gt = imutils.tensorboard_label(labels=aff_gts)
            #grid_irn_gt = imutils.tensorboard_label(labels=irn_gts)

            # 仅主进程打印日志+写入TensorBoard
            if args.local_rank==0:
                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, aff_loss: %.4f, pseudo_seg_loss %.4f, pseudo_seg_mAcc: %.4f"%(n_iter+1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('aff_loss'), avg_meter.pop('seg_loss'), seg_mAcc))

                writer.add_image("train/images", grid_imgs, global_step=n_iter)
                writer.add_image("train/preds", grid_preds, global_step=n_iter)
                writer.add_image("train/pseudo_gts", grid_labels, global_step=n_iter)
                writer.add_image("train/pseudo_ref_gts", grid_refined_gt, global_step=n_iter)
                writer.add_image("train/aff_gts", grid_aff_gt, global_step=n_iter)
                #writer.add_image("train/pseudo_irn_gts", grid_irn_gt, global_step=n_iter)
                writer.add_image("cam/valid_cams", grid_cam, global_step=n_iter)
                writer.add_image("cam/aff_cams", grid_aff_cam, global_step=n_iter)
                writer.add_image("cam/refined_aff_cams", grid_ref_aff_cam, global_step=n_iter)

                writer.add_image("attns/top_stages_case0", grid_attns[0], global_step=n_iter)
                writer.add_image("attns/top_stages_case1", grid_attns[1], global_step=n_iter)
                writer.add_image("attns/top_stages_case2", grid_attns[2], global_step=n_iter)
                writer.add_image("attns/top_stages_case3", grid_attns[3], global_step=n_iter)

                writer.add_image("attns/last_stage_case0", grid_attns[4], global_step=n_iter)
                writer.add_image("attns/last_stage_case1", grid_attns[5], global_step=n_iter)
                writer.add_image("attns/last_stage_case2", grid_attns[6], global_step=n_iter)
                writer.add_image("attns/last_stage_case3", grid_attns[7], global_step=n_iter)

                writer.add_scalars('train/loss', {"seg_loss": seg_loss.item(), "cls_loss": cls_loss.item()}, global_step=n_iter)
                writer.add_scalar('count/pos_count', pos_count.item(), global_step=n_iter)
                writer.add_scalar('count/neg_count', neg_count.item(), global_step=n_iter)
                
        # ---------------------- 7. 验证与模型保存（每eval_iters迭代） ----------------------
        if (n_iter+1) % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "wetr_iter_%d.pth"%(n_iter+1))
            if args.local_rank==0:
                logging.info('Validating...')
                torch.save(wetr.state_dict(), ckpt_name)# 保存模型权重
            cls_score, seg_score, cam_score, aff_score = validate(model=wetr, data_loader=val_loader, cfg=cfg)
            if args.local_rank==0:
                logging.info("val cls score: %.6f"%(cls_score))
                logging.info("cams score:")
                logging.info(cam_score)
                logging.info("aff cams score:")
                logging.info(aff_score)
                logging.info("segs score:")
                logging.info(seg_score)

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.crop_size = args.crop_size

    cfg.cam.high_thre = args.high_thre
    cfg.cam.low_thre = args.low_thre

    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir, timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)

    if args.local_rank == 0:
        setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp+'.log'))
        logging.info('\nargs: %s' % args)
        logging.info('\nconfigs: %s' % cfg)
    
    ## fix random seed
    setup_seed(1)
    train(cfg=cfg)

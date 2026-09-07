import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn 
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import tensorboard_logger as tb_logger
import numpy as np
from model import ProtPLL
from resnet import *
from utils.utils_algo import *
from utils.utils_loss import partial_loss, SupConLoss, partial_loss1, partial_loss2
from torch.nn import MSELoss
from utils.cifar10_pll_com_dataset import load_cifar10, generate_dataloader_partial, generate_dataloader_comple 


torch.set_printoptions(precision=2, sci_mode=False)

parser = argparse.ArgumentParser(description='PyTorch implementation of ICLR 2022 Oral paper PiCO')
parser.add_argument('--dataset', default='cifar10', type=str, 
                    choices=['cifar10', 'cifar100', 'cub200', 'fashion'],
                    help='dataset name (cifar10)')
parser.add_argument('--exp-dir', default='experiment/Prot_PLL', type=str,
                    help='experiment directory for saving checkpoints and logs')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=['resnet18','resnet18_mnist'],
                    help='network architecture (only resnet18 used in PiCO)')
parser.add_argument('-j', '--workers', default=32, type=int,
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=500, type=int, 
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-lr_decay_epochs', type=str, default='50,100,200',
                    help='where to decay lr, can be a list')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=True,
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--num-class', default=10, type=int,
                    help='number of class')
parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')
parser.add_argument('--proto_m', default=0.99, type=float,
                    help='momentum for computing the moving average of prototypes')
parser.add_argument('--conf_ema_range', default='0.95,0.8', type=str,
                    help='pseudo target updating coefficient (phi)')
parser.add_argument('--prot_start', default=1, type=int,
                    help='Start Prototype Updating')
parser.add_argument('--partial_rate', default=0.5, type=float,
                    help='ambiguity level (q)')
parser.add_argument('--hierarchical', action='store_true', 
                    help='for CIFAR-100 fine-grained training')

def main():
    mp.set_start_method('spawn')
    args = parser.parse_args()
    print(args)
    args.conf_ema_range = [float(item) for item in args.conf_ema_range.split(',')]
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    print(args)

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    model_path = 'ds_{ds}_pr_{pr}_lr_{lr}_ep_{ep}_ps_{ps}_pm_{pm}_arch_{arch}_heir_{heir}_sd_{seed}'.format(  #本来的
                                            ds=args.dataset,
                                            pr=args.partial_rate,
                                            lr=args.lr,
                                            ep=args.epochs,
                                            ps=args.prot_start,
                                            pm=args.proto_m,
                                            arch=args.arch, #本来的
                                            seed=args.seed,
                                            heir=args.hierarchical)
    args.exp_dir = os.path.join(args.exp_dir, model_path)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    
    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    cudnn.benchmark = True
    args.gpu = gpu
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print("=> creating model '{}'".format(args.arch))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.dataset == 'cub200':
        input_size = 224  # fixed as 224
        train_loader, train_givenY, train_sampler, test_loader = load_cub200(input_size=input_size
                                                                             , partial_rate=args.partial_rate,
                                                                             batch_size=args.batch_size)
    elif args.dataset == 'cifar10':
         data_train1, labels_train1, train_givenY1, partial_matrix_dataset_train, data_train2, labels_train2, \
         compleY_train, comple_matrix_dataset_train, train_loader1, train_loader2, test_loader, complelabels = load_cifar10(
            partial_rate=args.partial_rate, batch_size=args.batch_size)

    elif args.dataset == 'cifar100':
        train_loader, train_givenY, train_sampler, test_loader = load_cifar100(partial_rate=args.partial_rate,
                                                                               batch_size=args.batch_size,
                                                                               hierarchical=args.hierarchical)
    else:
        raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")

    print('Calculating uniform targets...')

    if args.gpu == 0:
        logger = tb_logger.Logger(logdir=os.path.join(args.exp_dir, 'tensorboard'), flush_secs=2)
    else:
        logger = None
    print('\nStart Training\n')
    best_acc = 0

    """ Train """
    iteration = 0
    train_label1thresh = labels_train1
    train_image1thresh = data_train1
    train_givenY1thresh = train_givenY1
    train_loader2thresh = train_loader2
    train_label2thresh = labels_train2

    while iteration < 10 and len(train_label2thresh) > 0:
        print('iteration:', iteration)
        model = ProtPLL(args, SupConResNet)

        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        tempY = train_givenY1thresh.sum(dim=1).unsqueeze(1).repeat(1, train_givenY1thresh.shape[1])
        confidence = train_givenY1thresh.float() / tempY
        confidence = confidence.cuda()

        train_givenY1thresh = train_givenY1thresh.cuda()

        loss_fn1 = partial_loss1(confidence)
        loss_fn2 = partial_loss2(confidence)

        train_loader1thresh = generate_dataloader_partial(train_image1thresh, train_label1thresh,
                                                          train_givenY1thresh.cuda(),
                                                          args.batch_size)

        for epoch in range(args.start_epoch, args.epochs):
            is_best = False
            start_upd_prot = epoch>=args.prot_start
            adjust_learning_rate(args, optimizer, epoch)
            train(train_loader1thresh, train_givenY1thresh, model, loss_fn1, loss_fn2, optimizer, epoch, args, logger, start_upd_prot)#训练
            loss_fn1.set_conf_ema_m(epoch, args)
            loss_fn2.set_conf_ema_m(epoch, args)
            acc_test, _ = test(model, test_loader, args, epoch, logger)
            mmc = loss_fn1.confidence.max(dim=1)[0].mean()

            with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
                f.write('Epoch {}: Acc {}, Best Acc {}. (lr {}, MMC {})\n'.format(epoch
                   , acc_test, best_acc, optimizer.param_groups[0]['lr'], mmc))
            if acc_test > best_acc:
               best_acc = acc_test
               is_best = True

            save_checkpoint({'epoch': epoch + 1,
                         'arch': args.arch,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         }, is_best=is_best, filename='{}/checkpoint.pth.tar'.format(args.exp_dir),
                        best_file_name='{}/checkpoint_best.pth.tar'.format(args.exp_dir))

        threshold = 0.99

        partial_labels_right, images_right, labels_right, partial_labels_wrong, images_wrong, labels_wrong, \
        complementary_lathresh = accuracy_comple(args, train_loader2thresh, model, threshold)

        partial_labels_right = partial_labels_right[1:, :]
        images_right = images_right[1:, :, :, :]
        labels_right = labels_right[1:]

        train_image1thresh = np.concatenate([train_image1thresh, images_right], axis=0) #numpy unit8  图像一定要保持unit8形式，不然第2个iteration的结果就变了
        train_label1thresh = torch.cat([train_label1thresh, labels_right], dim=0).long() #tensor
        train_givenY1thresh = torch.cat([train_givenY1thresh, partial_labels_right.cuda()], dim=0) #tensor

        partial_labels_wrong = partial_labels_wrong[1:, :] 
        complementary_lathresh = complementary_lathresh[1:].long()
        images_wrong = images_wrong[1:, :, :, :]
        labels_wrong = labels_wrong[1:].long()
        train_loader2thresh = generate_dataloader_comple(images_wrong, labels_wrong, partial_labels_wrong,
                                                         complementary_lathresh, args.batch_size)  # 产生新的compleloader
        train_label2thresh = labels_wrong
        iteration += 1


def train(train_loader, train_givenY, model, loss_fn1, loss_fn2, optimizer, epoch, args, tb_logger, start_upd_prot=False):
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('TrainAcc@Cls', ':2.2f')
    acc_proto = AverageMeter('TrainAcc@Proto', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, acc_cls, acc_proto],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    end = time.time()
    for i, (_, images_w,  labels, true_labels, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        X_w, Y, index = images_w.cuda(), labels.cuda(), index.cuda()
        Y_true = true_labels.long().detach().cuda()

        cls_out, score_prot, prototypes1, q1,  score1 = model(
            X_w, Y, args)

        if start_upd_prot:
            loss_fn1.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=Y)

        loss_cls1 = loss_fn1(cls_out, index, train_givenY)
        loss = loss_cls1

        acc = accuracy(cls_out, Y_true)[0]
        acc_cls.update(acc[0])
        acc = accuracy(score_prot, Y_true)[0] 
        acc_proto.update(acc[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)

    if args.gpu == 0:
        tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
        tb_logger.log_value('Prototype Acc', acc_proto.avg, epoch)


def test(model, test_loader, args, epoch, tb_logger):
    with torch.no_grad():
        print('==> Evaluation...')       
        model.eval()    
        top1_acc = AverageMeter("Top1")
        top5_acc = AverageMeter("Top5")

        for batch_idx, (images, labels) in enumerate(test_loader):
            X,  labels = images.cuda(), labels.cuda()
            outputs = model(X, args, eval_only=True)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1_acc.update(acc1[0])
            top5_acc.update(acc5[0])

        acc_tensors = torch.Tensor([top1_acc.avg,top5_acc.avg]).cuda(args.gpu)
        
        print('Test Accuracy is %.2f%% (%.2f%%)'%(acc_tensors[0],acc_tensors[1]))
        if args.gpu ==0:
            tb_logger.log_value('Top1 Acc', acc_tensors[0], epoch)
            tb_logger.log_value('Top5 Acc', acc_tensors[1], epoch)             
    return acc_tensors[0], outputs

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)

if __name__ == '__main__':
    main()

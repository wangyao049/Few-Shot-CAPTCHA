import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.protonet import ProtoNet
from methods.protonet_dw import ProtoNetDW
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file


def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):
    #     print(stop_epoch)

    if optimization == 'Adam':
        print(model.parameters())
        optimizer = torch.optim.Adam(model.parameters())
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0

    for epoch in range(start_epoch, stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader, optimizer)  # model are called by reference, no need to return
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        # validation
        acc = model.test_loop(val_loader)
        if acc > max_acc:
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

    return model


if __name__ == '__main__':
    np.random.seed(10)
    params = parse_args('train')

    # dataset
    if params.trainset_aug:
        base_file = configs.data_dir[params.dataset] + 'multiBase.json'
    else:
        base_file = configs.data_dir[params.dataset] + 'base.json'
    val_file = configs.data_dir[params.dataset] + 'val.json'

    # backbone
    if 'Conv' in params.model:
        if params.dataset in ['cnn', 'wiki', 'bjyh', 'msyh', 'gfyh']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224

    if params.dataset in ['cnn', 'wiki', 'bjyh', 'msyh', 'gfyh']:
        assert params.model == 'Conv4' and not params.train_aug, 'dataset only support Conv4 without augmentation'
        params.model = 'Conv4S'

    optimization = 'Adam'

    if params.stop_epoch == -1:
        if params.method == 'maml':
            if params.n_shot == 1:
                params.stop_epoch = 300
            elif params.n_shot == 5:
                params.stop_epoch = 200
            else:
                params.stop_epoch = 200  # default
        else:  # meta-learning methods
            if params.n_shot == 1:
                #                 print("1111111")
                params.stop_epoch = 600
            elif params.n_shot == 5:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 400  # default

    if params.method in ['protonet', 'protonet_dw', 'maml']:
        # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        n_query = max(1, int(16 * params.test_n_way / params.train_n_way))
        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
        base_datamgr = SetDataManager(image_size, n_query=n_query, **train_few_shot_params)
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

        test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(image_size, n_query=n_query, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
        # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

        if params.method == 'protonet':
            model = ProtoNet(model_dict[params.model], **train_few_shot_params)

        elif params.method == 'protonet_dw':
            model = ProtoNetDW(model_dict[params.model], **train_few_shot_params)

        elif params.method == 'maml':
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            model = MAML(model_dict[params.model], approx=(params.method == 'maml_approx'), **train_few_shot_params)

            if params.dataset in ['cnn', 'wiki', 'bjyh', 'msyh', 'gfyh']:
                # maml use different parameter in omniglot
                model.n_task = 32
                model.task_update_num = 1
                model.train_lr = 0.1
    else:
        raise ValueError('Unknown method')

    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, params.dataset, params.model, params.method)
    if params.trainset_aug:
        params.checkpoint_dir += '_trainset_aug'
    #     if params.train_aug:
    #         params.checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)

    # print(params.checkpoint_dir)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml':
        # maml use multiple tasks in one update
        #         stop_epoch = params.stop_epoch * model.n_task
        stop_epoch = 5000

    print('{}(dataset:{},{}-way {}-shot,{} epochs)'.format(params.method, params.dataset, params.train_n_way,
                                                           params.n_shot, stop_epoch))
    model = train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params)


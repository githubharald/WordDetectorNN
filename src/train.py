import argparse
import json

import torch
from path import Path
from torch.utils.tensorboard import SummaryWriter

from dataloader import DataLoaderIAM
from dataset import DatasetIAM, DatasetIAMSplit
from eval import evaluate
from loss import compute_loss
from net import WordDetectorNet
from visualization import visualize

global_step = 0


def validate(net, loader, writer):
    global global_step

    net.eval()
    loader.reset()
    res = evaluate(net, loader, max_aabbs=1000)

    for i, (img, aabbs) in enumerate(zip(res.batch_imgs, res.batch_aabbs)):
        vis = visualize(img, aabbs)
        writer.add_image(f'img{i}', vis.transpose((2, 0, 1)), global_step)
        writer.add_scalar('val_loss', res.loss, global_step)
        writer.add_scalar('val_recall', res.metrics.recall(), global_step)
        writer.add_scalar('val_precision', res.metrics.precision(), global_step)
        writer.add_scalar('val_f1', res.metrics.f1(), global_step)

    return res.metrics.f1()


def train(net, optimizer, loader, writer):
    global global_step

    net.train()
    loader.reset()
    loader.random()
    for i in range(len(loader)):
        # get batch
        loader_item = loader[i]

        # forward pass
        optimizer.zero_grad()
        y = net(loader_item.batch_imgs)
        loss = compute_loss(y, loader_item.batch_gt_maps)

        # backward pass, optimize loss
        loss.backward()
        optimizer.step()

        # output
        print(f'{i + 1}/{len(loader)}: {loss}')
        writer.add_scalar('loss', loss, global_step)
        global_step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--caching', action='store_true')
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--early_stopping', type=int, default=50)
    args = parser.parse_args()

    writer = SummaryWriter('../log')

    net = WordDetectorNet()
    if args.pretrained:
        net.load_state_dict(torch.load('../model/weights'))
    net.to('cuda')

    # dataset that actually holds the data and 2 views for training and validation set
    dataset = DatasetIAM(args.data_dir, net.input_size, net.output_size, caching=args.caching)
    dataset_train = DatasetIAMSplit(dataset, 2 * args.batch_size, len(dataset))
    dataset_val = DatasetIAMSplit(dataset, 0, 2 * args.batch_size)

    # loaders
    loader_train = DataLoaderIAM(dataset_train, args.batch_size, net.input_size, net.output_size)
    loader_val = DataLoaderIAM(dataset_val, args.batch_size, net.input_size, net.output_size)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters())

    # main training loop
    epoch = 0
    best_val_f1 = 0
    no_improvement_since = 0
    while True:
        epoch += 1
        print(f'Epoch: {epoch}')
        train(net, optimizer, loader_train, writer)

        if epoch % args.val_freq == 0:
            val_f1 = validate(net, loader_val, writer)
            if val_f1 > best_val_f1:
                print(f'Improved on validation set (f1: {best_val_f1}->{val_f1}), save model')
                no_improvement_since = 0
                best_val_f1 = val_f1
                torch.save(net.state_dict(), '../model/weights')
                with open('../model/metadata.json', 'w') as f:
                    json.dump({'epoch': epoch, 'val_f1': val_f1}, f)
            else:
                no_improvement_since += 1

        # stop training if there were too many validation steps without improvement
        if no_improvement_since >= args.early_stopping:
            print(f'No improvement for {no_improvement_since} validation steps, stop training')
            break


if __name__ == '__main__':
    main()

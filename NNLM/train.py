import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from my_dataset import MyDataSet
from utils import generate_vocab, generate_dataset, create_lr_scheduler, train_one_epoch
from model import NNLM
from params import *


def main():
    print(f"using {device} device.")
    print('Using {} dataloader workers every process'.format(nw))

    tb_writer = SummaryWriter()

    # 生成单词表
    word2index_dict, index2word_dict = generate_vocab(train_path)
    n_class = len(word2index_dict)  # number of Vocabulary
    print('number of Vocabulary =', n_class)

    # 实例化数据集
    train_dataset = generate_dataset(train_path, word2index_dict, n_step)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=nw)

    model = NNLM(n_class, m, n_step, n_hidden).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs,
                                       warmup=True, warmup_epochs=1)

    # Training
    for epoch in range(epochs):
        # train
        train_loss = train_one_epoch(model=model,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     device=device,
                                     epoch=epoch,
                                     lr_scheduler=lr_scheduler)
        tb_writer.add_scalar('train_loss', train_loss, epoch)


if __name__ == '__main__':
    main()


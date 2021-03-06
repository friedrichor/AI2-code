import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import generate_vocab, generate_dataset, create_lr_scheduler, train_one_epoch, evaluate
from model import TextRNN
from params import *


def main():
    print(f"using {device} device.")
    print('Using {} dataloader workers every process'.format(nw))

    tb_writer = SummaryWriter()
    # % load_ext tensorboard
    # % tensorboard --logdir runs
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    # 生成单词表
    word2index_dict, index2word_dict = generate_vocab(train_path)
    n_class = len(word2index_dict)  # number of Vocabulary
    print('number of Vocabulary =', n_class)

    # 实例化数据集
    train_dataset = generate_dataset(train_path, word2index_dict, n_step)
    vaild_dataset = generate_dataset(valid_path, word2index_dict, n_step)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=nw)

    valid_loader = DataLoader(vaild_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=nw)

    model = TextRNN(n_class).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs,
                                       warmup=True, warmup_epochs=1)

    for epoch in range(epochs):
        # train
        train_loss = train_one_epoch(model=model,
                                     loss_function=loss_function,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     device=device,
                                     epoch=epoch,
                                     lr_scheduler=lr_scheduler)
        # validate
        valid_loss = evaluate(model=model,
                              loss_function=loss_function,
                              data_loader=valid_loader,
                              device=device,
                              epoch=epoch)

        tb_writer.add_scalar('train_loss', train_loss, epoch)
        tb_writer.add_scalar('valid_loss', valid_loss, epoch)
        tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)

        if (epoch + 1) % save_epoch == 0:
            torch.save(model, os.path.join(models_path, f'weitghts-{epoch + 1}.ckpt'))


if __name__ == '__main__':
    main()


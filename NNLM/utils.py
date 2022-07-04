from my_dataset import MyDataSet


def generate_vocab(data_path):
    word_list = []

    f = open(data_path, 'r')
    lines = f.readlines()
    for sentence in lines:
        word_list = sentence.split()
    word_list = list(set(word_list))

    # 生成字典
    word2index_dict = {w: i + 2 for i, w in enumerate(word_list)}
    index2word_dict = {i + 2: w for i, w in enumerate(word_list)}

    word2index_dict['<PAD>'] = 0
    word2index_dict['<UNK>'] = 1
    index2word_dict[0] = '<PAD>'
    index2word_dict[1] = '<UNK>'

    return word2index_dict, index2word_dict


def generate_dataset(data_path, word2index_dict, n_step=5):
    """
    :param data_path: 数据集路径
    :param word2index_dict: word2index字典
    :param n_step: 窗口大小
    :return: 实例化后的数据集
    """
    input_list = []
    target_list = []

    f = open(data_path, 'r')
    lines = f.readlines()
    for sentence in lines:
        word_list = sentence.split()
        index_list = [word2index_dict[word] for word in word_list]
        if len(word_list) < n_step + 1:  # 句子中单词不足，padding
            word_list = ['<pad>'] * (n_step + 1 - len(word_list)) + word_list
        for i in range(len(word_list) - n_step):
            input = index_list[i: i + n_step]
            target = index_list[i + n_step]

            input_list.append(input)
            target_list.append(target)

    # 实例化数据集
    dataset = MyDataSet(input_list, target_list)

    return dataset


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = criterion
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input, target = data

        pred = model(input.to(device))

        loss = loss_function(pred, target.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return accu_loss.item() / (step + 1)


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
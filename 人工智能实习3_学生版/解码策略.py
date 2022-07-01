import random

import numpy as np
import torch

from tool.Global import *
import torch.nn.functional as F
import math

# 随机搜索:在下一个token生成的时候，按照概率值，进行多项式采样，选择一个token
def sampling(device, model, enc_vocab2id, dec_id2vocab, dec_vocab2id, source_sentence):
    source_sentence = source_sentence.split(char_space)

    # 最大搜索长度，当句子解码到该长度时，如果还没遇到char_end，则停止.
    dec_max_len = len(source_sentence) * 1.5

    # encoder为德语的编码
    enc_input = []
    for w in source_sentence:
        enc_input.append(enc_vocab2id[w])


    # 搜索结果
    final_result = []
    # 搜索结果的得分
    final_scores = []

    final_result.append([dec_vocab2id[char_start]])
    final_scores.append(0)



    enc_output=None


    while True:
        # 将之前解码出来的序列，放入编码器，继续解码
        dec_input = final_result[0]

        # 对该序列进行搜索
        if enc_output is None:
            enc_output, output = model(torch.tensor([enc_input]).to(device), torch.tensor([dec_input]).to(device), enc_output)
        else:
            _, output = model(torch.tensor([enc_input]).to(device), torch.tensor([dec_input]).to(device),
                                       enc_output)

        # prob为搜索到的单词在词表中的概率分布
        # 例如，词表为{0,1,2,3},则prob可能为[0.1,0.4,0.3,0.2]，分别表示对应每个词的概率
        prob = F.softmax(output[-1], dim=-1)

        # 多项式采样，按照概率值，随机抽取一个词，放入final_result中，例如上述prob中，最大概率抽中1
        random_char_idx=torch.multinomial(prob, 1).data[0].item()

        final_result[0].append(random_char_idx)

        # 计算该单词的log概率，并将其累加
        final_scores[0]+=math.log(prob[random_char_idx].item())

        # 判断该序列是否有必要继续搜索
        sentence_len = len(dec_input)
        last_word_id = dec_input[len(dec_input) - 1]
        last_word_vocab = dec_id2vocab[last_word_id]

        if last_word_vocab == char_end or sentence_len >= dec_max_len:
            # 解码到char_end，或者已经抵达最大长度，停止解码
            break

    # final_scores:list，存放每一个结果的概率，例如：[-11],(log后的得分，直接加和)
    # final_result:list[list],存放解码结果，例如[[102,34,65,123]],(每个list为一条可能的翻译结果)
    # 最后返回得分最大的一条搜索到的序列
    return final_scores[0], final_result[0]

# 贪心搜索:在下一个token生成的时候，按照概率值，选择概率值最大的token
def greedySearch(device, model, enc_vocab2id, dec_id2vocab, dec_vocab2id, source_sentence):
    # 搜索结果
    final_result = []
    # 搜索结果的得分
    final_scores = []

   
    # final_scores:list，存放每一个结果的概率，例如：[-11],(log后的得分，直接加和)
    # final_result:list[list],存放解码结果，例如[[102,34,65,123]],(每个list为一条可能的翻译结果)
    # 最后返回得分最大的一条搜索到的序列
    return final_scores[0], final_result[0]


# top-k sampling
def topKSampling(device, model, enc_vocab2id, dec_id2vocab, dec_vocab2id, source_sentence,k):

    final_scores=[]
    final_result=[]

    # final_scores:list，存放每一个结果的概率，例如：[-11],(log后的得分，直接加和)
    # final_result:list[list],存放解码结果，例如[[102,34,65,123]],(每个list为一条可能的翻译结果)
    # 最后返回得分最大的一条搜索到的序列
    return final_scores[0], final_result[0]

# top-p sampling
def topPSampling(device, model, enc_vocab2id, dec_id2vocab, dec_vocab2id, source_sentence,p):
    final_scores=[]
    final_result=[]

    # final_scores:list，存放每一个结果的概率，例如：[-11],(log后的得分，直接加和)
    # final_result:list[list],存放解码结果，例如[[102,34,65,123]],(每个list为一条可能的翻译结果)
    # 最后返回得分最大的一条搜索到的序列
    return final_scores[0], final_result[0]

# 束搜索
# k:束宽
def beamSearch(device,model, enc_vocab2id, dec_id2vocab, dec_vocab2id, source_sentence, k: int):

    final_result = []
    final_scores = []

   
    # final_scores:list，存放每一个结果的概率，例如：[-11,-12,-15],(log后的得分，直接加和)
    # final_result:list[list],存放解码结果，例如[[102,34,65,123],[12,142,54],[123,431,134]],(每个list为一条可能的翻译结果)
    # 最后返回得分最大的
    return final_scores[0], final_result[0]





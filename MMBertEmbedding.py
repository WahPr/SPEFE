import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
yyy = 0.7


class CPC(nn.Module):
    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.activation = getattr(nn, activation)
        self.relu = torch.nn.ReLU()

        if n_layers == 1:
            self.net = nn.Linear(
                in_features=y_size,
                # out_features=y_size
                out_features=x_size
            )
        
    def forward(self, x, y):
        x_pred = self.net(y)
        x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)
        x = x / x.norm(dim=1, keepdim=True)

        pos = torch.sum(x*x_pred, dim=-1)
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)

        # nce = -(pos - neg).mean()
        nce = self.relu((neg - pos).mean())


        return nce

class JointEmbeddings(nn.Module):
    def __init__(self, hidden_size, dropout_prob, dataset):
        super(JointEmbeddings, self).__init__()
        if dataset == 'mosi':
            self.VISUALDIM = MOSIVISUALDIM
            self.SPEECHDIM = CMUSPEECHDIM
        elif dataset == 'mosei':
            self.VISUALDIM = MOSEIVISUALDIM
            self.SPEECHDIM = CMUSPEECHDIM
        elif dataset == 'ur_funny':
            self.VISUALDIM = FUNNYVISUALDIM
            self.SPEECHDIM = FUNNYSPEECHDIM

        self.W_cv = nn.Linear(self.VISUALDIM + TEXTDIM, TEXTDIM)
        self.W_cs = nn.Linear(self.SPEECHDIM + TEXTDIM, TEXTDIM)

        self.Wv = nn.Linear(self.VISUALDIM, TEXTDIM)
        self.Ws = nn.Linear(self.SPEECHDIM, TEXTDIM)


        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def add_positional_encoding(self, input_embs, pair_ids):

        position_encoding_h = torch.arange(pair_ids.size(1), dtype=torch.float) / pair_ids.size(1)

        for i in range(pair_ids.size(1)):
            if i % 2:
                position_encoding_h[i] = torch.cos(1 / (torch.pow(1024, position_encoding_h[i])))
            else:
                position_encoding_h[i] = torch.sin(1 / (torch.pow(1024, position_encoding_h[i])))

        position_encoding_h = position_encoding_h.unsqueeze(0)  # 1*40
        if input_embs.is_cuda:
            position_encoding_h = position_encoding_h.cuda()

        # 获取输入数据的序列长度
        if pair_ids.size()[-1] == self.VISUALDIM:
            seq_length = pair_ids.size(2)

            # 生成位置编码
            # position_encoding = torch.arange(seq_length, dtype=torch.float) / seq_length * torch.pi/2
            position_encoding = torch.arange(seq_length, dtype=torch.float)/seq_length

            for i in range(seq_length):
                if i % 2:
                    position_encoding[i] = torch.cos(1/(torch.pow(1024, position_encoding[i])))
                else:
                    position_encoding[i] = torch.sin(1/(torch.pow(1024, position_encoding[i])))

            position_encoding = position_encoding.unsqueeze(0)
            if input_embs.is_cuda:
                position_encoding = position_encoding.cuda() # 1*768
            # pair_ids = (1 - yyy) * pair_ids + yyy * (position_encoding.unsqueeze(0) + position_encoding_h.unsqueeze(2))

            # position_encoding = position_encoding.repeat(pair_ids.size(0), 1)

        if pair_ids.size()[-1] == self.SPEECHDIM:
            seq_length = pair_ids.size(2)

            # 生成位置编码
            # position_encoding = torch.arange(seq_length, dtype=torch.float) / seq_length * torch.pi/2
            position_encoding = torch.arange(seq_length, dtype=torch.float)/seq_length

            for i in range(seq_length):
                if i % 2:
                    position_encoding[i] = torch.cos(1 / (torch.pow(1024, position_encoding[i])))
                else:
                    position_encoding[i] = torch.sin(1 / (torch.pow(1024, position_encoding[i])))

            position_encoding = position_encoding.unsqueeze(0)
            if input_embs.is_cuda:
                position_encoding = position_encoding.cuda()

                # 根据输入数据的维度确定位置编码的维度
            # position_encoding = position_encoding.repeat(pair_ids.size(0), 1)


        pair_ids = (1-yyy)*pair_ids + yyy*(position_encoding.unsqueeze(0)+position_encoding_h.unsqueeze(2))
        # pair_ids = (1 - yyy) * pair_ids + yyy * (torch.mm(position_encoding_h.squeeze(0).unsqueeze(1), position_encoding))

        return pair_ids

        # 将位置编码与输入数据相加
        # input_embs = input_embs + position_encoding.unsqueeze(-1)
        # return input_embs


    def forward(self, input_embs, pair_ids):
        assert input_embs is not None, "You miss input_embs"
        assert pair_ids is not None, "You miss pair_ids"

        # pair_ids = self.add_positional_encoding(input_embs, pair_ids)

        if pair_ids.size()[-1] == self.VISUALDIM:
            pair_embeds = F.relu(self.Wv(pair_ids.float()))
        elif pair_ids.size()[-1] == self.SPEECHDIM:
            pair_embeds = F.relu(self.Ws(pair_ids.float()))
        else:
            raise Exception('Wrong Dimension')

        # 添加位置编码
        # input_embs = self.add_positional_encoding(input_embs, pair_ids)

        input_embeds = torch.cat((input_embs, pair_embeds), dim=1)

        embeddings = self.LayerNorm(input_embeds)
        embeddings = self.dropout(embeddings)

        return embeddings


# class JointEmbeddings(nn.Module):
#     def __init__(self, hidden_size, dropout_prob, dataset):
#         super().__init__()
#
#         if dataset =='mosi':
#             self.VISUALDIM = MOSIVISUALDIM
#             self.SPEECHDIM = CMUSPEECHDIM
#         elif dataset =='mosei':
#             self.VISUALDIM = MOSEIVISUALDIM
#             self.SPEECHDIM = CMUSPEECHDIM
#         elif dataset == 'ur_funny':
#             self.VISUALDIM = FUNNYVISUALDIM
#             self.SPEECHDIM = FUNNYSPEECHDIM
#
#         self.W_cv = nn.Linear(self.VISUALDIM+TEXTDIM, TEXTDIM)  # mosi.shape(47+768,768)/mosei.shape(35+1024,1024)
#         self.W_cs = nn.Linear(self.SPEECHDIM+TEXTDIM, TEXTDIM)  # mosi.shape(74+768,768)/mosei.shape(74+1024,1024)
#
# #         # self.Wv = nn.Linear(self.VISUALDIM, TEXTDIM)  # mosi.shape(47->768)/mosei.shape(35->1024) 原始的
# #         # self.Ws = nn.Linear(self.SPEECHDIM, TEXTDIM)  # mosi.shape(74->768)/mosie.shape(74->1024) 原始的
# #
# #         self.visual_pos_embedding = nn.Parameter(torch.randn(1, self.VISUALDIM, TEXTDIM))
# #         self.audio_pos_embedding = nn.Parameter(torch.randn(1, self.SPEECHDIM, TEXTDIM))
# #         self.Wv = nn.Linear(self.VISUALDIM, TEXTDIM)
# #         self.Ws = nn.Linear(self.SPEECHDIM, TEXTDIM)
# #         # self.LayerNorm = nn.LayerNorm(hidden_size)
# #         # self.dropout = nn.Dropout(dropout_prob)
# #
# #     def forward(self, input_embs, pair_ids):
# #
# #         if pair_ids.size()[-1] == self.VISUALDIM:
# #              pair_embeds = F.relu(self.Wv(pair_ids.float()))
# #              pair_embeds += self.visual_pos_embedding[:, :pair_ids.size(1)]
# #         elif pair_ids.size()[-1] == self.SPEECHDIM:
# #              pair_embeds = F.relu(self.Ws(pair_ids.float()))
# #              pair_embeds += self.audio_pos_embedding[:, :pair_ids.size(1)]
# #         else:
# #             raise Exception('Wrong Dimension')
# #
# #         embeddings = torch.cat((input_embs, pair_embeds), dim=1)
# #         # embeddings = self.LayerNorm(inputs_embeds)  # torch.size(32,80,768)
# #         # embeddings = self.dropout(embeddings)
# #
# #         return embeddings
# #
#       #<------------------------------->
#         self.Wv = nn.Linear(self.VISUALDIM, self.VISUALDIM)
#         self.Ws = nn.Linear(self.SPEECHDIM, self.SPEECHDIM)
# #       #<------------------------------->添加
# #
# #       #<------------------------------->
#         self.Wv_v = nn.Linear(self.VISUALDIM, TEXTDIM)
#         self.Wv_s = nn.Linear(self.SPEECHDIM, TEXTDIM)
# #       # <------------------------------->添加
#
#         self.LayerNorm = nn.LayerNorm(hidden_size)
#         self.dropout = nn.Dropout(dropout_prob)
#
#     def forward(self, input_embs, pair_ids):            #input_embs形状(32,40,768) pair_ids形状(32,40,47)
#         assert input_embs != None, "You miss input_embs"
#         assert pair_ids != None, "You miss pair_ids"
#
#         if pair_ids.size()[-1] == self.VISUALDIM:   # 如果最后一个维度的大小等于self.VISUALDIM,表示它包含视觉数据
#             pair_embeds = F.relu(self.Wv(pair_ids.float()))    # 使用self.Wv线性层将视觉数据映射到文本数据，使用relu激活函数
#             pair_embeds = F.relu(pair_embeds + pair_ids.float())  # 添加
#             pair_embeds = F.relu(self.Wv_v(pair_embeds))    # 添加
#         elif pair_ids.size()[-1] == self.SPEECHDIM:   # 如果最后一个维度的大小等于self.SPEECHDIM,表示它包含语音数据
#             pair_embeds = F.relu(self.Ws(pair_ids.float()))   # 使用self.Ws线性层将语音数据映射到文本数据，使用relu激活函数
#             pair_embeds = F.relu(self.Ws(pair_embeds + pair_ids.float()))  # 添加
#             pair_embeds = F.relu(self.Wv_s(pair_embeds))  # 添加
#         else:
#             raise Exception('Wrong Dimension')
#
#         inputs_embeds = torch.cat((input_embs, pair_embeds), dim=1)  #torch.size(32,80,768)
#
#         embeddings = self.LayerNorm(inputs_embeds)   #torch.size(32,80,768)
#         embeddings = self.dropout(embeddings)
#
#         return embeddings



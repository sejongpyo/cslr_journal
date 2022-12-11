from tkinter.messagebox import NO
import torch
from torch import nn
import torch.nn.functional as F

from .model_utils import simple_adj_mat, make_adj_mat, make_norm_adj_mat, make_sel_mat, make_sel_mat_2, make_sel_mat_3

from .cnn import ResNet34
from .lstm import LSTM
from .gcn import GConv
from .tconv import TemporalConv

class CombinedModel_lstm(nn.Module):
    def __init__(self, cfg, loss):
        super(CombinedModel_lstm, self).__init__()
        num_classes = cfg['num_classes']
        cnn_pretrain = cfg['cnn_pretrain']
        lstm_dim = cfg['lstm_dim']
        lstm_layer = cfg['lstm_layer']
        lstm_drop = cfg['lstm_drop']
        g_outdim = cfg['g_outdim']
        g_bn = cfg['g_bn']
        g_layer = cfg['g_layer']
        key_dim = cfg['key_dim']
        self.device = cfg['device']
        self.loss = loss
        
        '''cnn'''
        self.cnn = ResNet34(cnn_pretrain) # 512
        self.cnn_out = 512
        
        '''lstm'''
        self.lstm = LSTM(self.cnn_out, lstm_dim, lstm_layer, lstm_drop)
        
        '''key selector'''
        self.key_layer = nn.Linear(lstm_dim*2, 1)
        
        '''graph'''
        self.gcn = GConv(lstm_dim*2, g_outdim, g_bn)
        # self.gcn_2 = GConv(g_outdim, g_outdim, g_bn)
        
        '''classifier'''
        self.classifier = nn.Linear(g_outdim, num_classes)
        
    def forward(self, x, prob_sizes, key=False):
        x = self.cnn(x, prob_sizes, True)
        x = self.lstm(x, prob_sizes)

        if key:
            key_logit = self.key_layer(x)
            adj = make_sel_mat_2(key_logit, prob_sizes, self.device)
            adj_2 = make_sel_mat_3(key_logit, prob_sizes, self.device)
        else:
            key_logit = None
            adj = simple_adj_mat(x, self.device)
            adj_2 = simple_adj_mat(x, self.device)

        x, _ = self.gcn(x, adj)
        # x, _ = self.gcn_2(x, adj)
        logit = self.classifier(x)
                
        return {"logit": logit, "key_logit": key_logit}

    def loss_calculation(self, logit, vid_lgt, label, label_lgt, bin_lbl=None):
        loss = 0
        loss += self.loss['ctc'](logit['logit'].log_softmax(-1).transpose(0, 1),
                                 label, vid_lgt, label_lgt).mean()
        if logit['key_logit'] != None:
            loss += self.loss['bce'](logit['key_logit'], bin_lbl)
        
        return loss
    
class CombinedModel_tconv(nn.Module):
    def __init__(self, cfg, loss):
        super(CombinedModel_tconv, self).__init__()
        num_classes = cfg['num_classes']
        cnn_pretrain = cfg['cnn_pretrain']
        lstm_dim = cfg['lstm_dim']
        lstm_layer = cfg['lstm_layer']
        lstm_drop = cfg['lstm_drop']
        g_outdim = cfg['g_outdim']
        g_bn = cfg['g_bn']
        g_layer = cfg['g_layer']
        key_dim = cfg['key_dim']
        conv_type = cfg['tconv_type']
        self.device = cfg['device']
        self.loss = loss
        
        self.cnn = ResNet34(cnn_pretrain) # 512
        self.cnn_out = 512
        
        '''tconv'''
        self.tconv = TemporalConv(self.cnn_out, lstm_dim*2, conv_type=conv_type, use_bn=True, num_classes=1)
        
        '''graph'''
        self.gcn = GConv(lstm_dim*2, g_outdim, g_bn)
        # self.gcn_2 = GConv(g_outdim, g_outdim, g_bn)
        
        '''classifier'''
        self.classifier = nn.Linear(g_outdim, num_classes)
        
    def forward(self, x, prob_sizes=[10], key=False):
        x = self.cnn(x, prob_sizes)
        tconv_out = self.tconv(x, prob_sizes, key)

        x = tconv_out['visual_feat']
        prob_sizes = tconv_out['feat_len']
        key_logit = tconv_out['conv_logits']

        if key:
            adj = make_sel_mat_2(key_logit, prob_sizes, self.device)
            #     front_graph = front_graph / front_graph.sum(dim=-1, keepdim=True)
            # adj_2 = make_sel_mat_3(key_logit, prob_sizes, self.device)
        else:
            key_logit = None
            adj = simple_adj_mat(x, self.device)

        x, _ = self.gcn(x, adj)
        # x, _ = self.gcn_2(x, adj)
        logit = self.classifier(x)
                
        return {"logit": logit, "key_logit": key_logit, "vid_len": prob_sizes}

    def loss_calculation(self, logit, vid_lgt, label, label_lgt, bin_lbl=None):
        loss = 0
        loss += self.loss['ctc'](logit['logit'].log_softmax(-1).transpose(0, 1),
                                 label, vid_lgt, label_lgt).mean()
        if logit['key_logit'] != None:
            loss += self.loss['bce'](logit['key_logit'], bin_lbl)
        
        return loss
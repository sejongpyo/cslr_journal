import os
import pdb
import time
from tokenize import group
import torch
import ctcdecode
import numpy as np
from itertools import groupby
import torch.nn.functional as F


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        self.ctc_decoder = ctcdecode.CTCBeamDecoder(vocab, beam_width=10, blank_id=blank_id,
                                                    num_processes=10)

    def decode(self, nn_output, vid_lgt, make, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, make, probs)

    def BeamSearch(self, nn_output, vid_lgt, make, probs=False):
        '''
        CTCBeamDecoder Shape:
                - Input:  nn_output (B, T, N), which should be passed through a softmax layer
                - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
                          beam_scores (B, N_beams), p=1/np.exp(beam_score)
                          timesteps (B, N_beams)
                          out_lens (B, N_beams)
        '''
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
        ret_list = []
        time_list = []
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]
            time_result = timesteps[batch_idx][0][:out_seq_len[batch_idx][0]]
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
                time_result = torch.stack([x[0] for x in groupby(time_result)])
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(first_result)])
            time_list.append(time_result)
        if make:
            # saved_bin_li = self.save_bin(time_list, vid_lgt)
            saved_bin_li = self.save_bin(nn_output, vid_lgt)
            return {'ret_list': ret_list, 'save_in': saved_bin_li}
        else:
            return {'ret_list': ret_list}

    # def save_bin(self, time_result, vid_lgt):
    #     saved_bin_li = []
    #     for idx in range(len(vid_lgt)):
    #         bin_arr = torch.zeros(vid_lgt[idx])
    #         bin_arr[time_result[idx].type(torch.LongTensor)] = 1
    #         bin_arr = bin_arr.unsqueeze(0).unsqueeze(-1)
    #         saved_bin_li.append(bin_arr)

    #     return saved_bin_li

    def save_bin(self, probs, prob_sizes):
        save_bin_li = []
        probs = probs.data.numpy()
            
        for j in range(len(probs)):
            bin_li = self.greedy_decode(probs[j][:prob_sizes[j]])
            bin_li = bin_li.unsqueeze(0).unsqueeze(-1)
            save_bin_li.append(bin_li)
            
        return save_bin_li

    def greedy_decode(self, prob):
        indexes = np.argmax(prob, axis=1)
        bin_li = []
        prev_index = -1
        for i in range(len(indexes)):
            if indexes[i] == 0:
                prev_index = -1
                bin_li.append(0)
                continue
            elif indexes[i] == prev_index:
                bin_li.append(1)
                continue
            else:
                bin_li.append(1)
                prev_index = indexes[i]
        return torch.tensor(bin_li, dtype=torch.float32)

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(max_result)])
        return ret_list

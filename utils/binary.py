import torch

def make_bin_lbl(bin_dict, vid_lgt, im_dir, index=None):
    bin_lbl = torch.zeros((len(im_dir), vid_lgt[0], 1))

    for i in range(len(im_dir)):
        if len(index) == 0 or index == None:
            lbl = bin_dict[im_dir[i]] # 1, n, 1
        else:
            temp = bin_dict[im_dir[i]]
            lbl = temp[:, index[i], :]

        try:
            bin_lbl[i] = lbl
        except:
            pad_tensor = torch.zeros((1, (vid_lgt[0] - vid_lgt[i]), 1))
            lbl = torch.cat([lbl, pad_tensor], dim = 1)
            bin_lbl[i] = lbl

    return bin_lbl

def make_bin_lbl_2(bin_li, vid_lgt):
    bin_lbl = torch.zeros((len(bin_li), vid_lgt[0], 1))

    for i in range(len(bin_li)):
        lbl = bin_li[i] # 1, n, 1

        try:
            bin_lbl[i] = lbl
        except:
            pad_tensor = torch.zeros((1, (vid_lgt[0] - vid_lgt[i]), 1))
            lbl = torch.cat([lbl, pad_tensor], dim = 1)
            bin_lbl[i] = lbl

    return bin_lbl
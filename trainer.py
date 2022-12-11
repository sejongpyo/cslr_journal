import torch
import sys
import numpy as np
from tqdm import tqdm

import utils
# from sklearn.metrics import recall_score
from evaluation.slr_eval.wer_calculation import evaluate

def only_ctc_tr(data_loader_dic, model, decoder, optimizer, epoch, hyper_cfg, work_dir, bin_dict):
    key = False
    make = False
    data_loader = data_loader_dic['train']

    # if epoch < 10:
    #     key = False
    #     make = False
    #     data_loader = data_loader_dic['train']
    # elif epoch == 10:
    #     key = False
    #     make = True
    #     data_loader = data_loader_dic['train_build']
    # elif epoch % 10 == 0:
    #     key = True
    #     make = True
    #     data_loader = data_loader_dic['train_build']
    # else:
    #     key = True
    #     make = False
    #     data_loader = data_loader_dic['train']

    print(f"--- Epoch {epoch} ---")
    model.train()
    total_sent = []
    total_info = []
    epoch_metrics = {'loss':[], 'acc':[]}
    for i_batch, sample in enumerate(data_loader):
        optimizer.zero_grad()
        vid = sample[0].to(hyper_cfg['device'])
        vid_lgt = sample[1]
        label = sample[2]
        label_lgt = sample[3]
        im_dir = sample[4]
        index = sample[5]
        
        logit_dict = model(vid, vid_lgt, key=key)

        if key:
            bin_lbl = utils.make_bin_lbl(bin_dict, logit_dict['vid_len'], im_dir).to(hyper_cfg['device'])
        else:
            bin_lbl = None

        loss = model.loss_calculation(logit_dict, logit_dict['vid_len'], label, label_lgt, bin_lbl)
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print(sample[4])
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hyper_cfg['grad_clip'])
        optimizer.step()

        total_info += [file_name.split("|")[0] for file_name in sample[4]]
        decoded_dict = decoder.decode(logit_dict['logit'], logit_dict['vid_len'], make=make, batch_first=True, probs=False)
        total_sent += decoded_dict['ret_list']

        if make:
            for idx in range(len(decoded_dict['save_in'])):
                bin_dict[im_dir[idx]] = decoded_dict['save_in'][idx].cpu()

        epoch_metrics['loss'].append(loss.item())
        sys.stdout.write(
            "\r[Epoch %d] [Batch %d/%d] [Loss: %f (%f)]"
            % (epoch, i_batch, len(data_loader),
                loss.item(), np.mean(epoch_metrics["loss"])))

    write2file(work_dir+"output-hypothesis-{}.ctm".format("train"), total_info, total_sent)
    acc = evaluate(
        prefix=work_dir, mode="train", output_file="output-hypothesis-{}.ctm".format("train"),
        evaluate_dir=hyper_cfg['evaluation_dir'],
        evaluate_prefix=hyper_cfg['evaluation_prefix'],
        output_dir="epoch_{}_result/".format(epoch),
        python_evaluate=True
    )
    print("train acc: ", acc)
        
    return np.mean(epoch_metrics["loss"]), acc, bin_dict, model

def only_ctc_te(data_loader, model, decoder, epoch, hyper_cfg, work_dir, bin_dict):
    print("")
    key = True

    model.eval()
    total_info = []
    total_sent = []
    with torch.no_grad():
        for i_batch, sample in tqdm(enumerate(data_loader)):
            vid = sample[0].to(hyper_cfg['device'])
            vid_lgt = sample[1]
            
            logit_dict = model(vid, vid_lgt, key=key)
            
            total_info += [file_name.split("|")[0] for file_name in sample[4]]
            decoded_dict = decoder.decode(logit_dict['logit'], logit_dict['vid_len'], make=True, batch_first=True, probs=False)
            total_sent += decoded_dict['ret_list']

    write2file(work_dir+"output-hypothesis-{}.ctm".format("test"), total_info, total_sent)
    acc = evaluate(
        prefix=work_dir, mode="test", output_file="output-hypothesis-{}.ctm".format("test"),
        evaluate_dir=hyper_cfg['evaluation_dir'],
        evaluate_prefix=hyper_cfg['evaluation_prefix'],
        output_dir="epoch_{}_result/".format(epoch),
        python_evaluate=True
    )
    print("test acc: ", acc)
        
    return acc, bin_dict

def only_ctc_dev(data_loader, model, decoder, epoch, hyper_cfg, work_dir, bin_dict):
    print("")
    # if epoch > 11:
    #     key = True
    # else:
    #     key = False
    key = True

    model.eval()
    total_info = []
    total_sent = []
    with torch.no_grad():
        for i_batch, sample in tqdm(enumerate(data_loader)):
            vid = sample[0].to(hyper_cfg['device'])
            vid_lgt = sample[1]
            
            logit_dict = model(vid, vid_lgt, key=key)

            total_info += [file_name.split("|")[0] for file_name in sample[4]]
            decoded_dict = decoder.decode(logit_dict['logit'], logit_dict['vid_len'], make=False, batch_first=True, probs=False)
            total_sent += decoded_dict['ret_list']

    write2file(work_dir+"output-hypothesis-{}.ctm".format("dev"), total_info, total_sent)
    acc = evaluate(
        prefix=work_dir, mode="dev", output_file="output-hypothesis-{}.ctm".format("dev"),
        evaluate_dir=hyper_cfg['evaluation_dir'],
        evaluate_prefix=hyper_cfg['evaluation_prefix'],
        output_dir="epoch_{}_result/".format(epoch),
        python_evaluate=True
    )
    print("dev acc: ", acc)
        
    return acc, bin_dict

def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                 word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100,
                                                 word[0]))

def only_ctc_tr_2(data_loader, model, decoder, optimizer, epoch, hyper_cfg, work_dir, bin_dict):
    key = True
    make = True
    print(f"--- Epoch {epoch} ---")
    model.train()
    total_sent = []
    total_info = []
    epoch_metrics = {'loss':[], 'acc':[]}
    for i_batch, sample in enumerate(data_loader):
        optimizer.zero_grad()
        vid = sample[0].to(hyper_cfg['device'])
        vid_lgt = sample[1]
        label = sample[2]
        label_lgt = sample[3]
        im_dir = sample[4]
        # index = sample[5]
        
        logit_dict = model(vid, vid_lgt, key=key)

        total_info += [file_name.split("|")[0] for file_name in sample[4]]
        decoded_dict = decoder.decode(logit_dict['logit'], logit_dict['vid_len'], make=make, batch_first=True, probs=False)
        total_sent += decoded_dict['ret_list']

        if key:
            bin_lbl = utils.make_bin_lbl_2(decoded_dict['save_in'], logit_dict['vid_len']).to(hyper_cfg['device'])
        else:
            bin_lbl = None

        loss = model.loss_calculation(logit_dict, logit_dict['vid_len'], label, label_lgt, bin_lbl)
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print(sample[4])
            continue
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), hyper_cfg['grad_clip'])
        optimizer.step()

        # if make:
        #     for idx in range(len(decoded_dict['save_in'])):
        #         bin_dict[im_dir[idx]] = decoded_dict['save_in'][idx].cpu()

        epoch_metrics['loss'].append(loss.item())
        sys.stdout.write(
            "\r[Epoch %d] [Batch %d/%d] [Loss: %f (%f)]"
            % (epoch, i_batch, len(data_loader),
                loss.item(), np.mean(epoch_metrics["loss"])))

    write2file(work_dir+"output-hypothesis-{}.ctm".format("train"), total_info, total_sent)
    acc = evaluate(
        prefix=work_dir, mode="train", output_file="output-hypothesis-{}.ctm".format("train"),
        evaluate_dir=hyper_cfg['evaluation_dir'],
        evaluate_prefix=hyper_cfg['evaluation_prefix'],
        output_dir="epoch_{}_result/".format(epoch),
        python_evaluate=True
    )
    print("train acc: ", acc)
        
    return np.mean(epoch_metrics["loss"]), acc, bin_dict, model

def only_lstm_tr(data_loader_dic, model, decoder, optimizer, epoch, hyper_cfg, work_dir, bin_dict):
    if epoch < 10:
        key = False
        make = False
        data_loader = data_loader_dic['train']
    elif epoch == 10:
        key = False
        make = True
        data_loader = data_loader_dic['train_build']
    elif epoch % 10 == 0:
        key = True
        make = True
        data_loader = data_loader_dic['train_build']
    else:
        key = True
        make = False
        data_loader = data_loader_dic['train']

    print(f"--- Epoch {epoch} ---")
    model.train()
    total_sent = []
    total_info = []
    epoch_metrics = {'loss':[], 'acc':[]}
    for i_batch, sample in enumerate(data_loader):
        optimizer.zero_grad()
        vid = sample[0].to(hyper_cfg['device'])
        vid_lgt = sample[1]
        label = sample[2]
        label_lgt = sample[3]
        im_dir = sample[4]
        index = sample[5]
        
        logit_dict = model(vid, vid_lgt, key=key)

        if key:
            bin_lbl = utils.make_bin_lbl(bin_dict, vid_lgt, im_dir, index).to(hyper_cfg['device'])
        else:
            bin_lbl = None

        loss = model.loss_calculation(logit_dict, vid_lgt, label, label_lgt, bin_lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hyper_cfg['grad_clip'])
        optimizer.step()

        total_info += [file_name.split("|")[0] for file_name in sample[4]]
        decoded_dict = decoder.decode(logit_dict['logit'], vid_lgt, make=make, batch_first=True, probs=False)
        total_sent += decoded_dict['ret_list']

        if make:
            for idx in range(len(decoded_dict['save_in'])):
                bin_dict[im_dir[idx]] = decoded_dict['save_in'][idx].cpu()

        epoch_metrics['loss'].append(loss.item())
        sys.stdout.write(
            "\r[Epoch %d] [Batch %d/%d] [Loss: %f (%f)]"
            % (epoch, i_batch, len(data_loader),
                loss.item(), np.mean(epoch_metrics["loss"])))

    write2file(work_dir+"output-hypothesis-{}.ctm".format("train"), total_info, total_sent)
    acc = evaluate(
        prefix=work_dir, mode="train", output_file="output-hypothesis-{}.ctm".format("train"),
        evaluate_dir=hyper_cfg['evaluation_dir'],
        evaluate_prefix=hyper_cfg['evaluation_prefix'],
        output_dir="epoch_{}_result/".format(epoch),
        python_evaluate=True
    )
    print("train acc: ", acc)
        
    return np.mean(epoch_metrics["loss"]), acc, bin_dict, model

def only_lstm_te(data_loader, model, decoder, epoch, hyper_cfg, work_dir, bin_dict):
    print("")
    if epoch > 11:
        key = True
    else:
        key = False

    model.eval()
    total_info = []
    total_sent = []
    with torch.no_grad():
        for i_batch, sample in tqdm(enumerate(data_loader)):
            vid = sample[0].to(hyper_cfg['device'])
            vid_lgt = sample[1]
            
            logit_dict = model(vid, vid_lgt, key=key)

            total_info += [file_name.split("|")[0] for file_name in sample[4]]
            decoded_dict = decoder.decode(logit_dict['logit'], vid_lgt, make=False, batch_first=True, probs=False)
            total_sent += decoded_dict['ret_list']
                
    write2file(work_dir+"output-hypothesis-{}.ctm".format("dev"), total_info, total_sent)
    acc = evaluate(
        prefix=work_dir, mode="dev", output_file="output-hypothesis-{}.ctm".format("dev"),
        evaluate_dir=hyper_cfg['evaluation_dir'],
        evaluate_prefix=hyper_cfg['evaluation_prefix'],
        output_dir="epoch_{}_result/".format(epoch),
        python_evaluate=True
    )
    print("dev acc: ", acc)
        
    return acc, bin_dict
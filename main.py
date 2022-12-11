import torch
import numpy as np
import os
import wandb
import yaml
import random
from torch.optim import Adam, SGD, lr_scheduler
from torch import nn

from data import BaseFeeder
from models.model import CombinedModel_lstm, CombinedModel_tconv # , CombinedModel_key, gcn_first
import utils
import trainer

def run():
    rng = utils.RandomState(seed=123)

    cur_dir = os.getcwd()
    out_dir = os.path.join(cur_dir, 'checkpoint')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    """hyperparameters"""
    main_cfg = utils.parameters.load_cfg("config.yaml")
    model_cfg, hyper_cfg = main_cfg['model'], main_cfg['hyper']
    gloss_dict = np.load(main_cfg['dict_path'], allow_pickle=True).item()
    model_cfg['num_classes'] = len(gloss_dict) + 1
    start_ep = 0
    print(f"{'---'*10}Used GPU device: {hyper_cfg['device']}{'---'*10}")

    '''wandb'''
    nameofrun = main_cfg['nameofrun']
    # wan_run = wandb.init(project="final_rwth", name = nameofrun, config=main_cfg) # final_rwth
    chk_dir = os.path.join(out_dir, nameofrun)
    best_path = os.path.join(chk_dir, f"{nameofrun}_best.pt")
    best_cfg = os.path.join(chk_dir, f"{nameofrun}_cfg.yaml")
    if not os.path.isdir(chk_dir):
        os.makedirs(chk_dir)
    with open(best_cfg, 'w') as yml:
        yaml.dump(main_cfg, yml)

    """load dataset"""
    dataset = {}
    data_loader = {}
    dataset_list = zip(["train", "train_build", "dev", "test"], [True, True, False, False])
    for idx, (mode, train_flag) in enumerate(dataset_list):
        feeder_arg = main_cfg['feeder_args']
        feeder_arg['prefix'] = main_cfg['dataset_root']
        feeder_arg['mode'] = mode.split("_")[0]
        if mode == "train_build":
            feeder_arg['build'] = True
        else:
            feeder_arg['build'] = False
        feeder_arg['transform_mode'] = train_flag
        dataset[mode] = BaseFeeder(gloss_dict=gloss_dict, **feeder_arg)
        data_loader[mode] = utils.parameters.build_dataloader(dataset[mode], mode, train_flag, main_cfg)
    
    """loss"""
    # loss = {'ctc': nn.CTCLoss(), 'bce': nn.BCEWithLogitsLoss()}
    loss = {'ctc': nn.CTCLoss(), 'bce': nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([2.4]).to(hyper_cfg['device']))}
    # reduction="none", zero_infinity=False

    """model"""
    model = CombinedModel_tconv(model_cfg, loss).to(hyper_cfg['device'])
    # model = CombinedModel_lstm(model_cfg, loss).to(hyper_cfg['device'])
    
    """load"""
    # state_dict = torch.load("/home/ubuntu/workspace/SLR_codes/rwth/checkpoint/tconv2_gcn1_pad2_4060/tconv2_gcn1_pad2_4060_best.pt")
    state_dict = torch.load("/home/ubuntu/workspace/SLR_codes/rwth/checkpoint/tconv2_gcn1_pad2_res18/tconv2_gcn1_pad2_res18_best.pt")
    model = utils.load_chk_point(model, state_dict)
    # rng.set_rng_state(state_dict["rng_state"])
    # start_ep = state_dict['epoch'] + 1
    
    """optimizer"""
    # optimizer = SGD(model.parameters(), lr=hyper_cfg['learning_rate'], momentum=0.9, nesterov=False, weight_decay=1e-4)
    optimizer = Adam(model.parameters(), lr=hyper_cfg['learning_rate'], weight_decay=1e-4)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60], gamma=0.2)

    """decoder"""
    decoder = utils.Decode(gloss_dict, model_cfg['num_classes'], 'beam')

    best_score = 100
    bin_dict = dict()
    for ep in range(start_ep, hyper_cfg['epochs']):
        # tr_loss, tr_acc, bin_dict, model = trainer.only_ctc_tr_2(data_loader['train'], model, decoder, optimizer, ep, hyper_cfg, chk_dir+"/", bin_dict)
        te_acc, bin_dict = trainer.only_ctc_te(data_loader['test'], model, decoder, ep, hyper_cfg, chk_dir+"/", bin_dict)

        # dev_acc, bin_dict = trainer.only_ctc_te(data_loader['test'], model, decoder, ep, hyper_cfg, chk_dir+"/", bin_dict)
        dev_acc, bin_dict = trainer.only_ctc_dev(data_loader['dev'], model, decoder, ep, hyper_cfg, chk_dir+"/", bin_dict)
        # break
        # tr_loss, tr_acc, bin_dict, model = trainer.only_lstm_tr(data_loader, model, decoder, optimizer, ep, hyper_cfg, chk_dir+"/", bin_dict)
        # te_acc, bin_dict = trainer.only_lstm_te(data_loader['dev'], model, decoder, ep, hyper_cfg, chk_dir+"/", bin_dict)
        # scheduler.step()

        # wan_run.log({"train loss": tr_loss, "train acc": tr_acc, "test acc": dev_acc})
    
        '''checkpoint'''
        # if best_score > dev_acc:
        #     best_score = dev_acc
        #     print(f"best {ep} epoch has been saved")
        #     torch.save({
        #         'epoch': ep,
        #         'model_state_dict': model.state_dict(),
        #         'rng_state': rng.save_rng_state()
        #     }, best_path)

if __name__=="__main__":
    run()
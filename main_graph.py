__author__ = "Jie Lei"

import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
# from torch.utils.data import DataLoader
# from model.HeteroDataLoader import DataLoader
from torch_geometric.loader import DataLoader

from tqdm import tqdm
# from tensorboardX import SummaryWriter

from model.tvqa_abc_graph import ABC
from tvqa_dataset_graph import TVQADataset, pad_collate, preprocess_inputs
import itertools
# from model.tvqa_abc import ABC
# from tvqa_dataset import TVQADataset, pad_collate, preprocess_inputs

from config import BaseOptions

def train(opt, dset, model, criterion, optimizer, epoch, previous_best_acc, train_loader):
    dset.set_mode("train")
    model.train()

    train_loss = []
    valid_acc_log = ["batch_idx\tacc"]
    train_corrects = []
    torch.set_grad_enabled(True)
    
    max_pad = {
        "q": opt.max_q_l, "a0": opt.max_q_l, "a1": opt.max_q_l, "a2": opt.max_q_l, "a3": opt.max_q_l, "a4": opt.max_q_l,
        "sub": opt.max_sub_l,
        "vcpt": opt.max_vcpt_l,
        "vid": opt.max_vid_l,
    }
    
    for batch_idx, (x,targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # model_inputs, targets, _ = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, opt.max_vid_l, device=opt.device)
        
        # TODO: change model_inputs to be a graph here!
        # forward(self, q, q_l, a0, a0_l, a1, a1_l, a2, a2_l, a3, a3_l, a4, a4_l, sub, sub_l, vcpt, vcpt_l, vid, vid_l):
        
        x = x.to('cuda')
        targets = targets.to('cuda')

        mod_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub", "vcpt", 'vid']

        if not opt.vid_feat_flag:
            mod_keys = mod_keys[:-1]

        # reshape into format model expects: currently each is #nodes,dim; but because of padding this should be bs,max_seq_len,dim
        bs = len(x['q']['batch'].unique())
        # for k in mod_keys:
            # x[k]['x'].reshape(bs,max_pad[k],-1) # if this fails, it's because one of the samples has an empty modality
        
        model_inputs = [ ( x[k]['x'].reshape(bs,max_pad[k],-1), x[f'{k}_l'] ) for k in mod_keys]
        model_inputs = list(itertools.chain.from_iterable(model_inputs))

        if not opt.vid_feat_flag:
            model_inputs = [*model_inputs, None, None]
            
        outputs = model(*model_inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
        optimizer.step()
        # scheduler.step()

        # measure accuracy and record loss
        train_loss.append(loss.item())
        pred_ids = outputs.data.max(1)[1]
        train_corrects += pred_ids.eq(targets.data).cpu().numpy().tolist()
        if batch_idx % opt.log_freq == 0:
            niter = epoch * len(train_loader) + batch_idx

            train_acc = sum(train_corrects) / float(len(train_corrects))
            train_loss = sum(train_loss) / float(len(train_corrects))
            # opt.writer.add_scalar("Train/Acc", train_acc, niter)
            # opt.writer.add_scalar("Train/Loss", train_loss, niter)

            # Test
            valid_acc, valid_loss = validate(opt, dset, model, mode="valid")
            # opt.writer.add_scalar("Valid/Loss", valid_loss, niter)

            valid_log_str = "%02d\t%.4f" % (batch_idx, valid_acc)
            valid_acc_log.append(valid_log_str)
            if valid_acc > previous_best_acc:
                previous_best_acc = valid_acc
                torch.save(model.state_dict(), os.path.join(opt.results_dir, "best_valid.pth"))
            print(" Train Epoch %d loss %.4f acc %.4f Val loss %.4f acc %.4f"
                  % (epoch, train_loss, train_acc, valid_loss, valid_acc))

            # reset to train
            torch.set_grad_enabled(True)
            model.train()
            dset.set_mode("train")
            train_corrects = []
            train_loss = []

        # torch.cuda.empty_cache()

        if opt.debug:
            break

    # additional log
    with open(os.path.join(opt.results_dir, "valid_acc.log"), "a") as f:
        f.write("\n".join(valid_acc_log) + "\n")

    return previous_best_acc


def validate(opt, dset, model, mode="valid"):
    dset.set_mode(mode)
    torch.set_grad_enabled(False)
    model.eval()
    valid_loader = DataLoader(dset, batch_size=opt.test_bsz, shuffle=False, collate_fn=pad_collate)

    valid_qids = []
    valid_loss = []
    valid_corrects = []

    max_pad = {
        "q": opt.max_q_l, "a0": opt.max_q_l, "a1": opt.max_q_l, "a2": opt.max_q_l, "a3": opt.max_q_l, "a4": opt.max_q_l,
        "sub": opt.max_sub_l,
        "vcpt": opt.max_vcpt_l,
        "vid": opt.max_vid_l,
    }
    
    for _, (x,targets) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        x = x.to('cuda')
        targets = targets.to('cuda')

        mod_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub", "vcpt", 'vid']
        if not opt.vid_feat_flag:
            mod_keys = mod_keys[:-1]

        # reshape into format model expects: currently each is #nodes,dim; but because of padding this should be bs,max_seq_len,dim
        bs = len(x['q']['batch'].unique())
        for k in mod_keys:
            x[k]['x'].reshape(bs,max_pad[k],-1) # if this fails, it's because one of the samples has an empty modality
        
        model_inputs = [ ( x[k]['x'].reshape(bs,max_pad[k],-1), x[f'{k}_l'] ) for k in mod_keys]
        model_inputs = list(itertools.chain.from_iterable(model_inputs))

        if not opt.vid_feat_flag:
            model_inputs = [*model_inputs, None, None]

        outputs = model(*model_inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        # valid_qids += [int(x) for x in qids]
        valid_loss.append(loss.item())
        pred_ids = outputs.data.max(1)[1]
        valid_corrects += pred_ids.eq(targets.data).cpu().numpy().tolist()

        if opt.debug:
            break

    valid_acc = sum(valid_corrects) / float(len(valid_corrects))
    valid_loss = sum(valid_loss) / float(len(valid_corrects))
    return valid_acc, valid_loss


if __name__ == "__main__":
    torch.manual_seed(2018)
    opt = BaseOptions().parse()
    # writer = SummaryWriter(opt.results_dir)
    # opt.writer = writer

    dset = TVQADataset(opt)
    opt.vocab_size = len(dset.word2idx)
    model = ABC(opt)
    if not opt.no_glove:
        model.load_embedding(dset.vocab_embedding)

    model.to(opt.device)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss(size_average=False).to(opt.device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=opt.lr, weight_decay=opt.wd)

    dset.set_mode("train")
    train_loader = DataLoader(dset, batch_size=opt.bsz, shuffle=False, num_workers=opt.num_workers)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.05, 
    #     anneal_strategy='linear', final_div_factor=10, max_lr=5e-4, 
    #     total_steps=len(train_loader) * opt.n_epoch + 1)

    best_acc = 0.
    early_stopping_cnt = 0
    early_stopping_flag = False
    for epoch in range(opt.n_epoch):
        if not early_stopping_flag:
            # train for one epoch, valid per n batches, save the log and the best model
            cur_acc = train(opt, dset, model, criterion, optimizer, epoch, best_acc, train_loader)

            # remember best acc
            is_best = cur_acc > best_acc
            best_acc = max(cur_acc, best_acc)
            if not is_best:
                early_stopping_cnt += 1
                if early_stopping_cnt >= opt.max_es_cnt:
                    early_stopping_flag = True
        else:
            print("early stop with valid acc %.4f" % best_acc)
            # opt.writer.export_scalars_to_json(os.path.join(opt.results_dir, "all_scalars.json"))
            # opt.writer.close()
            break  # early stop break

        if opt.debug:
            break


__author__ = "Jie Lei"

import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
# from torch.utils.data import DataLoader
from model.HeteroDataLoader import DataLoader
# from torch_geometric.loader import DataLoader

from tqdm import tqdm
# from tensorboardX import SummaryWriter

from model.tvqa_abc_graph import ABC
from tvqa_dataset_graph import TVQADataset, pad_collate, preprocess_inputs
import itertools
# from model.tvqa_abc import ABC
# from tvqa_dataset import TVQADataset, pad_collate, preprocess_inputs

from config import BaseOptions

torch.cuda.set_device(0)
import wandb

def train(opt, dset, model, criterion, main_optimizer, graph_optimizer, epoch, previous_best_acc, train_loader, scheduler):
    dset.set_mode("train")
    model.train()

    train_loss = []
    valid_acc_log = ["batch_idx\tacc"]
    train_corrects = []
    torch.set_grad_enabled(True)
    
    max_pad = {
        "q": gc['max_q_l'], "a0": gc['max_q_l'], "a1": gc['max_q_l'], "a2": gc['max_q_l'], "a3": gc['max_q_l'], "a4": gc['max_q_l'],
        "sub": gc['max_sub_l'],
        "vcpt": gc['max_vcpt_l'],
        "vid": gc['max_vid_l'],
    }

    # results = {
    #     'lr': [gc['lr']],
    #     'graph_lr': [gc['graph_lr']],
    #     'acc': np.arange(10),
    #     'loss': np.random.random(10),
    # }
    # return results
    # wandb.watch(model, criterion, log="all")

    for batch_idx, (x,targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # model_inputs, targets, _ = preprocess_inputs(batch, gc['max_sub_l'], gc['max_vcpt_l'], gc['max_vid_l'], device=gc['device'])
        
        # TODO: change model_inputs to be a graph here!
        # forward(self, q, q_l, a0, a0_l, a1, a1_l, a2, a2_l, a3, a3_l, a4, a4_l, sub, sub_l, vcpt, vcpt_l, vid, vid_l)
        
        x = x.cuda()
        targets = targets.cuda()

        mod_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub", "vcpt", 'vid']
        graph_keys = ['sub', 'q', 'a0', 'a1', 'a2', 'a3', 'a4']

        if not gc['vid_feat_flag']:
            mod_keys = mod_keys[:-1]

        # reshape into format model expects: currently each is #nodes,dim; but because of padding this should be bs,max_seq_len,dim
        bs = len(x['q']['batch'].unique())
        # for k in mod_keys:
            # x[k]['x'].reshape(bs,max_pad[k],-1) # if this fails, it's because one of the samples has an empty modality
        
        # try:
        model_inputs = []
        for k in mod_keys:
            if k in graph_keys:
                model_inputs.append((None, None))
            else:
                model_inputs.append(( x[k]['x'].reshape(bs,max_pad[k],-1), x[f'{k}_l'] ))
        model_inputs = list(itertools.chain.from_iterable(model_inputs))
    
        # except:
        #     hi=2
        #     assert False

        if not gc['vid_feat_flag']:
            model_inputs = [*model_inputs, None, None]

        # add full graph input
        model_inputs.append(x)
            
        try:
            outputs = model(*model_inputs)
        except:
            print('Overflowed memory bounds')
            torch.cuda.empty_cache()
            continue
        
        loss = criterion(outputs, targets)
        # main_optimizer.zero_grad()
        graph_optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gc['clip'])
        
        # main_optimizer.step()
        graph_optimizer.step()
        scheduler.step()

        # measure accuracy and record loss
        train_loss.append(loss.item())
        pred_ids = outputs.data.max(1)[1]
        train_corrects += pred_ids.eq(targets.data).cpu().numpy().tolist()
        train_acc = sum(train_corrects) / float(len(train_corrects))
        train_loss = sum(train_loss) / float(len(train_corrects))

        if gc['train_steps'] % gc['log_freq'] == 0:
            niter = epoch * len(train_loader) + batch_idx

            # gc['writer'].add_scalar("Train/Acc", train_acc, niter)
            # gc['writer'].add_scalar("Train/Loss", train_loss, niter)

            # Test
            # if not gc['debug']:
            #     valid_acc, valid_loss = validate(gc, dset, model, mode="valid")
            #     # gc['writer'].add_scalar("Valid/Loss", valid_loss, niter)

            #     valid_log_str = "%02d\t%.4f" % (batch_idx, valid_acc)
            #     valid_acc_log.append(valid_log_str)
            #     if valid_acc > previous_best_acc:
            #         previous_best_acc = valid_acc
            #         torch.save(model.state_dict(), os.path.join(gc['results_dir'], "best_valid.pth"))
            #     print(" Train Epoch %d loss %.4f acc %.4f Val loss %.4f acc %.4f" % (epoch, train_loss, train_acc, valid_loss, valid_acc))
            #     wandb.log({'train_loss': train_loss, 'train_acc': train_acc})

            # else:
            print(" Train Epoch %d loss %.4f acc %.4f" % (epoch, train_loss, train_acc))
            wandb.log({'train_loss': train_loss, 'train_acc': train_acc})
            if train_loss < .1:
                exit()
            # reset to train
            torch.set_grad_enabled(True)
            model.train()
            dset.set_mode("train")

        # torch.cuda.empty_cache()
        train_corrects = []
        train_loss = []
        # if gc['debug']:
        #     break
        # if batch_idx > 3000:
        #     break
        gc['train_steps'] += 1
        if batch_idx >= gc['num_batches']-1 and gc['num_batches'] > 0:
            break

    # additional log
    with open(os.path.join(gc['results_dir'], "valid_acc.log"), "a") as f:
        f.write("\n".join(valid_acc_log) + "\n")

    # valid_acc, valid_loss = validate(gc, dset, model, mode="valid")
    return train_acc, 0


def validate(gc, dset, model, mode="valid"):
    dset.set_mode(mode)
    torch.set_grad_enabled(False)
    model.eval()
    valid_loader = DataLoader(dset, batch_size=gc['test_bs'], shuffle=False, collate_fn=pad_collate)

    valid_qids = []
    valid_loss = []
    valid_corrects = []

    max_pad = {
        "q": gc['max_q_l'], "a0": gc['max_q_l'], "a1": gc['max_q_l'], "a2": gc['max_q_l'], "a3": gc['max_q_l'], "a4": gc['max_q_l'],
        "sub": gc['max_sub_l'],
        "vcpt": gc['max_vcpt_l'],
        "vid": gc['max_vid_l'],
    }
    
    for _, (x,targets) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        x = x.cuda()
        targets = targets.cuda()

        mod_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub", "vcpt", 'vid']
        graph_keys = ['sub', 'q', 'a0', 'a1', 'a2', 'a3', 'a4']

        if not gc['vid_feat_flag']:
            mod_keys = mod_keys[:-1]

        # reshape into format model expects: currently each is #nodes,dim; but because of padding this should be bs,max_seq_len,dim
        bs = len(x['q']['batch'].unique())
        # for k in mod_keys:
            # x[k]['x'].reshape(bs,max_pad[k],-1) # if this fails, it's because one of the samples has an empty modality
        
        # try:
        model_inputs = []
        for k in mod_keys:
            if k in graph_keys:
                model_inputs.append((None, None))
            else:
                model_inputs.append(( x[k]['x'].reshape(bs,max_pad[k],-1), x[f'{k}_l'] ))
        model_inputs = list(itertools.chain.from_iterable(model_inputs))
    
        # except:
        #     hi=2
        #     assert False

        if not gc['vid_feat_flag']:
            model_inputs = [*model_inputs, None, None]

        # add full graph input
        model_inputs.append(x)

        outputs = model(*model_inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        # valid_qids += [int(x) for x in qids]
        valid_loss.append(loss.item())
        pred_ids = outputs.data.max(1)[1]
        valid_corrects += pred_ids.eq(targets.data).cpu().numpy().tolist()

        if gc['debug']:
            break

    valid_acc = sum(valid_corrects) / float(len(valid_corrects))
    valid_loss = sum(valid_loss) / float(len(valid_corrects))
    return valid_acc, valid_loss


def main(_gc):
    torch.manual_seed(2018)
    # opt = BaseOptions().parse()
    global gc; gc = _gc

    wandb.init(
        project="tvqa_graphnn" if not gc['sweep'] else None, 
        entity="socialiq" if not gc['sweep'] else None, 
        name=gc['name'] if not (gc['name']=='' or gc['sweep']) else None,
        config={k: v for k,v in gc.items() if k in ['bs', 'graph_lr', 'n_epoch', 'hidden_dim', '']}
    )

    gc['input_streams'] = gc['input_streams'].split(' ')
    # writer = SummaryWriter(gc['results_dir'])
    # gc['writer'] = writer

    gc['normalize_v'] = not gc['no_normalize_v']
    gc['device'] = torch.device("cuda:%d" % gc['device'] if gc['device'] >= 0 else "cpu")
    gc['with_ts'] = not gc['no_ts']
    gc['input_streams'] = [] if gc['input_streams'] is None else gc['input_streams']
    gc['vid_feat_flag'] = True if "imagenet" in gc['input_streams'] else False
    gc['h5driver'] = None if gc['no_core_driver'] else "core"
    gc['results_dir'] = gc['out_dir']
    
    dset = TVQADataset(gc)
    gc['vocab_size'] = len(dset.word2idx)
    model = ABC(gc)
    if not gc['no_glove']:
        model.load_embedding(dset.vocab_embedding)

    model.cuda()
    cudnn.benchmark = True
    global criterion
    criterion = nn.CrossEntropyLoss(size_average=False).cuda()

    # graph_optimizer = torch.optim.AdamW(model.hetero_gnn.parameters(),lr=gc['graph_lr'])
    graph_optimizer = torch.optim.AdamW(model.parameters(),lr=gc['graph_lr'])
    # other_params = list(map(lambda elt: elt[1], filter(lambda np: ('hetero_gnn' not in np[0]) and np[1].requires_grad, model.named_parameters())))
    # main_optimizer = torch.optim.Adam(other_params, lr=gc['lr'], weight_decay=gc['wd'])
    main_optimizer = None

    dset.set_mode("train")
    train_loader = DataLoader(dset, batch_size=gc['bs'], shuffle=False, num_workers=gc['num_workers'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(graph_optimizer, pct_start=0.05, anneal_strategy='linear', final_div_factor=10, max_lr=5e-4, total_steps=len(train_loader) * gc['n_epoch'] + 1)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.05, 
    #     anneal_strategy='linear', final_div_factor=10, max_lr=5e-4, 
    #     total_steps=len(train_loader) * gc['n_epoch'] + 1)

    # results = {
    #     'lr': [gc['lr']],
    #     'graph_lr': [gc['graph_lr']],
    #     'acc': np.arange(10),
    #     'loss': np.random.random(10),
    # }
    # return results

    best_acc = 0.
    early_stopping_cnt = 0
    early_stopping_flag = False
    results = {
        'train_accs': [],
        'valid_accs': []
    }
    gc['train_steps'] = 0
    for epoch in range(gc['n_epoch']):
        train_acc, valid_acc = train(gc, dset, model, criterion, main_optimizer, graph_optimizer, epoch, best_acc, train_loader, scheduler)
        continue

        if not early_stopping_flag or gc['debug']:
            # train for one epoch, valid per n batches, save the log and the best model
            train_acc, valid_acc = train(gc, dset, model, criterion, main_optimizer, graph_optimizer, epoch, best_acc, train_loader, scheduler)
            results['train_accs'].append(train_acc)
            results['valid_accs'].append(valid_acc)
            cur_acc = valid_acc

            # remember best acc
            is_best = cur_acc > best_acc
            best_acc = max(cur_acc, best_acc)
            if not is_best:
                early_stopping_cnt += 1
                if early_stopping_cnt >= gc['max_es_cnt']:
                    early_stopping_flag = True
        else:
            print("early stop with valid acc %.4f" % best_acc)
            # gc['writer'].export_scalars_to_json(os.path.join(gc['results_dir'], "all_scalars.json"))
            # gc['writer'].close()
            break  # early stop break

        # if gc['debug']:
        #     break
    return results
    

if __name__ == '__main__':
    import sys; sys.path.append('/work/awilf/utils/'); from alex_utils import *
    from defaults import defaults
    main_wrapper(main, defaults, results=True)

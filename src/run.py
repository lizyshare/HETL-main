import argparse
import logging
import torch.nn
import torch.optim
import os
import sys
import time
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from utils import *
from evaluation import *
from transformer import *

parser = argparse.ArgumentParser()

# Dataset information
parser.add_argument("--dataset", type=str, default="jf17k") # 数据集名字
parser.add_argument("--vocab_size", type=int, default=29148) # 数据量大小
parser.add_argument("--vocab_file", type=str, default="./data/jf17k/vocab.txt") # 词汇表文件
parser.add_argument("--train_file", type=str, default="./data/jf17k/train.json")
parser.add_argument("--test_file", type=str, default="./data/jf17k/test.json")
parser.add_argument("--ground_truth_file", type=str, default="./data/jf17k/all.json") # 所有实体关系文件
parser.add_argument("--num_relations", type=int, default=501)
parser.add_argument("--max_arity", type=int, default=6) # 最大参数数量
parser.add_argument("--max_seq_len", type=int, default=11) # 最大序列长度 2*max_arity-1
parser.add_argument("--num_workers", type=int, default=1)

# Hyperparameter
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--hyper_layers", type=int,  default=2) # 超图注意力层层数
parser.add_argument("--hyper_dropout", type=float, default=0.9) # 超图注意力层丢失率
parser.add_argument("--hyper_activation", type=str, default="elu")
parser.add_argument("--hyper_heads", type=int,  default=4) # 超图注意力层头数
parser.add_argument("--trans_layers", type=int,  default=12) # Transformer层数
parser.add_argument("--trans_dropout", type=float, default=0.2) # Transformer层丢失率
parser.add_argument("--trans_heads", type=int, default=4) # Transformer层注意力头数
parser.add_argument("--decoder_activation", type=str, default="gelu")
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--weight_decay", type=float, default=0.01) # 权重衰减率
parser.add_argument("--entity_soft", type=float, default=0.9) # 实体软标签
parser.add_argument("--relation_soft", type=float, default=0.0) # 关系软标签
parser.add_argument("--hyperedge_dropout", type=float, default=0.85) # 超边学习的丢失率
parser.add_argument("--remove_mask", type=bool, default=False)

# others for training
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--warmup_proportion", type=float, default=0.05)
parser.add_argument("--device", type=str, default="0123")

# Ablation experiment
parser.add_argument("--use_edge", type=bool, default=True)
parser.add_argument("--use_node", type=bool, default=True)
parser.add_argument("--use_hypergraph", type=bool, default=True)
parser.add_argument("--use_attention", type=bool, default=True)

# directory position settings
parser.add_argument("--result_save_dir", type=str, default="results")
parser.add_argument("--ckpt_save_dir", type=str, default="ckpts")

args = parser.parse_args()
args.num_entities = args.vocab_size - args.num_relations - 2
if not os.path.exists(args.result_save_dir):
    os.mkdir(args.result_save_dir)
if not os.path.exists(args.ckpt_save_dir):
    os.mkdir(args.ckpt_save_dir)
dir_name = os.path.join(args.result_save_dir,args.dataset)

# 保存结果和模型
dir_name_result = os.path.join(args.result_save_dir,args.dataset)
if not os.path.exists(dir_name_result):
    os.mkdir(dir_name_result)

dir_name_ckpt = os.path.join(args.ckpt_save_dir,args.dataset)
if not os.path.exists(dir_name_ckpt):
    os.mkdir(dir_name_ckpt)

logging.basicConfig(
    format='%(asctime)s  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    filename=f'{dir_name_result}/train.log',
    filemode="w",
    level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
def main(limit=1e9):
    # prepare GPU or GPUs
    device = torch.device(f"cuda:{args.device[0]}")
    devices = []
    for i in range(len(args.device)):
        devices.append(torch.device(f"cuda:{args.device[i]}"))

    vocabulary = Vocabulary(
        vocab_file=args.vocab_file,
        num_relations=args.num_relations,
        num_entities=args.num_entities)
    train_examples, _ = read_examples(args.train_file, args.max_arity)
    test_examples, _ = read_examples(args.test_file, args.max_arity)
    train_dataset = MultiDataset(vocabulary, train_examples, args.max_arity, args.max_seq_len)
    test_dataset = MultiDataset(vocabulary, test_examples, args.max_arity, args.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                shuffle=False, num_workers=1)
    max_train_steps = args.epoch * len(train_loader)
    graph = build_graph(vocabulary, train_examples, args.hyperedge_dropout, device)
    
    if len(devices) > 1:
        model = torch.nn.DataParallel(Transformer(graph, args.vocab_size, args.trans_layers, args.hyper_layers, args.hidden_dim,
                                                        args.trans_heads, args.hyper_heads, args.use_hypergraph, args.trans_dropout, args.hyper_dropout,
                                                        args.decoder_activation, args.hyper_activation, args.use_edge, args.remove_mask, args.use_node), device_ids=devices)
        model.to(device)
    else:
        model = Transformer(graph, args.vocab_size, args.trans_layers, args.hyper_layers, args.hidden_dim, args.trans_heads, args.hyper_heads,
                                args.use_hypergraph, args.trans_dropout, args.hyper_dropout, args.decoder_activation, args.hyper_activation,
                                args.use_edge, args.remove_mask, args.use_node).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=max_train_steps, 
                            pct_start=args.warmup_proportion, anneal_strategy="linear", cycle_momentum=False)
    limit = min(args.epoch, limit)
    training_time = int(0)
    for epoch in range(limit):
        time_start_epoch = time.time()
        for item in tqdm(train_loader):
            model.train()
            item = (i.to(device) for i in item)
            input_ids, input_mask, mask_position, mask_label, mask_output, edge_labels, query_type = item
            result = model(input_ids, input_mask, mask_position, mask_output, edge_labels)
            entities, relations = (query_type == 1), (query_type == -1)

            label_entity = mask_output[entities] * (args.entity_soft / (args.num_entities - 1))
            label_entity[torch.arange(label_entity.shape[0]), mask_label[entities]] = 1 - args.entity_soft
            label_relation = mask_output[relations] * (args.relation_soft / (args.num_relations - 1))
            label_relation[torch.arange(label_relation.shape[0]), mask_label[relations]] = 1 - args.relation_soft            
            loss1 = torch.nn.functional.cross_entropy(result[entities], label_entity, reduction='none')
            loss2 = torch.nn.functional.cross_entropy(result[relations], label_relation, reduction='none')
            loss = torch.cat((loss1, loss2)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        time_end_epoch = time.time()
        training_time += round(time_end_epoch - time_start_epoch)
        hours, minutes, seconds = calculate_training_time(training_time)
        logger.info(f"epoch: {epoch}\tlr: {scheduler.get_last_lr()[0]:.6f}\ttrain time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        if epoch % 5 == 0 or epoch == limit - 1:
            eval_performance = predict(
                model=model,
                test_loader=test_loader,
                all_features=test_dataset.features,
                vocabulary=vocabulary,
                device=device)
            show_perforamance(eval_performance)
            Exp_name_params = [
                args.dataset,
                "epoch_"+str(epoch),
            ]
            torch.save(
                model.state_dict(),
                os.path.join(dir_name_ckpt, "_".join(Exp_name_params)+".ckpt"),
                )
def calculate_training_time(training_time: int):
    minutes, seconds = divmod(training_time, 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds

def predict(model, test_loader, all_features, vocabulary, device):
    eval_result_file = os.path.join(dir_name_result, "eval_result.json")

    gt_dict = generate_ground_truth(
        ground_truth_path=args.ground_truth_file,
        vocabulary=vocabulary,
        max_arity=args.max_arity,
        max_seq_length=args.max_seq_len)

    step = 0
    hyper_idx = 0
    ent_lst = []
    rel_lst = []
    _2_r_lst = []
    _2_ht_lst = []
    _n_r_lst = []
    _n_ht_lst = []
    _n_a_lst = []
    _n_v_lst = []
    model.eval()
    with torch.no_grad():
        for item in tqdm(test_loader):
            item = (i.to(device) for i in item)
            input_ids, input_mask, mask_position, mask_label, mask_output, edge_labels, query_type = item

            output = model(input_ids, input_mask, mask_position, mask_output, edge_labels)

            batch_results = output.cpu().numpy()
            ent_ranks, rel_ranks, _2_r_ranks, _2_ht_ranks, \
            _n_r_ranks, _n_ht_ranks, _n_a_ranks, _n_v_ranks = batch_evaluation(
                hyper_idx, batch_results, all_features, gt_dict)
            ent_lst.extend(ent_ranks)
            rel_lst.extend(rel_ranks)
            _2_r_lst.extend(_2_r_ranks)
            _2_ht_lst.extend(_2_ht_ranks)
            _n_r_lst.extend(_n_r_ranks)
            _n_ht_lst.extend(_n_ht_ranks)
            _n_a_lst.extend(_n_a_ranks)
            _n_v_lst.extend(_n_v_ranks)
            step += 1
            hyper_idx += output.size(0)

    eval_result = compute_metrics(
        ent_lst=ent_lst,
        rel_lst=rel_lst,
        _2_r_lst=_2_r_lst,
        _2_ht_lst=_2_ht_lst,
        _n_r_lst=_n_r_lst,
        _n_ht_lst=_n_ht_lst,
        _n_a_lst=_n_a_lst,
        _n_v_lst=_n_v_lst,
        eval_result_file=eval_result_file
    )
    return eval_result

def show_perforamance(eval_performance):
    def pad(x):
        return x + (10 - len(x)) * ' '
    all_entity = f"{pad('ENTITY')}\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        eval_performance['entity']['mrr'],
        eval_performance['entity']['hits1'],
        eval_performance['entity']['hits3'],
        eval_performance['entity']['hits5'],
        eval_performance['entity']['hits10'])

    all_relation = f"{pad('RELATION')}\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        eval_performance['relation']['mrr'],
        eval_performance['relation']['hits1'],
        eval_performance['relation']['hits3'],
        eval_performance['relation']['hits5'],
        eval_performance['relation']['hits10'])

    all_ht = f"{pad('HEAD/TAIL')}\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        eval_performance['ht']['mrr'],
        eval_performance['ht']['hits1'],
        eval_performance['ht']['hits3'],
        eval_performance['ht']['hits5'],
        eval_performance['ht']['hits10'])

    all_r = f"{pad('PRIMARY_R')}\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        eval_performance['r']['mrr'],
        eval_performance['r']['hits1'],
        eval_performance['r']['hits3'],
        eval_performance['r']['hits5'],
        eval_performance['r']['hits10'])

    logger.info("\n-------- Evaluation Performance --------\n%s\n%s\n%s\n%s\n%s" % (
        "\t".join([pad("TASK"), "MRR", "Hits@1", "Hits@3", "Hits@5", "Hits@10"]),
        all_ht, all_r, all_entity, all_relation))

if __name__ == '__main__':
    logger.info('-----------  Configuration Arguments -----------')
    for arg, value in vars(args).items():
        logger.info('%s: %s' % (arg, value))
    logger.info('------------------------------------------------')
    main()

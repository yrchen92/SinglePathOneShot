import os
import sys
import torch
import argparse
import random
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import time
import logging
import argparse
from network import ShuffleNetV2_OneShot
from utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_lastest_model, get_parameters
from flops import get_cand_flops
from torch.nn import CrossEntropyLoss
from pytorch_transformers import AdamW

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from utils_glue import compute_metrics, convert_examples_to_features, output_modes, processors
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

logger = logging.getLogger(__name__)



ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

def fix_parameters(model):
    for name, param in model.named_parameters():
	    param.requires_grad = False

get_random_cand = lambda:tuple(np.random.randint(9) for i in range(6))
def get_uniform_sample_cand(*,timeout=500):
    return get_random_cand()
    flops_l, flops_r, flops_step = 290, 360, 10
    bins = [[i, i+flops_step] for i in range(flops_l, flops_r, flops_step)]
    idx = np.random.randint(len(bins))
    l, r = bins[idx]
    for i in range(timeout):
        cand = get_random_cand()
        if l*1e6 <= get_cand_flops(cand) <= r*1e6:
            return cand
    return get_random_cand()

class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data

def get_args():
    parser = argparse.ArgumentParser("ShuffleNetV2_OneShot")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                    help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--model_type", default=None, type=str, required=True,
                    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--task_name", default=None, type=str, required=True,   
                    help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")
    parser.add_argument('--hidden_size', type=int, default=128,
                    help="hidden state size")
    parser.add_argument("--no_cuda", action='store_true',
                    help="Avoid using CUDA when available")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")

    parser.add_argument('--auto_continue', type=bool, default=False, help='report frequency')
    parser.add_argument('--display_interval', type=int, default=20, help='report frequency')
    parser.add_argument('--val_interval', type=int, default=20, help='report frequency')
    parser.add_argument('--save_interval', type=int, default=20, help='report frequency')

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                            "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--no_segment', action='store_true',
                    help="force to remove segmentation ids")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")

    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--total_iters', type=int, default=150000, help='total iters')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='path for saving trained models')
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')

    # hyper-parameter search
    parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
    parser.add_argument("--available_gpus",
                        default='0,1,2,3,4,5',
                        type=str,
                        help="available_gpus")
    parser.add_argument("--need_gpus",
                        default=2,
                        type=int,
                        help="need_gpus")
    parser.add_argument("--conf_file",
                        default='./conf.json',
                        type=str,
                        help="seach space configuration")
    parser.add_argument("--job_id",
                        default=0,
                        type=int,
                        help="job id")

    args = parser.parse_args()
    return args

def load_and_cache_examples(args, task, tokenizer, data_type="train"):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = []
        if data_type == "train":
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == "dev":
            examples = processor.get_dev_examples(args.data_dir)
        elif data_type == "test":
            examples = processor.get_test_examples(args.data_dir)
        else:
            raise KeyError(str(task))
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def cal_logit_loss(student_predicts, teacher_predicts):
    student_likelihood = torch.nn.functional.log_softmax(student_predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(teacher_predicts, dim=-1)
    return (- targets_prob * student_likelihood).mean()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    args = get_args()

    args.output_dir = os.path.join(args.output_dir, "jobs", str(args.job_id))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
                        # level = logging.INFO)
    t = time.time()
    local_time = time.localtime(t)
    log_dir = os.path.join(args.output_dir, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_format = '[%(asctime)s] %(message)s'
    fh = logging.FileHandler(os.path.join('{}/train_{}_r{}{:02}{}'.format(log_dir, args.local_rank, local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if args.local_rank in [-1, 0]:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARN)

    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    t_model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    fix_parameters(t_model)

    # if args.local_rank == 0:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    t_model.to(args.device)

    # train
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type="train")
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    train_dataprovider = DataIterator(train_dataloader)
    # if args.max_steps > 0:
    #     args.total_iters = args.max_steps
    #     args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    # else:
    #     args.total_iters = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    
    # valid
    eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='dev')
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    eval_dataprovider = DataIterator(eval_dataloader)

    if args.task_name == "mnli":
        eval_task_names_mm = "mnli-mm"
        eval_outputs_dirs_mm = args.output_dir + '-MM'
        eval_dataset_mm = load_and_cache_examples(args, eval_task_names_mm, tokenizer, data_type='dev')
        if not os.path.exists(eval_outputs_dirs_mm) and args.local_rank in [-1, 0]:
            os.makedirs(eval_outputs_dirs_mm)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        # eval_sampler_mm = SequentialSampler(eval_dataset_mm) if args.local_rank == -1 else DistributedSampler(eval_dataset_mm)
        eval_sampler_mm = SequentialSampler(eval_dataset_mm)
        eval_dataloader_mm = DataLoader(eval_dataset_mm, sampler=eval_sampler_mm, batch_size=args.eval_batch_size)
    
    emb_w = t_model.bert.embeddings.word_embeddings.weight.clone().detach()
    model = ShuffleNetV2_OneShot(config.vocab_size, args.hidden_size, num_labels, emb_w)
    model.to(args.device)

    if args.local_rank in [-1, 0]:
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", args.total_iters)
        # Eval!
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        if args.task_name == "mnli": 
            logger.info("***** Running mnli-mm evaluation *****")
            logger.info("  Num examples = %d", len(eval_dataset_mm))
            logger.info("  Batch size = %d", args.eval_batch_size)
        logger.info('load data successfully')
        num_params = 0
        t_num_params = 0
        t_num_params_grad = 0
        for param in model.parameters():
            if param.requires_grad:
                num_params += param.numel()
        for param in t_model.parameters():
            t_num_params += param.numel()
            if param.requires_grad:
                t_num_params_grad += param.numel()
        num_params = num_params / 1e6
        t_num_params = t_num_params / 1e6
        t_num_params_grad = t_num_params_grad / 1e6
        logger.info("t_model have {:.2f}/{:.2f}M parameters in total\tmodel have {:.2f}M parameters in total\n".format(t_num_params_grad, t_num_params, num_params))    

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # optimizer = torch.optim.SGD(get_parameters(model),
    #                             lr=args.learning_rate,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    # criterion_smooth = CrossEntropyLabelSmooth(num_labels, 0.1)
    criterion_smooth = CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                    lambda step : (1.0-step/args.total_iters) if step <= args.total_iters else 0, last_epoch=-1)

    all_iters = 0
    if args.auto_continue:
        lastest_model, iters = get_lastest_model(args)
        if lastest_model is not None:
            if args.local_rank in [-1, 0]:
                logger.info('load from checkpoint {}'.format(lastest_model))
            all_iters = iters
            checkpoint = torch.load(lastest_model, map_location=None if not args.no_cuda else 'cpu')
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] 
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=True)
            for i in range(iters):
                scheduler.step()

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # t_model = torch.nn.DataParallel(t_model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
        # t_model = torch.nn.parallel.DistributedDataParallel(t_model, device_ids=[args.local_rank],
        #                                                   output_device=args.local_rank,
        #                                                   find_unused_parameters=True)

    args.optimizer = optimizer
    args.loss_function = criterion_smooth.cuda()
    args.scheduler = scheduler
    args.train_dataprovider = train_dataprovider
    args.eval_dataprovider = eval_dataprovider
    args.eval_dataloader = eval_dataloader

    if args.eval:
        # if args.eval_resume is not None:
        #     checkpoint = torch.load(args.eval_resume, map_location=None if not args.no_cuda else 'cpu')
        #     model.load_state_dict(checkpoint, strict=True)
        validate(t_model, device, args, all_iters=all_iters, eval_num=1)    
        exit(0)

    while all_iters < args.total_iters:
        all_iters = train(model, device, args, val_interval=args.val_interval, bn_process=False, all_iters=all_iters, t_model=t_model)
        validate(model, device, args, all_iters=all_iters, eval_num=10)

    # all_iters = train(model, device, args, val_interval=int(1280000/args.batch_size), bn_process=True, all_iters=all_iters)
    # save_checkpoint({'state_dict': model.state_dict(),}, args.total_iters, tag='bnps-')

def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters

def train(model, device, args, *, val_interval, bn_process=False, all_iters=None, t_model=None):
    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
    train_dataprovider = args.train_dataprovider

    t1 = time.time()
    model.train()
    if t_model is not None:
        t_model.train()
    optimizer.zero_grad()
    # scheduler.step()
    tr_loss = 0.0
    for iters in range(1, val_interval + 1):
        if bn_process:
            adjust_bn_momentum(model, iters)
        d_st = time.time()
        batch = train_dataprovider.next()
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] \
                                                and not args.no_segment else None,
                  'labels':         batch[3]}

        data_time = time.time() - d_st
        # get_random_cand = lambda:tuple(np.random.randint(4) for i in range(4))
        
        t_outputs = t_model(**inputs)
        inputs['architecture'] = get_uniform_sample_cand()
        output = model(**inputs)
        
        r_loss = loss_function(output, inputs['labels'])
        t_loss = cal_logit_loss(output, t_outputs[1].detach())
        loss = r_loss + t_loss
        if args.n_gpu > 1:
            r_loss = r_loss.mean()
            t_loss = t_loss.mean()
            loss = loss.mean() # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            r_loss = r_loss / args.gradient_accumulation_steps
            t_loss = t_loss / args.gradient_accumulation_steps
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        tr_loss += loss.item()

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        for p in model.parameters():
            if p.grad is not None and p.grad.sum() == 0:
                p.grad = None

        if iters % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            all_iters += 1

        if all_iters % args.display_interval == 0 and args.local_rank in [-1, 0]:
            preds = output.detach().cpu().numpy()
            label_ids = inputs['labels'].detach().cpu().numpy()
            if args.output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            elif args.output_mode == "regression":
                preds = np.squeeze(preds)
            result = compute_metrics(args.task_name, preds, label_ids)
            printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(all_iters, scheduler.get_lr()[0], loss.item()) + \
                        'r_loss = {:.6f},\tt_loss = {:.6f},\tgrad = {:.6f},\t'.format(r_loss.item(), t_loss.item(), total_norm) + \
                        'data_time = {:.6f},\ttrain_time = {:.6f}'.format(data_time, (time.time() - t1) / args.display_interval)
            for k, v in result.items():
                printInfo += ',\t{}={:.6f}'.format(k, v)
                        
            logger.info(printInfo)
            t1 = time.time()

        if all_iters % args.save_interval == 0:
            save_checkpoint(args, {'state_dict': model.state_dict(),}, all_iters)

    return all_iters

def validate(model, device, args, *, all_iters=None, eval_num=1):
    loss_function = args.loss_function
    eval_dataprovider = args.eval_dataprovider
    model.eval()
    max_val_iters = 250
    t1  = time.time()
    for _ in range(eval_num):
        preds = None
        labels = None
        eval_loss = 0
        eval_num = 0
        results = {}
        with torch.no_grad():
            # for _ in range(1, max_val_iters + 1):
            for batch in args.eval_dataloader:
                batch = eval_dataprovider.next()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] \
                                                        and not args.no_segment else None,
                        'labels':         batch[3]}
                
                # get_random_cand = lambda:tuple(np.random.randint(4) for i in range(4))
                inputs['architecture'] = get_random_cand()
                output = model(**inputs)

                if isinstance(output, tuple):
                    output = output[1]

                eval_num += inputs['input_ids'].size(0)
                loss = loss_function(output, inputs['labels'])
                eval_loss += loss.item()
                
                if preds is None:
                    preds = output.detach().cpu().numpy()
                    labels = inputs['labels'].detach().cpu().numpy()
                else:
                    preds = np.append(preds, output.detach().cpu().numpy(), axis=0)
                    labels = np.append(labels, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / eval_num
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(args.task_name, preds, labels)
        result['eval_loss'] = eval_loss
        results.update(result)
        if args.local_rank in [-1, 0]:
            logger.info("***** Eval {} results *****".format(args.task_name))
            log_info = 'Iter {}:'.format(all_iters)
            for key in sorted(result.keys()):
                log_info += '\teavl_{} = {:.6f}'.format(key, result[key])
            log_info += '\n{}'.format(str(inputs['architecture']))
        logger.info(log_info)

if __name__ == "__main__":
    main()


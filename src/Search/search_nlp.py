import os
import sys
import time
import glob
import numpy as np
import pickle
import torch
import logging
import argparse
import torch
import random

from network import ShuffleNetV2_OneShot
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

from tester import get_cand_err
from flops import get_cand_flops

from torch.autograd import Variable
import collections
import sys
sys.setrecursionlimit(10000)
import argparse

import functools
print = functools.partial(print, flush=True)

from utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors, logging_info)

logger = logging.getLogger(__name__)

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


class EvolutionSearcher(object):

    def __init__(self, args):
        self.args = args
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.flops_limit = args.flops_limit
        
        # Prepare GLUE task
        args.task_name = args.task_name.lower()
        if args.task_name not in processors:
            raise ValueError("Task not found: %s" % (args.task_name))
        processor = processors[args.task_name]()
        args.output_mode = output_modes[args.task_name]
        label_list = processor.get_labels()
        self.num_labels = len(label_list)     
        if args.t_model is not None:
            # Load pretrained model and tokenizer
            if args.local_rank not in [-1, 0]:
                torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
            args.t_model_type = args.t_model_type.lower()
            config_class, model_class, tokenizer_class = MODEL_CLASSES[args.t_model_type]
            config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=self.num_labels, finetuning_task=args.task_name)
            tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
            t_model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
            if args.local_rank == 0:
                torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
            t_model.to(args.device)
            logger.info("Training/evaluation parameters %s", args)
            # multi-gpu training (should be after apex fp16 initialization)
            if args.n_gpu > 1:
                t_model = torch.nn.DataParallel(t_model)
            # Distributed training (should be after apex fp16 initialization)
            if args.local_rank != -1:
                t_model = torch.nn.parallel.DistributedDataParallel(t_model, device_ids=[args.local_rank],
                                                                output_device=args.local_rank,
                                                                find_unused_parameters=True)
            self.t_model = t_model
        self.model = ShuffleNetV2_OneShot().to(args.device)
        if args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        if args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank,
                                                            find_unused_parameters=True)
        supernet_state_dict = torch.load(
            '../Supernet/models/checkpoint-latest.pth.tar')['state_dict']
        self.model.load_state_dict(supernet_state_dict)
        self.log_dir = args.log_dir
        self.checkpoint_name = os.path.join(self.log_dir, 'checkpoint.pth.tar')

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []

        self.nr_layer = 20
        self.nr_state = 4

    def save_checkpoint(self):
        if self.args.local_rank in [-1, 0]:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            info = {}
            info['memory'] = self.memory
            info['candidates'] = self.candidates
            info['vis_dict'] = self.vis_dict
            info['keep_top_k'] = self.keep_top_k
            info['epoch'] = self.epoch
            torch.save(info, self.checkpoint_name)
            logging_info(self.local_rank, 'save checkpoint to', self.checkpoint_name)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        info = torch.load(self.checkpoint_name)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        logging_info(self.local_rank, 'load checkpoint from', self.checkpoint_name)
        return True

    def is_legal(self, cand):
        assert isinstance(cand, tuple) and len(cand) == self.nr_layer
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False

        if 'flops' not in info:
            info['flops'] = get_cand_flops(cand)

        logging_info(self.local_rank, cand, info['flops'])

        if info['flops'] > self.flops_limit:
            logging_info(self.local_rank, 'flops limit exceed')
            return False

        info['err'] = get_cand_err(self.model, cand, self.args)

        info['visited'] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        logging_info(self.local_rank, 'select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random(self, num):
        logging_info(self.local_rank, 'random select ........')
        cand_iter = self.stack_random_cand(
            lambda: tuple(np.random.randint(self.nr_state) for i in range(self.nr_layer)))
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            logging_info(self.local_rank, 'random {}/{}'.format(len(self.candidates), num))
        logging_info(self.local_rank, 'random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging_info(self.local_rank, 'mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(choice(self.keep_top_k[k]))
            for i in range(self.nr_layer):
                if np.random.random_sample() < m_prob:
                    cand[i] = np.random.randint(self.nr_state)
            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging_info(self.local_rank, 'mutation {}/{}'.format(len(res), mutation_num))

        logging_info(self.local_rank, 'mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        logging_info(self.local_rank, 'crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = choice(self.keep_top_k[k])
            p2 = choice(self.keep_top_k[k])
            return tuple(choice([i, j]) for i, j in zip(p1, p2))
        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging_info(self.local_rank, 'crossover {}/{}'.format(len(res), crossover_num))

        logging_info(self.local_rank, 'crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        logging_info(self.local_rank, 'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            logging_info(self.local_rank, 'epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['err'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['err'])

            logging_info(self.local_rank, 'epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                logging_info(self.local_rank, 'No.{} {} Top-1 err = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['err']))
                ops = [i for i in cand]
                logging_info(self.local_rank, ops)

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

        self.save_checkpoint()

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
                raise KeyError(task_name)
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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default='log')
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.1)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--flops-limit', type=float, default=330 * 1e6)
    parser.add_argument('--max-train-iters', type=int, default=200)
    parser.add_argument('--max-test-iters', type=int, default=40)
    parser.add_argument('--train-batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=200)

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
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    t = time.time()

    searcher = EvolutionSearcher(args)

    searcher.search()

    logging_info(args.local_rank, 'total searching time = {:.2f} hours'.format(
        (time.time() - t) / 3600))

if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except:
        import traceback
        traceback.print_exc()
        time.sleep(1)
        os._exit(1)

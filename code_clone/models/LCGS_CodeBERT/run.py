from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaForSequenceClassification, RobertaTokenizer)

from torch.optim import AdamW
from tqdm import tqdm, trange
import multiprocessing
from model import Model

cpu_cont = 16
logger = logging.getLogger(__name__)

class InputFeatures(object):
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 url1,
                 url2

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        self.url1=url1
        self.url2=url2

def convert_examples_to_features(item):
    url1, url2, label, tokenizer, args, cache, url_to_code = item
    code1 = url_to_code[url1]
    code2 = url_to_code[url2]
    code1_tokens = tokenizer.tokenize(code1)
    code2_tokens = tokenizer.tokenize(code2)
    max_len_per_code = (args.code_length - 3) // 2
    code1_tokens = code1_tokens[:max_len_per_code]
    code2_tokens = code2_tokens[:max_len_per_code]
    source_tokens = [tokenizer.cls_token] + code1_tokens + [tokenizer.sep_token] + code2_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.code_length - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, label, url1, url2)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', lcgs_features=None):
        self.examples = []
        self.args = args
        self.lcgs_cwj = []
        self.lcgs_mcs = []
        index_filename = file_path
        data_dir = os.path.dirname(index_filename)

        jsonl_path = os.path.join(data_dir, "data.jsonl")
        if not os.path.exists(jsonl_path):
            jsonl_path = os.path.join(os.path.dirname(data_dir), 'data.jsonl')
        if not os.path.exists(jsonl_path):
            logger.error(f"Cannot find data.jsonl at {jsonl_path}")
            raise FileNotFoundError(f"data.jsonl not found")

        split_features = None
        if lcgs_features is not None:
            for key in ['train', 'valid', 'test']:
                if key in os.path.basename(file_path):
                    split_features = lcgs_features[key]
                    break
            if split_features is not None:
                logger.info(f"Using LCGS features ({len(split_features)} pairs) for {file_path}")

        logger.info(f"Loading code from {jsonl_path} ...")
        url_to_code = {}
        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                url_to_code[str(js['idx'])] = js['func']
        logger.info(f"Loaded {len(url_to_code)} functions.")
        data = []
        cache = {}
        logger.info(f"Loading pairs from {index_filename} ...")

        feat_idx = 0
        with open(index_filename, encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                parts = line.split('\t') if '\t' in line else line.split(',')
                if len(parts) != 3:
                    feat_idx += 1
                    continue

                url1, url2, label = parts
                url1, url2 = str(url1).strip(), str(url2).strip()
                label = int(label)
                if label > 1:
                    label = 1
                if url1 not in url_to_code or url2 not in url_to_code:
                    feat_idx += 1
                    continue

                data.append((url1, url2, label, tokenizer, args, cache, url_to_code))

                if split_features is not None and feat_idx < len(split_features):
                    cwj_val, mcs_val = split_features[feat_idx]
                    self.lcgs_cwj.append(float(cwj_val))
                    self.lcgs_mcs.append(int(mcs_val))
                else:
                    self.lcgs_cwj.append(0.0)
                    self.lcgs_mcs.append(0)
                feat_idx += 1

        self.examples = []
        total_len = len(data)
        logger.info(f"Start converting {total_len} examples...")

        for i, item in enumerate(data):
             self.examples.append(convert_examples_to_features(item))
             if (i+1) % 10000 == 0:
                 logger.info(f"Processed {i+1} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (torch.tensor(self.examples[item].input_ids),
                torch.tensor(self.examples[item].label),
                torch.tensor(self.lcgs_cwj[item], dtype=torch.float),
                torch.tensor(self.lcgs_mcs[item], dtype=torch.long))

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer):

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)

    args.max_steps=args.epochs*len( train_dataloader)
    args.save_steps = len(train_dataloader) - 1
    args.warmup_steps=args.max_steps//5
    model.to(args.device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    if args.lambda_param > 0:
        logger.info("  LCGS lambda = %f, mcs_margin = %f", args.lambda_param, args.mcs_margin)

    global_step=0
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_f1=0

    model.zero_grad()

    for idx in range(args.epochs):

        tr_num=0
        train_loss=0

        total_steps = len(train_dataloader)
        log_interval = max(1, total_steps // 10)

        logger.info(f"***** Epoch {idx} Start *****")

        for step, batch in enumerate(train_dataloader):
            (inputs, labels, cwj_batch, mcs_batch)=[x.to(args.device) for x in batch]
            model.train()
            ce_loss, logits = model(inputs, labels)

            if args.lambda_param > 0:
                pred_prob = logits[:, 1]
                y = labels.float()

                epsilon = 0.1
                r5_weights = cwj_batch + epsilon
                diff_pos = F.relu(cwj_batch - pred_prob)
                diff_neg = torch.abs(pred_prob)
                l1_error = y * diff_pos + (1 - y) * diff_neg
                l1_loss = (r5_weights * l1_error).mean()

                kappa = 2.0
                mask = (mcs_batch == 1) & (y == 1)
                if mask.any():
                    l2_loss = torch.sigmoid(kappa * (args.mcs_margin - pred_prob[mask])).mean()
                else:
                    l2_loss = torch.tensor(0.0, device=inputs.device)
                loss = ce_loss + args.lambda_param * (0.5 * l1_loss + 0.5 * l2_loss)
            else:
                loss = ce_loss

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss

            avg_loss=round(train_loss/tr_num,5)

            if (step + 1) % log_interval == 0:
                progress = (step + 1) / total_steps * 100
                logger.info(f"Epoch {idx} Progress: {int(progress)}% | Avg Loss: {avg_loss}")

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, eval_when_training=True)

                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)

                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

def evaluate(args, model, tokenizer, eval_when_training=False):

    eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,num_workers=4)

    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]
    y_trues=[]
    for batch in eval_dataloader:
        (inputs, labels, _, _)=[x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss,logit = model(inputs, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    logits=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)
    best_threshold=0.5
    best_f1=0

    y_preds=logits[:,1]>best_threshold
    from sklearn.metrics import recall_score
    recall=recall_score(y_trues, y_preds)
    from sklearn.metrics import precision_score
    precision=precision_score(y_trues, y_preds)
    from sklearn.metrics import f1_score
    f1=f1_score(y_trues, y_preds)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold":best_threshold,

    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return result

def test(args, model, tokenizer, best_threshold=0):

    eval_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]
    y_trues=[]
    for batch in eval_dataloader:
        (inputs, labels, _, _)=[x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss,logit = model(inputs, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)

    logger.info(f"Logits shape before reshape: {logits.shape}")
    logger.info(f"Y_trues shape: {y_trues.shape}")

    if logits.ndim == 1 and len(logits) == len(y_trues) * 2:
        logger.info("Reshaping flattened logits from (2N,) to (N, 2)")
        logits = logits.reshape(-1, 2)

    if logits.ndim == 1:
        y_preds = logits > best_threshold
    else:
        y_preds = logits[:, 1] > best_threshold

    from sklearn.metrics import recall_score, precision_score, f1_score
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)

    result = {
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_threshold": best_threshold,
    }

    logger.info("***** Test Results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    with open(os.path.join(args.output_dir,"predictions.txt"),'w') as f:
        for example,pred in zip(eval_dataset.examples,y_preds):
            if pred:
                f.write(str(example.url1)+'\t'+str(example.url2)+'\t'+'1'+'\n')
            else:
                f.write(str(example.url1)+'\t'+str(example.url2)+'\t'+'0'+'\n')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    parser.add_argument("--lambda_param", default=0.0, type=float,
                        help="LCGS constraint loss weight (0 = no LCGS)")
    parser.add_argument("--mcs_margin", default=0.5, type=float,
                        help="R4 MCS hard constraint margin threshold")
    parser.add_argument("--lcgs_features", default=None, type=str,
                        help="Path to LCGS features pkl file (gnn_lcgs_features.pkl)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    log_format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s'
    date_format = '%m/%d/%Y %H:%M:%S'

    logging.basicConfig(format=log_format, datefmt=date_format, level=logging.INFO)

    fh = logging.FileHandler(os.path.join(args.output_dir, 'result.txt'), mode='w')
    fh.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logger.addHandler(fh)

    logger.info("Training/evaluation parameters %s", args)

    lcgs_features = None
    if args.lcgs_features:
        with open(args.lcgs_features, 'rb') as f:
            lcgs_features = pickle.load(f)
        logger.info(f"Loaded LCGS features: {list(lcgs_features.keys())}")

    set_seed(args)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels=1
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaModel.from_pretrained(args.model_name_or_path,config=config)
    model=Model(model,config,tokenizer,args)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, file_path=args.train_data_file, lcgs_features=lcgs_features)
        train(args, train_dataset, model, tokenizer)

    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result=evaluate(args, model, tokenizer)

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, tokenizer,best_threshold=0.5)

    return results

if __name__ == "__main__":
    main()

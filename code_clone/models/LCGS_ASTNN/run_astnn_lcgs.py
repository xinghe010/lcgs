import os
import sys
import time
import warnings
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from astnn.model import BatchProgramCC
from LCGS_core.lcgs_trainer import LCGSTrainer

class CloneDataset(Dataset):

    def __init__(self, data_df):
        self.data = data_df
        self.func_x = self.data['func_x'].tolist()
        self.func_y = self.data['func_y'].tolist()

        raw_labels = self.data['label'].values
        self.binary_labels = (raw_labels > 0).astype(np.float32)

        if 'weight' in self.data.columns:
            stored_weights = self.data['weight'].values
        else:
            stored_weights = np.ones(len(self.data))

        self.weights = np.ones_like(stored_weights, dtype=np.float32)
        mask_neg = (self.binary_labels == 0)
        mask_low = (stored_weights < 0.9)

        self.weights[mask_low] = 0.5
        self.weights[~mask_neg] = np.where(mask_low[~mask_neg], 0.5, 1.0)
        self.weights[mask_neg] = 1.5

        self.cwjs = self.data.get(
            'cwj', pd.Series(np.zeros(len(self.data)))
        ).values.astype(np.float32)
        self.mcs = self.data.get(
            'mcs_equal', pd.Series(np.zeros(len(self.data)))
        ).values.astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.func_x[idx],
                self.func_y[idx],
                self.binary_labels[idx],
                self.cwjs[idx],
                self.mcs[idx],
                self.weights[idx])

def collate_fn(batch):
    batch_zip = list(zip(*batch))
    x1 = list(batch_zip[0])
    x2 = list(batch_zip[1])

    labels = torch.tensor(batch_zip[2]).unsqueeze(1)
    cwjs = torch.tensor(batch_zip[3]).unsqueeze(1)
    mcs = torch.tensor(batch_zip[4])
    weights = torch.tensor(batch_zip[5]).unsqueeze(1)

    return x1, x2, labels, cwjs, mcs, weights

def setup_logging(lang, save_base='modelsave_LCGS'):
    if not os.path.exists(save_base):
        os.mkdir(save_base)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_base, f"{lang}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    log_file_path = os.path.join(save_dir, 'log.txt')
    log_f = open(log_file_path, 'w', encoding='utf-8')
    return save_dir, log_f

def log_msg(msg, log_f):
    print(msg)
    log_f.write(msg + "\n")
    log_f.flush()

def validate(model_trainer, dataloader, use_gpu):
    model_trainer.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            val1, val2, labels, cwjs, mcs, batch_weights = batch
            curr_batch_size = len(labels)

            if use_gpu:
                labels = labels.cuda(non_blocking=True)
                cwjs = cwjs.cuda(non_blocking=True)
                mcs = mcs.cuda(non_blocking=True)
                batch_weights = batch_weights.cuda(non_blocking=True)

            model_trainer.model.batch_size = curr_batch_size
            model_trainer.model.hidden = model_trainer.model.init_hidden()

            loss, _ = model_trainer(val1, val2, labels, cwjs, mcs, batch_weights)
            total_loss += loss.item() * curr_batch_size
            total_samples += curr_batch_size

    model_trainer.train()
    return total_loss / total_samples if total_samples > 0 else 0.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASTNN + LCGS Training')
    parser.add_argument('--dataset', default='bcb', choices=['bcb', 'poj'],
                        help='Dataset: bcb (Java) or poj (C)')
    parser.add_argument('--data_dir', default=None,
                        help='Path to processed LCGS data')
    parser.add_argument('--lambda_param', type=float, default=0.1,
                        help='LCGS constraint weight')
    parser.add_argument('--mcs_margin', type=float, default=0.5,
                        help='R4 hard margin threshold')
    parser.add_argument('--use_mgda', action='store_true',
                        help='Use MGDA for loss balancing')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate (Adamax)')
    args = parser.parse_args()

    sys.setrecursionlimit(5000)

    lang = 'astnn_' + args.dataset
    if args.data_dir is None:
        args.data_dir = os.path.join(os.path.dirname(__file__), 'data_LCGS') + '/'

    root = args.data_dir
    save_dir, log_f = setup_logging(lang)

    log_msg(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_f)
    log_msg(f"Dataset: {lang}", log_f)
    log_msg(f"Lambda: {args.lambda_param}, MCS Margin: {args.mcs_margin}", log_f)
    log_msg(f"MGDA: {args.use_mgda}", log_f)

    log_msg("Loading data...", log_f)
    train_file = root + lang + '/train/blocks.pkl'
    train_data = pd.read_pickle(train_file)

    initial_len = len(train_data)
    train_data = train_data[
        train_data.apply(lambda x: len(x['func_x']) > 0 and len(x['func_y']) > 0, axis=1)
    ]
    train_data.reset_index(drop=True, inplace=True)

    if len(train_data) == 0:
        log_msg("Error: Training set is empty!", log_f)
        sys.exit(1)

    log_msg(f"Train: {len(train_data)} (filtered {initial_len - len(train_data)} empty)", log_f)

    test_data = pd.read_pickle(root + lang + '/test/blocks.pkl')
    test_data = test_data[
        test_data.apply(lambda x: len(x['func_x']) > 0 and len(x['func_y']) > 0, axis=1)
    ]
    test_data.reset_index(drop=True, inplace=True)

    dev_file = root + lang + '/dev/blocks.pkl'
    dev_data = None
    if os.path.exists(dev_file):
        dev_data = pd.read_pickle(dev_file)
        dev_data = dev_data[
            dev_data.apply(lambda x: len(x['func_x']) > 0 and len(x['func_y']) > 0, axis=1)
        ]
        dev_data.reset_index(drop=True, inplace=True)
        log_msg(f"Validation: {len(dev_data)}", log_f)

    w2v_path = root + lang + "/train/embedding/node_w2v_128"
    log_msg(f"Loading Word2Vec from {w2v_path}", log_f)
    word2vec = Word2Vec.load(w2v_path).wv

    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 1
    USE_GPU = torch.cuda.is_available()

    train_dataset = CloneDataset(train_data)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True
    )

    if dev_data is not None:
        dev_loader = DataLoader(
            CloneDataset(dev_data), batch_size=args.batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_fn, pin_memory=True
        )

    test_loader = DataLoader(
        CloneDataset(test_data), batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=True
    )

    model = BatchProgramCC(
        EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS,
        args.batch_size, USE_GPU, embeddings
    )

    lcgs_model = LCGSTrainer(
        model, model_type='astnn',
        lambda_param=args.lambda_param,
        mcs_margin=args.mcs_margin,
        use_mgda=args.use_mgda
    )

    device = torch.device("cuda" if USE_GPU else "cpu")
    log_msg(f"Device: {device}", log_f)

    if USE_GPU:
        lcgs_model.cuda()

    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_msg(f"Trainable parameters: {pytorch_total_params}", log_f)

    log_msg('Start training...', log_f)

    for epoch in range(args.epochs):
        log_msg(f"\n{'=' * 10} Epoch {epoch + 1}/{args.epochs} {'=' * 10}", log_f)
        start_time = time.time()

        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")

        for batch in pbar:
            train1_inputs, train2_inputs, train_labels, train_cwjs, train_mcs, train_weights = batch
            curr_batch_size = len(train_labels)

            if USE_GPU:
                train_labels = train_labels.cuda(non_blocking=True)
                train_cwjs = train_cwjs.cuda(non_blocking=True)
                train_mcs = train_mcs.cuda(non_blocking=True)
                train_weights = train_weights.cuda(non_blocking=True)

            model.zero_grad()
            model.batch_size = curr_batch_size
            model.hidden = model.init_hidden()

            loss, output = lcgs_model(
                train1_inputs, train2_inputs, train_labels,
                train_cwjs, train_mcs, train_weights
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * curr_batch_size
            total_samples += curr_batch_size

        end_time = time.time()
        avg_loss = total_loss / total_samples
        log_msg(
            f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Time={end_time - start_time:.2f}s",
            log_f
        )

        if dev_data is not None:
            val_loss = validate(lcgs_model, dev_loader, USE_GPU)
            log_msg(f"  Train Loss: {avg_loss:.4f}, Valid Loss: {val_loss:.4f}", log_f)

        model_save_path = os.path.join(save_dir, f'astnn_lcgs_epoch_{epoch + 1}.pt')
        torch.save(model.state_dict(), model_save_path)
        log_msg(f"  Model saved: {model_save_path}", log_f)

    log_msg("\nTesting...", log_f)
    predicts = []
    trues = []

    lcgs_model.eval()
    with torch.no_grad():
        pbar_test = tqdm(test_loader, desc="Testing")

        for batch in pbar_test:
            test1_inputs, test2_inputs, test_labels, test_cwjs, test_mcs, test_weights = batch
            curr_batch_size = len(test_labels)

            if USE_GPU:
                test_labels = test_labels.cuda(non_blocking=True)
                test_cwjs = test_cwjs.cuda(non_blocking=True)
                test_mcs = test_mcs.cuda(non_blocking=True)
                test_weights = test_weights.cuda(non_blocking=True)

            lcgs_model.model.batch_size = curr_batch_size
            lcgs_model.model.hidden = lcgs_model.model.init_hidden()

            loss, output = lcgs_model(
                test1_inputs, test2_inputs, test_labels,
                test_cwjs, test_mcs, test_weights
            )

            predicted = (output.data > 0.5).cpu().numpy()
            predicts.extend(predicted)
            trues.extend(test_labels.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')

    result_str = (
        f"\n{'=' * 20} Results {'=' * 20}\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Dataset: {lang}\n"
        f"Lambda: {args.lambda_param}, MCS Margin: {args.mcs_margin}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall:    {recall:.4f}\n"
        f"F1 Score:  {f1:.4f}\n"
        f"{'=' * 49}"
    )

    log_msg(result_str, log_f)
    with open(os.path.join(save_dir, 'result_summary.txt'), 'w') as f_res:
        f_res.write(result_str)

    log_f.close()
    print(f"Done! Results saved in {save_dir}")

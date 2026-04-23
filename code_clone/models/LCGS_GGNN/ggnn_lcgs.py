import os
import sys
import time
import random
import pickle
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gnn.models import GGNN
from LCGS_core.lcgs_trainer import LCGSTrainer

project_root = str(Path(__file__).resolve().parents[3])

DATASET_CONFIG = {
    'bcb': {'lang': 'java', 'createclone_module': 'gnn.createclone'},
    'poj': {'lang': 'c', 'createclone_module': 'gnn.createclone_c'},
}

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_createclone_module(dataset):
    cfg = DATASET_CONFIG[dataset]
    if cfg['lang'] == 'c':
        from gnn import createclone_c as cc
    else:
        from gnn import createclone as cc
    return cc

def inject_features(dataset, feature_list, device):
    new_dataset = []
    for i, (data_pack, label) in enumerate(dataset):
        if i < len(feature_list):
            cwj, mcs = feature_list[i]
        else:
            cwj, mcs = 0.0, 0
        new_dataset.append((data_pack, label, cwj, mcs))
    return new_dataset

def load_lcgs_features(feature_path):
    if not os.path.exists(feature_path):
        print(f"Error: LCGS feature file not found at {feature_path}.")
        print("Please run pipeline_gnn_lcgs.py first.")
        sys.exit(1)
    print(f"Loading LCGS features from {feature_path}...")
    with open(feature_path, 'rb') as f:
        return pickle.load(f)

def test(model, dataset, feature_list, device, threshold=0):
    model.eval()
    tp, tn, fp, fn = 0, 0, 0, 0
    dataset_with_feats = inject_features(dataset, feature_list, device)

    with torch.no_grad():
        for data, label, cwj, mcs in tqdm(dataset_with_feats, desc="Testing"):
            label_val = torch.tensor(label, dtype=torch.float, device=device)

            x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2 = data
            x1 = torch.tensor(x1, dtype=torch.long, device=device)
            x2 = torch.tensor(x2, dtype=torch.long, device=device)
            edge_index1 = torch.tensor(edge_index1, dtype=torch.long, device=device)
            edge_index2 = torch.tensor(edge_index2, dtype=torch.long, device=device)
            if edge_attr1 is not None:
                edge_attr1 = torch.tensor(edge_attr1, dtype=torch.long, device=device)
                edge_attr2 = torch.tensor(edge_attr2, dtype=torch.long, device=device)

            data1 = [x1, edge_index1, edge_attr1]
            data2 = [x2, edge_index2, edge_attr2]

            h1 = model(data1)
            h2 = model(data2)
            output = F.cosine_similarity(h1, h2)
            prediction = output.item()

            if prediction > threshold and label_val.item() == 1:
                tp += 1
            if prediction <= threshold and label_val.item() == -1:
                tn += 1
            if prediction > threshold and label_val.item() == -1:
                fp += 1
            if prediction <= threshold and label_val.item() == 1:
                fn += 1

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    print(f"TP:{tp} TN:{tn} FP:{fp} FN:{fn}")
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    res = f"{now}: P:{p:.4f} R:{r:.4f} F1:{f1:.4f}\n"
    model.train()
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GGNN + LCGS Training')
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--dataset", default='bcb', choices=['bcb', 'poj'],
                        help='Dataset: bcb (Java) or poj (C)')
    parser.add_argument("--graphmode", default='astandnext')
    parser.add_argument("--nextsib", default=False)
    parser.add_argument("--ifedge", default=False)
    parser.add_argument("--whileedge", default=False)
    parser.add_argument("--foredge", default=False)
    parser.add_argument("--blockedge", default=False)
    parser.add_argument("--nexttoken", default=False)
    parser.add_argument("--nextuse", default=False)
    parser.add_argument("--data_setting", default='0')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--data", default=project_root + '/data/')
    parser.add_argument('--root', default=project_root + '/src/models/gnn/')

    parser.add_argument("--lambda_param", type=float, default=0.1,
                        help='LCGS constraint weight')
    parser.add_argument("--mcs_margin", type=float, default=0.6,
                        help='R4 hard margin threshold')
    parser.add_argument("--bench", default=None,
                        help='Override data bench directory (e.g. poj_LCGS)')

    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset} ({DATASET_CONFIG[args.dataset]['lang']})")

    cc = get_createclone_module(args.dataset)

    if args.bench is None:
        args.bench = args.dataset

        lcgs_bench = args.bench + '_LCGS'
        lcgs_data_path = os.path.join(args.data, lcgs_bench)
        if os.path.exists(lcgs_data_path):
            print(f"Using LCGS-augmented dataset: {lcgs_bench}")
            args.bench = lcgs_bench
    else:
        print(f"Using explicit bench: {args.bench}")

    lcgs_feature_path = os.path.join(args.data, args.bench, 'gnn_lcgs_features.pkl')
    feature_storage = load_lcgs_features(lcgs_feature_path)

    print("Creating graph data...")
    cache_file = os.path.join(args.root, f"{args.bench}_ggnn_cached_data.pkl")
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            astdict, vocablen, vocabdict, traindata_raw, validdata_raw, testdata_raw = pickle.load(f)
    else:
        astdict, vocablen, vocabdict = cc.createast(args)
        treedict = cc.createseparategraph(
            args, astdict, vocablen, vocabdict, device,
            mode=args.graphmode, nextsib=args.nextsib, ifedge=args.ifedge,
            whileedge=args.whileedge, foredge=args.foredge,
            blockedge=args.blockedge, nexttoken=args.nexttoken, nextuse=args.nextuse
        )
        traindata_raw, validdata_raw, testdata_raw = cc.creategmndata(
            args, args.data_setting, treedict, vocablen, vocabdict, device
        )
        print(f"Saving cache to {cache_file}...")
        with open(cache_file, 'wb') as f:
            pickle.dump((astdict, vocablen, vocabdict, traindata_raw, validdata_raw, testdata_raw), f)

    print(f"Train: {len(traindata_raw)}, Valid: {len(validdata_raw)}, Test: {len(testdata_raw)}")

    traindata = inject_features(traindata_raw, feature_storage.get('train', []), device)
    validdata = inject_features(validdata_raw, feature_storage.get('valid', []), device)
    testdata_features = feature_storage.get('test', [])

    num_layers = int(args.num_layers)
    base_model = GGNN(vocablen, embedding_dim=100, num_layers=num_layers, device=device).to(device)
    lcgs_model = LCGSTrainer(
        base_model, model_type='ggnn',
        lambda_param=args.lambda_param, mcs_margin=args.mcs_margin
    ).to(device)

    optimizer = optim.Adam(base_model.parameters(), lr=args.lr)

    results_dir = os.path.join(args.root, 'results')
    saved_models_dir = os.path.join(args.root, 'saved_models')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(saved_models_dir, exist_ok=True)

    def create_batches(data):
        return [data[i:i + args.batch_size] for i in range(0, len(data), args.batch_size)]

    epochs = trange(args.num_epochs, leave=True, desc="Epoch")
    for epoch in epochs:
        epoch_start = time.time()
        random.shuffle(traindata)
        batches = create_batches(traindata)
        totalloss = 0.0
        main_index = 0.0

        for batch in tqdm(batches, desc="Batches", leave=False):
            optimizer.zero_grad()
            batchloss = 0

            for data, label, cwj_val, mcs_val in batch:
                label_t = torch.tensor(label, dtype=torch.float, device=device)
                cwj_t = torch.tensor(float(cwj_val), dtype=torch.float, device=device)
                mcs_t = torch.tensor(int(mcs_val), dtype=torch.long, device=device)

                x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2 = data
                x1 = torch.tensor(x1, dtype=torch.long, device=device)
                x2 = torch.tensor(x2, dtype=torch.long, device=device)
                edge_index1 = torch.tensor(edge_index1, dtype=torch.long, device=device)
                edge_index2 = torch.tensor(edge_index2, dtype=torch.long, device=device)
                if edge_attr1 is not None:
                    edge_attr1 = torch.tensor(edge_attr1, dtype=torch.long, device=device)
                    edge_attr2 = torch.tensor(edge_attr2, dtype=torch.long, device=device)

                data1 = [x1, edge_index1, edge_attr1]
                data2 = [x2, edge_index2, edge_attr2]

                loss, _ = lcgs_model.forward_ggnn(
                    data1, data2, label_t, cwj_t, mcs_t
                )
                batchloss = batchloss + loss

            batchloss.backward(retain_graph=True)
            optimizer.step()

            totalloss += batchloss.item()
            main_index += len(batch)
            avg_loss = totalloss / main_index
            epochs.set_description("Epoch (Loss=%g)" % round(avg_loss, 5))

        epoch_train_time = time.time() - epoch_start

        testresults = test(
            base_model, testdata_raw, testdata_features,
            device, threshold=args.threshold
        )
        epoch_total_time = time.time() - epoch_start
        print(f"Epoch {epoch} time: {epoch_train_time:.2f}s train, {epoch_total_time:.2f}s total")

        result_filename = os.path.join(
            args.root, f'results_ggnn_lcgs_{args.dataset}.txt'
        )
        with open(result_filename, 'a') as f:
            f.write(f"Epoch {epoch} (time={epoch_total_time:.1f}s): {testresults}")

        save_path = os.path.join(
            saved_models_dir,
            f'ggnn_lcgs_{args.graphmode}_{args.dataset}_{epoch}.pt'
        )
        torch.save(base_model.state_dict(), save_path)

    print("Training complete!")

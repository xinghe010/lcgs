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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gnn.models import GMNnet
from LCGS_core.lcgs_trainer import LCGSTrainer

project_root = str(Path(__file__).resolve().parents[3])

DATASET_CONFIG = {
    'bcb': {'lang': 'java'},
    'poj': {'lang': 'c'},
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

class GMNDataset(Dataset):

    def __init__(self, raw_data, features, device):
        self.data = []
        for i, (data_pack, label) in enumerate(raw_data):
            x1, x2, ei1, ei2, ea1, ea2 = data_pack
            x1 = torch.tensor(x1, dtype=torch.long)
            x2 = torch.tensor(x2, dtype=torch.long)
            ei1 = torch.tensor(ei1, dtype=torch.long)
            ei2 = torch.tensor(ei2, dtype=torch.long)

            if ea1 is not None:
                ea1 = torch.tensor(ea1, dtype=torch.long)
                ea2 = torch.tensor(ea2, dtype=torch.long)
            else:
                ea1 = torch.tensor([], dtype=torch.long)
                ea2 = torch.tensor([], dtype=torch.long)

            if i < len(features):
                cwj, mcs = features[i]
            else:
                cwj, mcs = 0.0, 0

            label_val = 1.0 if float(label) >= 1.0 else 0.0
            self.data.append({
                'x1': x1, 'x2': x2,
                'ei1': ei1, 'ei2': ei2,
                'ea1': ea1, 'ea2': ea2,
                'label': torch.tensor([label_val], dtype=torch.float),
                'cwj': torch.tensor(float(cwj), dtype=torch.float),
                'mcs': torch.tensor(int(mcs), dtype=torch.long)
            })

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

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

def test(model, dataset, feature_list, device, threshold=0.5):
    model.eval()
    tp, tn, fp, fn = 0, 0, 0, 0
    dataset_with_feats = inject_features(dataset, feature_list, device)

    with torch.no_grad():
        for data, label, cwj, mcs in tqdm(dataset_with_feats, desc="Testing"):
            label_val = 1.0 if float(label) >= 1.0 else 0.0
            label_t = torch.tensor(label_val, dtype=torch.float, device=device)

            x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2 = data
            x1 = torch.tensor(x1, dtype=torch.long, device=device)
            x2 = torch.tensor(x2, dtype=torch.long, device=device)
            edge_index1 = torch.tensor(edge_index1, dtype=torch.long, device=device)
            edge_index2 = torch.tensor(edge_index2, dtype=torch.long, device=device)
            if edge_attr1 is not None:
                edge_attr1 = torch.tensor(edge_attr1, dtype=torch.long, device=device)
                edge_attr2 = torch.tensor(edge_attr2, dtype=torch.long, device=device)

            data_pack = [x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]

            h1, h2 = model(data_pack)
            output = F.cosine_similarity(h1, h2)
            prediction = output.item()

            if prediction > threshold and label_t.item() == 1:
                tp += 1
            if prediction <= threshold and label_t.item() == 0:
                tn += 1
            if prediction > threshold and label_t.item() == 0:
                fp += 1
            if prediction <= threshold and label_t.item() == 1:
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
    parser = argparse.ArgumentParser(description='GMN + LCGS Training')
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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--threshold", type=float, default=0.5)
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
    cache_file = os.path.join(args.root, f"{args.bench}_gmn_cached_data.pkl")
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

    train_features = feature_storage.get('train', [])
    if len(traindata_raw) != len(train_features):
        print(f"Warning: Data mismatch! GMN={len(traindata_raw)} vs Features={len(train_features)}")

    testdata_features = feature_storage.get('test', [])

    num_layers = int(args.num_layers)
    base_model = GMNnet(vocablen, embedding_dim=100, num_layers=num_layers, device=device).to(device)
    lcgs_model = LCGSTrainer(
        base_model, model_type='gmn',
        lambda_param=args.lambda_param, mcs_margin=args.mcs_margin
    ).to(device)

    optimizer = optim.Adam(base_model.parameters(), lr=args.lr)

    saved_models_dir = os.path.join(args.root, 'saved_models')
    os.makedirs(saved_models_dir, exist_ok=True)

    train_dataset = GMNDataset(traindata_raw, train_features, device)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda x: x, num_workers=0
    )

    epochs = trange(args.num_epochs, leave=True, desc="Epoch")

    for epoch in epochs:
        epoch_start = time.time()
        totalloss = 0.0
        main_index = 0.0

        for batch in tqdm(train_loader, desc="Batches", leave=False):
            optimizer.zero_grad()
            batchloss = 0

            for item in batch:
                x1 = item['x1'].to(device)
                x2 = item['x2'].to(device)
                ei1 = item['ei1'].to(device)
                ei2 = item['ei2'].to(device)
                ea1 = item['ea1'].to(device) if item['ea1'].numel() > 0 else None
                ea2 = item['ea2'].to(device) if item['ea2'].numel() > 0 else None

                label_tensor = item['label'].to(device)
                cwj_tensor = item['cwj'].to(device)
                mcs_tensor = item['mcs'].to(device)

                data_pack = [x1, x2, ei1, ei2, ea1, ea2]

                loss, _ = lcgs_model.forward_gmn(
                    data_pack, label_tensor, cwj_tensor, mcs_tensor
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
            args.root, f'results_gmn_lcgs_{args.dataset}.txt'
        )
        with open(result_filename, 'a') as f:
            f.write(f"Epoch {epoch} (time={epoch_total_time:.1f}s): {testresults}")

        save_path = os.path.join(
            saved_models_dir,
            f'gmn_lcgs_{args.graphmode}_{args.dataset}_{epoch}.pt'
        )
        torch.save(base_model.state_dict(), save_path)

    print("Training complete!")

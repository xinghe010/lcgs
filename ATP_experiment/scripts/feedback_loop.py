#!/usr/bin/env python3

import os, sys, json, copy, random, pickle, subprocess, logging, shutil, argparse
import torch
import numpy as np
from torch.utils.data import ConcatDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
BASE = os.environ.get("PROVER_BASE", _PROJ_ROOT)
TD = os.environ.get("SCRIPTS_DIR", f"{BASE}/TWGNN/scripts" if os.path.isdir(f"{BASE}/TWGNN/scripts") else _THIS_DIR)
CH = os.environ.get("CHAINY_DIR", f"{BASE}/dataset/MPTP2078-master/chainy" if os.path.isdir(f"{BASE}/dataset/MPTP2078-master/chainy") else f"{BASE}/MPTP2078-master/chainy")
EP = os.environ.get("EPROVER_BIN",
        f"{BASE}/E-prover/PROVER/eprover" if os.path.exists(f"{BASE}/E-prover/PROVER/eprover")
        else (f"{BASE}/eprover/PROVER/eprover" if os.path.exists(f"{BASE}/eprover/PROVER/eprover") else "eprover"))

PremiseSelectionModel = None
FormulaGraphDataset = None
MODEL_TYPE = None
MODEL_DIR = None
BASE_DATASET = None
SHARED_FILES = []

def init_model_env(model_type):
    global PremiseSelectionModel, FormulaGraphDataset
    global MODEL_TYPE, MODEL_DIR, BASE_DATASET, SHARED_FILES
    import __main__

    MODEL_TYPE = model_type

    if model_type == 'LAH':
        MODEL_DIR = f"{BASE}/LAH_TWGNN"
        code_dir = f"{MODEL_DIR}/code"
        sys.path.insert(0, code_dir)
        BASE_DATASET = f"{MODEL_DIR}/dataset_D0"
        SHARED_FILES = ["node_dict.pkl", "statements", "test_problems_provable.json"]

    elif model_type == 'LFR':
        MODEL_DIR = f"{BASE}/TW-GNN-LFR"
        code_dir = MODEL_DIR
        sys.path.insert(0, code_dir)
        BASE_DATASET = f"{MODEL_DIR}/dataset_D0"
        SHARED_FILES = ["node_dict.pkl", "node_attr_dict.pkl", "edge_attr_dict.pkl",
                        "statements", "test_problems_provable.json"]

    elif model_type == 'CL':
        MODEL_DIR = f"{BASE}/CL-TWGNN"
        code_dir = MODEL_DIR
        sys.path.insert(0, code_dir)
        BASE_DATASET = f"{MODEL_DIR}/dataset_D0"
        SHARED_FILES = ["node_dict.pkl", "add.cnf", "add.atom2idx",
                        "statements", "test_problems_provable.json"]

    elif model_type == 'DEEPMATH':
        MODEL_DIR = f"{BASE}/DeepMath"
        code_dir = MODEL_DIR
        sys.path.insert(0, code_dir)

        BASE_DATASET = f"{BASE}/LAH_TWGNN/dataset_D0"
        SHARED_FILES = ["statements", "test_problems_provable.json"]
        return

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    from model import PremiseSelectionModel as _PSM
    from dataset import FormulaGraphDataset as _FGD, PairData as _PD
    PremiseSelectionModel = _PSM
    FormulaGraphDataset = _FGD

    __main__.PairData = _PD

    try:
        from dataset import Graph as _G
        __main__.Graph = _G
    except ImportError:
        pass
    try:
        from dataset import Node as _N
        __main__.Node = _N
    except ImportError:
        pass

def setup_logger(logfile):
    logger = logging.getLogger("feedback_loop")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def gen_incremental_dataset(new_deps, neg_deps, round_dir, logger):
    os.makedirs(os.path.join(round_dir, "train", "raw"), exist_ok=True)

    for f in SHARED_FILES:
        src = os.path.join(BASE_DATASET, f)
        dst = os.path.join(round_dir, f)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(os.path.realpath(src), dst)

    dataset = []
    for conj, prems in new_deps.items():
        for p in prems:
            dataset.append([conj, p, 1])
    for conj, prems in neg_deps.items():
        for p in prems:
            dataset.append([conj, p, 0])

    random.shuffle(dataset)
    train_json = os.path.join(round_dir, "train", "raw", "train.json")
    with open(train_json, 'w') as f:
        for d in dataset:
            f.write(json.dumps(d) + '\n')

    n_pos = sum(1 for d in dataset if d[2] == 1)
    n_neg = sum(1 for d in dataset if d[2] == 0)
    logger.info(f"  Incremental data: {len(dataset)} samples (pos:{n_pos} neg:{n_neg})")
    return round_dir

def load_one_dataset(ds_path):
    if MODEL_TYPE == 'LFR':
        return FormulaGraphDataset(
            os.path.join(ds_path, "train"), "train",
            os.path.join(ds_path, "statements"),
            os.path.join(ds_path, "node_dict.pkl"),
            os.path.join(ds_path, "node_attr_dict.pkl"),
            os.path.join(ds_path, "edge_attr_dict.pkl"),
            rename=True)
    else:
        return FormulaGraphDataset(
            os.path.join(ds_path, "train"), "train",
            os.path.join(ds_path, "statements"),
            os.path.join(ds_path, "node_dict.pkl"),
            rename=True)

def create_model(dim, layers, device, reg=None):
    if MODEL_TYPE == 'CL':

        orig_dir = os.getcwd()
        os.chdir(MODEL_DIR)
        model = PremiseSelectionModel(793, dim, layers, reg or ['cnf', 'bound'],
                                      device=device).to(device)
        os.chdir(orig_dir)
        return model
    elif MODEL_TYPE == 'LAH':
        return PremiseSelectionModel(793, dim, layers).to(device)
    else:
        return PremiseSelectionModel(793, dim, layers).to(device)

def train_model(datasets, model_save, dim, layers, epochs, device, logger,
                seed=24, init_model=None, lr=0.001, batch_size=32,
                reg=None, hyper=None, max_len=2048):
    os.makedirs(model_save, exist_ok=True)

    if MODEL_TYPE == 'DEEPMATH':
        from deepmath_model import DeepMathModel, DeepMathDataset, build_char_vocab, load_statements
        import copy as _copy

        torch.manual_seed(seed)
        if 'cuda' in str(device): torch.cuda.manual_seed_all(seed)

        stmts_path = os.path.join(BASE_DATASET, "statements")
        vocab_path = os.path.join(model_save, "vocab.pkl")
        if os.path.exists(vocab_path):
            vocab = pickle.load(open(vocab_path, 'rb'))
        else:
            vocab = build_char_vocab(stmts_path)
            pickle.dump(vocab, open(vocab_path, 'wb'))
        logger.info(f"  Vocabulary: {len(vocab)} chars")

        all_ds = []
        for ds_path in datasets:
            tj = os.path.join(ds_path, "train", "raw", "train.json")
            sp = os.path.join(ds_path, "statements")
            if not os.path.exists(sp): sp = stmts_path
            ds = DeepMathDataset(tj, sp, vocab, max_len=max_len)
            all_ds.append(ds)
            logger.info(f"  Dataset: {ds_path} ({len(ds)} samples)")
        from torch.utils.data import ConcatDataset as _CD
        combined = _CD(all_ds) if len(all_ds) > 1 else all_ds[0]
        loader = DataLoader(combined, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        logger.info(f"  Total samples: {len(combined)}")

        model = DeepMathModel(len(vocab), embed_dim=80, out_dim=dim).to(device)
        if init_model and os.path.exists(init_model):
            ckpt = torch.load(init_model, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            logger.info(f"  Initialized from existing model: {init_model}")

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        lr_scheduler = ReduceLROnPlateau(optimizer)
        top_models = []
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            for conj, prem, labels in loader:
                conj, prem, labels = conj.to(device), prem.to(device), labels.to(device)
                optimizer.zero_grad()
                loss, pred = model(conj, prem, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            if epoch % 10 == 0 or epoch == epochs:
                logger.info(f"  Epoch {epoch}/{epochs}: loss={avg_loss:.4f}")
            sd = _copy.deepcopy(model.state_dict())
            top_models.append((avg_loss, sd))
            top_models = sorted(top_models, key=lambda x: x[0])[:5]
            lr_scheduler.step(avg_loss)

        avg_sd = {}
        sds = [sd for _, sd in top_models]
        for key in sds[0]:
            avg_sd[key] = sum(sd[key] for sd in sds) / len(sds)
        torch.save({'model': avg_sd, 'vocab': vocab, 'dim': dim, 'max_len': max_len},
                   os.path.join(model_save, "averaged_top5.pt"))
        logger.info(f"  Model saved: {model_save}")
        return

    torch.manual_seed(seed)
    if 'cuda' in str(device):
        torch.cuda.manual_seed_all(seed)

    model = create_model(dim, layers, device, reg)

    if init_model and os.path.exists(init_model):
        checkpoint = torch.load(init_model, map_location=device)
        sd = checkpoint.get('model', checkpoint)
        model.load_state_dict(sd)
        logger.info(f"  Initialized from existing model: {init_model}")

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = ReduceLROnPlateau(optimizer)

    all_datasets = []
    for ds_path in datasets:
        ds = load_one_dataset(ds_path)
        all_datasets.append(ds)
        logger.info(f"  Loaded dataset: {ds_path} ({len(ds)} samples)")

    combined = ConcatDataset(all_datasets)
    loader = DataLoader(combined, batch_size=batch_size, shuffle=True,
                        follow_batch=["x_s", "x_t"])
    logger.info(f"  Total training samples: {len(combined)}")

    update_fn = None
    if MODEL_TYPE == 'CL':
        try:
            from scales import update_grad_scales
            update_fn = update_grad_scales
        except ImportError:
            pass

    top_models = []
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for i, batch in enumerate(loader, 1):
            optimizer.zero_grad()
            batch = batch.to(device)

            if MODEL_TYPE == 'CL':
                loss, targets, pred_targets = model(batch, device, hyper or [0.8, 0.1])
            elif MODEL_TYPE == 'LFR':
                loss = model(batch)
            else:
                loss, targets, pred_targets = model(batch)

            if MODEL_TYPE == 'LAH':
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / i
        if epoch % 10 == 0 or epoch == epochs:
            logger.info(f"  Epoch {epoch}/{epochs}: loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            if update_fn and MODEL_TYPE == 'CL':

                class _Args:
                    QoutFlag = True
                    out_levels = 2
                    bkwd_scaling_factorO = 0.0
                    use_hessian = True
                    update_every = 10
                try:
                    update_fn(model, loader, device, _Args(), hyper or [0.8, 0.1])
                except Exception:
                    pass

        sd = copy.deepcopy(model.state_dict())
        top_models.append((avg_loss, sd))
        top_models = sorted(top_models, key=lambda x: x[0])[:5]
        lr_scheduler.step(avg_loss)

    for rank, (t_loss, sd) in enumerate(top_models):
        torch.save({"model": sd}, os.path.join(model_save, f"{rank+1}.pt"))

    avg_sd = {}
    sds = [sd for _, sd in top_models]
    for key in sds[0]:
        avg_sd[key] = sum(sd[key] for sd in sds) / len(sds)
    torch.save({"model": avg_sd}, os.path.join(model_save, "averaged_top5.pt"))
    logger.info(f"  Model saved: {model_save}")

def generate_ranking(model_path, dim, layers, output_path, device, logger):
    logger.info(f"  Ranking...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if MODEL_TYPE == 'LAH':
        cmd = (f"cd {MODEL_DIR}/code && CUDA_VISIBLE_DEVICES=0 python3 select_premises_lah.py"
               f" --model {model_path} --statements ../dataset_D0/statements"
               f" --node_dict ../dataset_D0/node_dict.pkl --chainy_dir {CH}/"
               f" --test_problems ../dataset_D0/test_problems_provable.json"
               f" --output {output_path} --device {device} --top_k 1024"
               f" --dim {dim} --layers {layers}")
    elif MODEL_TYPE == 'LFR':
        cmd = (f"cd {MODEL_DIR} && CUDA_VISIBLE_DEVICES=0 python3 select_premises_lfr.py"
               f" --model {model_path} --statements dataset_D0/statements"
               f" --node_dict dataset_D0/node_dict.pkl"
               f" --node_attr_dict dataset_D0/node_attr_dict.pkl"
               f" --edge_attr_dict dataset_D0/edge_attr_dict.pkl"
               f" --chainy_dir {CH}/"
               f" --test_problems dataset_D0/test_problems_provable.json"
               f" --output {output_path} --device {device} --top_k 1024"
               f" --dim {dim} --layers {layers}")
    elif MODEL_TYPE == 'CL':
        cmd = (f"cd {MODEL_DIR} && CUDA_VISIBLE_DEVICES=0 python3 select_premises_cl.py"
               f" --model {model_path} --statements dataset_D0/statements"
               f" --node_dict dataset_D0/node_dict.pkl --chainy_dir {CH}/"
               f" --test_problems dataset_D0/test_problems_provable.json"
               f" --output {output_path} --device {device} --top_k 1024"
               f" --dim {dim} --layers {layers}")
    elif MODEL_TYPE == 'DEEPMATH':
        cmd = (f"cd {BASE} && CUDA_VISIBLE_DEVICES=0 python3 DeepMath/train_and_rank.py"
               f" --mode rank --model {model_path}"
               f" --statements_dir {BASE}/LAH_TWGNN/statements"
               f" --chainy_dir {CH}/"
               f" --test_problems {BASE_DATASET}/test_problems_provable.json"
               f" --output {output_path} --device {device} --top_k 1024"
               f" --dim {dim}")

    subprocess.run(cmd, shell=True, capture_output=True)
    if os.path.exists(output_path):
        d = json.load(open(output_path))
        logger.info(f"  Ranking done: {len(d)} problems")
    else:
        logger.info(f"  Ranking failed!")

def get_safe_shards(requested):
    try:
        r = subprocess.run(['free', '-m'], capture_output=True, text=True)
        avail = int(r.stdout.split('\n')[1].split()[-1])
        if avail < 10000: return min(requested, 2)
        elif avail < 15000: return min(requested, 4)
        elif avail < 25000: return min(requested, 6)
        return min(requested, 12)
    except:
        return min(requested, 6)

def run_cascade(ranking_path, output_dir, n_shards, logger):
    tp = f"{BASE_DATASET}/test_problems_provable.json"
    n_shards = get_safe_shards(n_shards)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"  Cascade evaluation ({n_shards} shards)...")

    procs = []
    for i in range(1, n_shards + 1):
        cmd = (f"python3 {TD}/evaluate_cascade.py run --split {i}/{n_shards}"
               f" --premise_rankings {ranking_path} --test_problems {tp}"
               f" --eprover_path {EP} --chainy_dir {CH} --output_dir {output_dir}/")
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        procs.append(p)
    for p in procs:
        p.wait()

    subprocess.run(f"python3 {TD}/evaluate_cascade.py merge --output_dir {output_dir}/",
                   shell=True, capture_output=True)

    metrics_file = os.path.join(output_dir, "cascade_metrics.json")
    if os.path.exists(metrics_file):
        m = json.load(open(metrics_file))
        logger.info(f"  Proved: {m['proved_count']}/280 ({m['proved_rate']}%)")
        return m['proved_count']
    return 0

def extract_new_deps(cascade_dir, ranking_path, existing_deps, logger):
    merged_file = os.path.join(cascade_dir, "cascade_merged.json")
    if not os.path.exists(merged_file):
        logger.info("  cascade_merged.json not found!")
        return {}, {}

    merged = json.load(open(merged_file))
    new_pos_deps = {}
    hard_neg_deps = {}

    all_used = {}
    for prob, result in merged.items():
        if not result.get('proved'):
            continue
        used = set(result.get('used_axioms', []))
        if not used:
            continue
        all_used[prob] = used
        existing_pos = existing_deps.get(prob, set())
        new_prems = used - existing_pos
        if new_prems:
            new_pos_deps[prob] = new_prems

    if os.path.exists(ranking_path):
        rankings = json.load(open(ranking_path))
        for prob, used in all_used.items():
            if prob not in rankings:
                continue
            top_ranked = rankings[prob][:128]
            existing_pos = existing_deps.get(prob, set())
            hard_negs = [p for p in top_ranked if p not in used and p not in existing_pos]
            n_hard = min(len(hard_negs), max(len(used), 5))
            if n_hard > 0:
                hard_neg_deps[prob] = set(hard_negs[:n_hard])

    total_new_pos = sum(len(v) for v in new_pos_deps.values())
    total_hard_neg = sum(len(v) for v in hard_neg_deps.values())
    logger.info(f"  New positives: {total_new_pos} (from {len(new_pos_deps)} problems)")
    logger.info(f"  Hard negatives: {total_hard_neg} (from {len(hard_neg_deps)} problems)")
    return new_pos_deps, hard_neg_deps

def main():
    parser = argparse.ArgumentParser(description='Generic feedback loop (LAH/LFR/CL)')
    parser.add_argument('--model_type', type=str, required=True, choices=['LAH', 'LFR', 'CL', 'DEEPMATH'])
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_shards', type=int, default=6)
    parser.add_argument('--init_model', type=str, default=None,
                        help='Initial model path (skip round 1 training)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Feedback loop data directory (default: {MODEL_DIR}/feedback_loop_v2)')
    parser.add_argument('--logfile', type=str, default=None)

    parser.add_argument('--reg', nargs='+', default=['cnf', 'bound'])
    parser.add_argument('--hyper', nargs='+', type=float, default=[0.8, 0.1])
    args = parser.parse_args()

    init_model_env(args.model_type)

    if args.data_dir is None:
        args.data_dir = f"{MODEL_DIR}/feedback_loop_v2"
    if args.logfile is None:
        args.logfile = f"{MODEL_DIR}/feedback_loop_v2.log"

    os.makedirs(args.data_dir, exist_ok=True)
    logger = setup_logger(args.logfile)
    results_file = os.path.join(args.data_dir, "loop_results.json")

    logger.info("=" * 60)
    logger.info(f"{args.model_type} feedback loop")
    logger.info(f"Iterations: {args.iterations}, dim={args.dim}, layers={args.layers}, "
                f"lr={args.lr}, bs={args.batch_size}, epochs={args.epochs}")
    logger.info("=" * 60)

    d0_train = os.path.join(BASE_DATASET, "train", "raw", "train.json")
    d0_data = [json.loads(l) for l in open(d0_train)]
    existing_deps = {}
    for conj, prem, label in d0_data:
        if label == 1:
            existing_deps.setdefault(conj, set()).add(prem)

    dataset_dirs = [BASE_DATASET]
    results = {}

    if os.path.exists(results_file):
        results = json.load(open(results_file))
        logger.info(f"Resume: {len([k for k in results if k.startswith('round_')])} rounds already done")

    for prev_round in range(1, args.iterations + 1):
        prev_dir = os.path.join(args.data_dir, f"round_{prev_round}")
        if f"round_{prev_round}" in results and os.path.isdir(prev_dir):
            dataset_dirs.append(prev_dir)
            prev_train = os.path.join(prev_dir, "train", "raw", "train.json")
            if os.path.exists(prev_train):
                for line in open(prev_train):
                    d = json.loads(line)
                    if d[2] == 1:
                        existing_deps.setdefault(d[0], set()).add(d[1])

    all_proved = set()
    for iteration in range(1, args.iterations + 1):
        if f"round_{iteration}" in results:

            mf = os.path.join(args.data_dir, f"cascade/round_{iteration}/cascade_merged.json")
            if os.path.exists(mf):
                md = json.load(open(mf))
                all_proved |= {k for k, v in md.items() if v.get('proved')}
            logger.info(f"\n### Round {iteration}: done (cumulative={len(all_proved)}), skipping ###")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"### Round {iteration} ###")
        logger.info(f"{'='*60}")

        round_dir = os.path.join(args.data_dir, f"round_{iteration}")
        model_save = os.path.join(args.data_dir, f"models/round_{iteration}")
        ranking_path = os.path.join(args.data_dir, f"rankings/round_{iteration}.json")
        cascade_dir = os.path.join(args.data_dir, f"cascade/round_{iteration}")

        if iteration == 1 and args.init_model:
            logger.info(f"[1/4] Using existing model: {args.init_model}")
            os.makedirs(model_save, exist_ok=True)
            dst = os.path.join(model_save, "averaged_top5.pt")
            if not os.path.exists(dst):
                shutil.copy2(args.init_model, dst)
        else:
            prev_model = None
            if iteration > 1:
                prev_model = os.path.join(args.data_dir,
                    f"models/round_{iteration-1}/averaged_top5.pt")
                if not os.path.exists(prev_model):
                    prev_model = None
            logger.info(f"[1/4] Training (datasets: {len(dataset_dirs)}"
                        f"{', fine-tune' if prev_model else ''})")
            train_model(dataset_dirs, model_save, args.dim, args.layers,
                        args.epochs, args.device, logger,
                        init_model=prev_model, lr=args.lr,
                        batch_size=args.batch_size,
                        reg=args.reg if args.model_type == 'CL' else None,
                        hyper=args.hyper if args.model_type == 'CL' else None)

        logger.info(f"[2/4] Ranking")
        generate_ranking(f"{model_save}/averaged_top5.pt",
                         args.dim, args.layers, ranking_path, args.device, logger)

        logger.info(f"[3/4] Cascade evaluation")
        proved_count = run_cascade(ranking_path, cascade_dir, args.n_shards, logger)

        mf = os.path.join(cascade_dir, "cascade_merged.json")
        if os.path.exists(mf):
            md = json.load(open(mf))
            round_proved = {k for k, v in md.items() if v.get('proved')}
            new_proved = round_proved - all_proved
            all_proved |= round_proved
            logger.info(f"  Round={proved_count}, new={len(new_proved)}, cumulative={len(all_proved)}")

        results[f"round_{iteration}"] = len(all_proved)
        json.dump(results, open(results_file, 'w'), indent=2)

        logger.info(f"[4/4] Extracting new deps")
        new_pos, hard_neg = extract_new_deps(cascade_dir, ranking_path, existing_deps, logger)

        if not new_pos and not hard_neg:
            logger.info("  No new data, loop converged")
        else:
            gen_incremental_dataset(new_pos, hard_neg, round_dir, logger)
            dataset_dirs.append(round_dir)
            for conj, prems in new_pos.items():
                existing_deps.setdefault(conj, set()).update(prems)

        logger.info(f"  Cumulative training datasets: {len(dataset_dirs)}")

    logger.info(f"\n{'='*60}")
    logger.info(f"{args.model_type} feedback loop complete")
    logger.info(f"{'='*60}")
    for k, v in sorted(results.items()):
        if k.startswith('round_'):
            logger.info(f"  {k}: cumulative {v}/280")
    if results:
        vals = [v for k, v in results.items() if k.startswith('round_')]
        logger.info(f"  Final cumulative: {max(vals)}/280")

if __name__ == '__main__':
    main()

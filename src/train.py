"""Leakage-controlled CPU benchmarks for METLIN SMRT.

Tune only on validation labels. Persist selected hyperparameters before reading
test labels for scoring. All saved predictions are in seconds.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / '.runtime', ROOT / '.torch_runtime'):
    if folder.exists():
        sys.path.insert(0, str(folder))
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import numpy as np
import pandas as pd
import joblib
import sklearn
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits

SPEC = {
    'protocols': ['random', 'scaffold'], 'training_seeds': [17, 42, 73],
    'tuning_seed': 17, 'selection_metric': 'validation MAE in seconds',
    'rf': {'n_estimators': 160, 'max_features': 0.33,
           'min_samples_leaf_candidates': [1, 3], 'max_depth': None},
    'mlp': {'hidden_units': [1000, 500, 200, 100], 'activation': 'ReLU',
            'learning_rate_candidates': [0.001, 0.0003],
            'batch_size': 512, 'max_epochs': 40, 'patience': 7,
            'min_delta_mae_seconds': 0.05, 'hidden_kernel_l2': 0.0001,
            'loss': 'MSE on training-standardized target plus hidden-kernel L2',
            'optimizer': 'Adam', 'target_scaling': 'training mean and population std',
            'checkpoint_rule': 'lowest validation MAE', 'prediction_clipping': False},
    'dummy': 'training median', 'test_use': 'score only after hyperparameters are locked',
    'refit_on_train_plus_validation': False,
    'notes': 'Architecture-inspired reimplementation, not exact original replication.'
}


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def log(text):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), text, flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        widths = [1024] + SPEC['mlp']['hidden_units']
        for a, b in zip(widths[:-1], widths[1:]):
            layers += [torch.nn.Linear(a, b), torch.nn.ReLU()]
        layers += [torch.nn.Linear(widths[-1], 1)]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).squeeze(1)

    def kernel_penalty(self):
        return sum(m.weight.square().sum() for m in self.layers[:-1]
                   if isinstance(m, torch.nn.Linear))


@torch.inference_mode()
def predict_mlp(model, x, mu, sd):
    model.eval()
    chunks = []
    for i in range(0, len(x), 2048):
        out = model(torch.from_numpy(np.asarray(x[i:i + 2048], dtype=np.float32)))
        chunks.append(out.numpy() * sd + mu)
    return np.concatenate(chunks)


def fit_mlp(x_train, y_train, x_val, y_val, seed, learning_rate, output, quick_epochs=None):
    seed_all(seed)
    model = Network()
    mu, sd = float(y_train.mean()), float(y_train.std())#normalization
    if sd <= 0:
        raise ValueError('Training target has zero variance')
    x_t = torch.from_numpy(np.asarray(x_train, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray((y_train - mu) / sd, dtype=np.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    cfg = SPEC['mlp']
    max_epochs = quick_epochs or cfg['max_epochs']
    best_mae, patience_best, best_epoch, stale = float('inf'), float('inf'), 0, 0
    best_state, history = None, []
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(x_t))
        total, squared = 0., 0.
        epoch_start = time.perf_counter()
        for begin in range(0, len(order), cfg['batch_size']):
            ix = order[begin:begin + cfg['batch_size']]
            optimizer.zero_grad(set_to_none=True)#梯度下降找最优模型参数
            prediction = model(x_t[ix])
            mse = (prediction - y_t[ix]).square().mean()
            loss = mse + cfg['hidden_kernel_l2'] * model.kernel_penalty() #损失函数
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ix)
            squared += float(mse.detach()) * len(ix)
        val_pred = predict_mlp(model, x_val, mu, sd)
        mae = float(mean_absolute_error(y_val, val_pred))
        improved = mae < best_mae
        # Store the true minimum; patience counter uses the declared minimum change.
        if mae < patience_best - cfg['min_delta_mae_seconds']:
            stale = 0
            patience_best = mae
        else:
            stale += 1
        if improved:
            best_mae, best_epoch = mae, epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append({'epoch': epoch, 'train_loss': total / len(x_t),
                        'train_standardized_mse': squared / len(x_t),
                        'validation_mae_s': mae,
                        'epoch_seconds': time.perf_counter() - epoch_start})
        log(f'MLP seed={seed} lr={learning_rate} epoch={epoch} val_MAE={mae:.3f}s '
            f'epoch_time={history[-1]["epoch_seconds"]:.1f}s')
        if stale >= cfg['patience']:
            break
    model.load_state_dict(best_state)
    metadata = {'model': 'mlp', 'seed': seed, 'learning_rate': learning_rate,
                'validation_mae': best_mae, 'best_epoch': best_epoch,
                'epochs_run': len(history), 'train_seconds': time.perf_counter() - start,
                'target_mean': mu, 'target_std': sd, 'config': cfg}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': best_state, 'metadata': metadata}, output.with_suffix('.pt'))
    save_json(output.with_suffix('.json'), metadata)
    pd.DataFrame(history).to_csv(output.with_suffix('.history.csv'), index=False)
    return metadata


def fit_rf(x_train, y_train, x_val, y_val, seed, min_leaf, output, threads, quick_trees=None):
    cfg = SPEC['rf']
    model = RandomForestRegressor(n_estimators=quick_trees or cfg['n_estimators'],
                                 max_features=cfg['max_features'], max_depth=cfg['max_depth'],
                                 min_samples_leaf=min_leaf, n_jobs=threads, random_state=seed)
    start = time.perf_counter()
    log(f'RF seed={seed} min_samples_leaf={min_leaf} fitting {model.n_estimators} trees')
    model.fit(x_train, y_train)
    mae = float(mean_absolute_error(y_val, model.predict(x_val)))
    metadata = {'model': 'rf', 'seed': seed, 'min_samples_leaf': min_leaf,
                'validation_mae': mae, 'train_seconds': time.perf_counter() - start,
                'n_estimators': model.n_estimators, 'max_features': cfg['max_features'],
                'max_depth': cfg['max_depth']}
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output.with_suffix('.joblib'), compress=3)
    save_json(output.with_suffix('.json'), metadata)
    log(f'RF complete val_MAE={mae:.3f}s elapsed={metadata["train_seconds"]:.1f}s')
    return metadata


def load_data(root, protocol):#数据加载与防止数据泄漏
    x = np.load(root / 'processed/X.npy', mmap_mode='r')
    frame = pd.read_csv(root / 'processed/molecules.csv')
    splits = pd.read_csv(root / f'splits/{protocol}.csv')
    assert frame.row_id.tolist() == list(range(len(frame)))
    assert not splits.row_id.duplicated().any()
    assert set(splits.row_id) == set(frame.row_id)
    split = splits.set_index('row_id').loc[frame.row_id, 'split'].to_numpy()
    train, val = np.flatnonzero(split == 'train'), np.flatnonzero(split == 'val')
    test = np.flatnonzero(split == 'test')
    for key in ['structure_group'] + (['scaffold_group'] if protocol == 'scaffold' else []):
        groups = [set(frame.iloc[ix][key].astype(str)) for ix in [train, val, test]]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    return x, frame, train, val, test


def train_protocol(root, protocol, threads, stage):#先用seed17训练，找到RF 和 MLP 的最佳配置。
    x, frame, train, val, _ = load_data(root, protocol)
    x_tr, x_va = np.asarray(x[train], dtype=np.float32), np.asarray(x[val], dtype=np.float32)
    # Access only train and validation target vectors in training stages.
    y_tr = frame.iloc[train].rt.to_numpy(dtype=np.float64)
    y_va = frame.iloc[val].rt.to_numpy(dtype=np.float64)
    model_dir = root / 'outputs/models'
    selection_path = root / f'outputs/{protocol}_hyperparameters.json'
    if stage == 'tune':
        candidates = []
        for leaf in SPEC['rf']['min_samples_leaf_candidates']:
            out = model_dir / f'{protocol}_rf_tune_leaf{leaf}'
            if out.with_suffix('.json').exists() and out.with_suffix('.joblib').exists():
                record = json.loads(out.with_suffix('.json').read_text())
            else:
                record = fit_rf(x_tr, y_tr, x_va, y_va, 17, leaf, out, threads)
            record['artifact_stem'] = out.name
            candidates.append(record)
        for lr in SPEC['mlp']['learning_rate_candidates']:
            out = model_dir / f'{protocol}_mlp_tune_lr{lr}'.replace('.', 'p')
            if out.with_suffix('.json').exists() and out.with_suffix('.pt').exists():
                record = json.loads(out.with_suffix('.json').read_text())
            else:
                record = fit_mlp(x_tr, y_tr, x_va, y_va, 17, lr, out)
            record['artifact_stem'] = out.name
            candidates.append(record)
        selection = {m: min([r for r in candidates if r['model'] == m],
                            key=lambda r: r['validation_mae']) for m in ['rf', 'mlp']}
        selection['locked_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        selection['all_candidates'] = candidates
        save_json(selection_path, selection)
        return
    selection = json.loads(selection_path.read_text())
    import shutil
    for name in ['rf', 'mlp']:
        chosen = selection[name]
        source = model_dir / chosen['artifact_stem']
        for seed in SPEC['training_seeds']:
            out = model_dir / f'{protocol}_{name}_seed{seed}'
            suffix = '.joblib' if name == 'rf' else '.pt'
            if out.with_suffix('.json').exists() and out.with_suffix(suffix).exists():
                continue
            if seed == 17:
                for ext in [suffix, '.json'] + (['.history.csv'] if name == 'mlp' else []):
                    shutil.copyfile(source.with_suffix(ext), out.with_suffix(ext))
            elif name == 'rf':
                fit_rf(x_tr, y_tr, x_va, y_va, seed, chosen['min_samples_leaf'], out, threads)
            else:
                fit_mlp(x_tr, y_tr, x_va, y_va, seed, chosen['learning_rate'], out)


def evaluate(root, threads):#测试集评估，这个函数主要完成以下工作：1.从保存的超参数记录中选择验证集 MAE 更低的模型类型。2.加载每个模型对应的训练结果。3.在测试集上生成预测。4.将真实值与预测值保存到 CSV 文件。
    (root / 'outputs/predictions').mkdir(parents=True, exist_ok=True)
    # Choices must already exist for both protocols. No test score enters selection.
    selected = {}
    for protocol in SPEC['protocols']:
        choice = json.loads((root / f'outputs/{protocol}_hyperparameters.json').read_text())
        best = min([choice[m] for m in ['rf', 'mlp']], key=lambda x: x['validation_mae'])
        selected[protocol] = {'model': best['model'], 'validation_mae': best['validation_mae'],
                              'selection_seed': 17, 'hyperparameters_locked_at': choice['locked_at']}
    save_json(root / 'outputs/selected_model.json', selected)
    for protocol in SPEC['protocols']:
        x, frame, train, val, test = load_data(root, protocol)
        x_te = np.asarray(x[test], dtype=np.float32)
        for name in ['dummy', 'rf', 'mlp']:
            for seed in ([17] if name == 'dummy' else SPEC['training_seeds']):
                stem = root / f'outputs/models/{protocol}_{name}_seed{seed}'
                if name == 'dummy':
                    value = float(frame.iloc[train].rt.median())
                    pred = np.full(len(test), value)
                    save_json(stem.with_suffix('.json'), {'model': 'dummy', 'median_seconds': value})
                elif name == 'rf':
                    model = joblib.load(stem.with_suffix('.joblib'))
                    model.n_jobs = threads
                    pred = model.predict(x_te)
                else:
                    checkpoint = torch.load(stem.with_suffix('.pt'), map_location='cpu', weights_only=True)
                    model = Network()
                    model.load_state_dict(checkpoint['state_dict'])
                    meta = checkpoint['metadata']
                    pred = predict_mlp(model, x_te, meta['target_mean'], meta['target_std'])
                result = pd.DataFrame({'row_id': frame.iloc[test].row_id.to_numpy(),
                                       'pubchem': frame.iloc[test].pubchem.to_numpy(),
                                       'y_true': frame.iloc[test].rt.to_numpy(), 'y_pred': pred})
                assert np.isfinite(pred).all()
                result.to_csv(root / f'outputs/predictions/{protocol}_{name}_seed{seed}.csv', index=False)
                log(f'Test predictions saved: {protocol} {name} seed={seed}')
    save_json(root / 'outputs/evaluation_completed.json', {'completed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                                                         'test_predictions_written': 14})


def benchmark(root, threads):
    x, frame, train, val, _ = load_data(root, 'random')
    # Use a training subset and validation subset only; never inspect test performance.
    ix, iv = train[:8192], val[:2048]
    out = root / 'outputs/logs/benchmark_mlp'
    fit_mlp(x[ix], frame.iloc[ix].rt.to_numpy(), x[iv], frame.iloc[iv].rt.to_numpy(),
            17, 0.001, out, quick_epochs=2)
    fit_rf(x[ix], frame.iloc[ix].rt.to_numpy(), x[iv], frame.iloc[iv].rt.to_numpy(),
           17, 1, root / 'outputs/logs/benchmark_rf', threads, quick_trees=20)


def protocol_worker(root, protocol, threads, stage):
    # Independent processes preserve each protocol's RNG while using the CPU.
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=threads):
        log(f'Protocol worker started: {protocol}; threads={threads}')
        if stage in ['all', 'tune']:
            train_protocol(root, protocol, threads, 'tune')
        if stage in ['all', 'fit']:
            train_protocol(root, protocol, threads, 'fit')
        log(f'Protocol worker complete: {protocol}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=['all', 'benchmark', 'tune', 'fit', 'evaluate'], default='all')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2,
                        help='Independent protocol processes; each uses --threads CPU threads')
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        raise ValueError('Choose 1 to 16 CPU threads')
    root = args.project_root.resolve()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.stage != 'benchmark':
        identities = {}
        for relative in ['processed/X.npy', 'processed/molecules.csv',
                         'splits/random.csv', 'splits/scaffold.csv']:
            h = hashlib.sha256()
            with (root / relative).open('rb') as handle:
                for chunk in iter(lambda: handle.read(2 ** 20), b''):
                    h.update(chunk)
            identities[relative] = h.hexdigest()
        identity_path = root / 'outputs/input_identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != identities:
            raise RuntimeError('Data or split hashes changed; refusing to reuse existing models.')
        save_json(identity_path, identities)
        spec_path = root / 'outputs/experiment_spec.json'
        if spec_path.exists() and json.loads(spec_path.read_text()) != SPEC:
            raise RuntimeError('Existing experiment spec differs; use a new output directory.')
        save_json(spec_path, SPEC)
    save_json(root / 'outputs/logs/environment.json', {
        'python': sys.version, 'platform': platform.platform(),
        'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
        'torch': torch.__version__, 'threads_per_worker': args.threads,
        'protocol_workers': args.workers, 'device': 'cpu'})
    with threadpool_limits(limits=args.threads):
        if args.stage == 'benchmark':
            benchmark(root, args.threads)
            return
        if args.stage in ['all', 'tune', 'fit']:
            # All workers finish selection/fitting before any test predictions.
            if args.workers == 2:
                with ProcessPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(protocol_worker, root, p, args.threads, args.stage)
                               for p in SPEC['protocols']]
                    for future in futures:
                        future.result()
            else:
                for p in SPEC['protocols']:
                    if args.stage in ['all', 'tune']:
                        train_protocol(root, p, args.threads, 'tune')
                    if args.stage in ['all', 'fit']:
                        train_protocol(root, p, args.threads, 'fit')
        if args.stage in ['all', 'evaluate']:
            evaluate(root, args.threads)


if __name__ == '__main__':
    main()

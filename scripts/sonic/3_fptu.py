#!/usr/bin/env python
"""
3_fptu.py

Single-cell FPTU: extends the core-gene network (r, from 2_rols.py) to the
DPD modules. Kept as its own script, isolated from 2_rols.py, specifically
because this is the step with a segfault history.

Rebuilds the same PerturbSeq object as 2_rols.py (cheap, the expensive
DPD computation already happened once in 1_prep_stv.py and is baked into
the saved combined_expr checkpoint), then calls generate_mra_input(fptu=True)
and cstarpy's run_regularized_mra_fptu. Connections FROM DPD modules are not
built by hand here: run_regularized_mra_fptu's own _assemble_inputs starts
every entry as forbidden and only unblocks the core-genes-to-DPD block, so
"no connections from DPD" is the function's default behaviour.

Naming convention: r/r_fptu/r_total (local response, lowercase)
and r_minv_total (its pseudoinverse reconstruction, still lowercase since it
is derived from inference). Replaces the old G_total name.

"""
import argparse
import os
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.sparse


def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)


parser = argparse.ArgumentParser()
parser.add_argument('--condition', required=True, choices=['Rest', 'Stim8hr', 'Stim48hr'])
parser.add_argument('--donors', default='D1_D2_D3_D4')
parser.add_argument('--popsize', type=int, default=10)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--method', default='ols', choices=['odr', 'ols', 'br'],
                    help='Regression method for run_regularized_mra_fptu.')
parser.add_argument('--ckpt_dir', required=True,
                    help='checkpoints/{CONDITION} -- input from 1_prep_stv.py and '
                         '2_rols.py (r_rols), output of this script too.')
args = parser.parse_args()

CONDITION = args.condition
run_tag = f'{CONDITION}_{args.donors}'
os.makedirs(args.ckpt_dir, exist_ok=True)

log(f'Condition: {CONDITION}')
log(f'run_tag: {run_tag}')
log(f'popsize: {args.popsize}')
log(f'FPTU method: {args.method}')
log(f'Checkpoint dir (input + output): {args.ckpt_dir}')

# Load checkpoints from 1_prep_stv.py and 2_rols.py
log('Loading checkpoint from 1_prep_stv.py...')
mat = scipy.sparse.load_npz(os.path.join(args.ckpt_dir, f'combined_expr_{run_tag}.npz'))
genes_ = pd.read_csv(os.path.join(args.ckpt_dir, f'combined_genes_{run_tag}.csv'))['gene'].tolist()
cells_ = pd.read_csv(os.path.join(args.ckpt_dir, f'combined_cells_{run_tag}.csv'))['cell_id'].tolist()
combined_df = pd.DataFrame(mat.toarray(), index=genes_, columns=cells_)
log(f'combined_df shape: {combined_df.shape} (DPD_stim row already included)')

ntc_cell_ids = pd.read_csv(os.path.join(args.ckpt_dir, f'ntc_cell_ids_{run_tag}.csv'))['cell_id'].tolist()
perturbed_gene_list = pd.read_csv(
    os.path.join(args.ckpt_dir, f'perturbed_gene_list_{run_tag}.csv'))['gene'].tolist()
guide_df = pd.read_csv(os.path.join(args.ckpt_dir, f'guide_assignments_{run_tag}.csv'))
guide_assignments_dict = {row['cell_id']: [row['gene']] for _, row in guide_df.iterrows()}

log('Loading r_rols from 2_rols.py...')
r_rols = pd.read_csv(os.path.join(args.ckpt_dir, f'r_rols_{run_tag}.csv'), index_col=0)
log(f'r_rols shape: {r_rols.shape}')

dpd_btla_path = os.path.join(args.ckpt_dir, f'dpd_btla_per_cell_{run_tag}.csv')
if os.path.exists(dpd_btla_path):
    dpd_btla_per_cell = pd.read_csv(dpd_btla_path, index_col=0)['DPD_cell']
    dpd_btla_row = pd.Series(0.0, index=combined_df.columns, dtype=np.float32)
    dpd_btla_row.loc[dpd_btla_per_cell.index] = dpd_btla_per_cell.values.astype(np.float32)
    DPD_EPS = 1e-3
    dpd_btla_row.loc[ntc_cell_ids] = DPD_EPS
    combined_df.loc['DPD_btla'] = dpd_btla_row
    log(f'DPD_btla row added (NTC placeholder, DPD_EPS={DPD_EPS}).')
else:
    log(f'{dpd_btla_path} not found, proceeding with DPD_stim only.')

# Build PerturbSeq object
perturbed_gene_set = set(perturbed_gene_list)
guide_assignments_dict = {
    cell: genes for cell, genes in guide_assignments_dict.items()
    if genes[0] in perturbed_gene_set
}
valid_pert_cells = set(guide_assignments_dict.keys())
valid_cols = [c for c in combined_df.columns if c in valid_pert_cells or c in set(ntc_cell_ids)]
combined_df = combined_df[valid_cols]
log(f'combined_df: {combined_df.shape}')

log('Building PerturbSeq object...')
from cstarpy.preprocessing import PerturbSeq
pert_obj = PerturbSeq(
    data=combined_df,
    perturbed_genes=perturbed_gene_list,
    cells_ntc=ntc_cell_ids,
    guide_assignments=guide_assignments_dict,
    excluded_targets=[],
    dpd_prefix='DPD_')
log(f'dpd_modules detected: {pert_obj.dpd_modules}')

# FPTU: generate_mra_input(fptu=True)
log(f'Running generate_mra_input (fptu=True, popsize={args.popsize})...')
pert_obj.generate_mra_input(
    fptu=True,
    popsize=args.popsize,
    show_progress_bar=True,
    random_seed=args.random_seed)

mra_input_fptu = pert_obj.mra_input
index_pert = list(mra_input_fptu.index_pert)
index_unpert = list(mra_input_fptu.index_unpert)
log(f'FPTU R shape: {mra_input_fptu.R.shape}')
log(f'index_pert: {len(index_pert)} nodes')
log(f'index_unpert: {index_unpert}')

data_perturbed_fptu = mra_input_fptu.R
data_unperturbed_fptu = mra_input_fptu.R_unpert
pert_fptu = mra_input_fptu.pert.astype(bool)

log(f'data_perturbed_fptu shape: {data_perturbed_fptu.shape}')
log(f'data_unperturbed_fptu shape: {data_unperturbed_fptu.shape}')
if data_unperturbed_fptu is None or data_unperturbed_fptu.shape[0] == 0:
    raise RuntimeError(
        f'DPD rows missing from FPTU MraInput (index_unpert={index_unpert}). '
        f'Check that DPD_* rows are present in combined_df and dpd_prefix is set correctly.')

import cstarpy.inference

n_pert = len(index_pert)
G_not_fptu = np.zeros((len(index_unpert), n_pert), dtype=bool)

# Scale the unperturbed (DPD) rows by their own max absolute value before FPTU, then
# rescale the result back afterward. Without this, the scale mismatch between data_unperturbed_fptu
# (DPD, pseudoreplica-aggregated) and data_perturbed_fptu (gene expression) can
# destabilise the regression.
dpd_scale = np.abs(data_unperturbed_fptu).max()
data_unperturbed_fptu_scaled = data_unperturbed_fptu / dpd_scale
log(f'DPD scale factor: {dpd_scale:.4f}')


# threshold is a convergence parameter, not biological: auto-retry lower if
# the iterative pruning discards every connection before convergence.
def run_fptu_with_retry(data_perturbed, data_unperturbed, pert, G_not, method, label,
                        thresholds=(0.1, 0.05, 0.01, 0.005, 0.001)):
    for t in thresholds:
        try:
            arr = cstarpy.inference.run_regularized_mra_fptu(
                data_perturbed=data_perturbed,
                data_unperturbed=data_unperturbed,
                pert=pert,
                G_not=G_not,
                method=method,
                threshold=t,
                pvalue_threshold=0.4)
            log(f'{label}: converged at threshold={t}')
            return arr, t
        except ValueError as e:
            if 'threshold' not in str(e):
                raise
            log(f'{label}: threshold={t} failed ({e}), retrying lower...')
    raise RuntimeError(f'{label}: FPTU did not converge for any threshold in {thresholds}')


log(f"Running FPTU (cstarpy's run_regularized_mra_fptu, method='{args.method}')...")
r_fptu_arr, used_threshold = run_fptu_with_retry(
    data_perturbed_fptu, data_unperturbed_fptu_scaled, pert_fptu, G_not_fptu, args.method, 'FPTU')
r_fptu_arr = r_fptu_arr * dpd_scale

r_fptu = pd.DataFrame(r_fptu_arr, index=index_unpert, columns=index_pert)
log(f'r_fptu shape: {r_fptu.shape}')
log(f'r_fptu range: [{r_fptu_arr.min():.4f}, {r_fptu_arr.max():.4f}]')
r_fptu.to_csv(os.path.join(args.ckpt_dir, f'r_fptu_{run_tag}.csv'))

# Assemble r_total / r_minv_total (core genes + DPD nodes)
all_nodes = index_pert + index_unpert
n_total = len(all_nodes)
n_core = n_pert

r_arr_diag = r_rols.reindex(index=index_pert, columns=index_pert).values.copy()
np.fill_diagonal(r_arr_diag, -1.0)

r_total_arr = np.zeros((n_total, n_total))
r_total_arr[:n_core, :n_core] = r_arr_diag
r_total_arr[n_core:, :n_core] = r_fptu_arr
np.fill_diagonal(r_total_arr, -1.0)

r_total = pd.DataFrame(r_total_arr, index=all_nodes, columns=all_nodes)
r_minv_total_arr = -np.linalg.pinv(r_total_arr)
np.fill_diagonal(r_minv_total_arr, 0.0)
r_minv_total = pd.DataFrame(r_minv_total_arr, index=all_nodes, columns=all_nodes)

r_total.to_csv(os.path.join(args.ckpt_dir, f'r_total_{run_tag}.csv'))
r_minv_total.to_csv(os.path.join(args.ckpt_dir, f'r_minv_total_{run_tag}.csv'))
log(f'r_total / r_minv_total saved. r_minv_total range: '
    f'[{r_minv_total_arr.min():.4f}, {r_minv_total_arr.max():.4f}]')

with open(os.path.join(args.ckpt_dir, f'fptu_threshold_used_{run_tag}.txt'), 'w') as f:
    f.write(str(used_threshold))
with open(os.path.join(args.ckpt_dir, f'fptu_method_used_{run_tag}.txt'), 'w') as f:
    f.write(args.method)

log('3_fptu.py complete.')

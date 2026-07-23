#!/usr/bin/env python
"""
2_rols.py

Single-cell ROLS network inference on the core genes. Replaces the ROLS half
of script_2_rols_fptu.py. FPTU is now a separate script (3_fptu.py) so a
crash there does not require rerunning this (the expensive) part.

Loads the checkpoint from 1_prep_stv.py (combined_expr already includes the
DPD_stim row, and DPD_btla too if an external dpd_btla_per_cell_{run_tag}.csv
happens to exist), builds the PerturbSeq object, and runs cstarpy's own
run_jax_mra directly, no custom chunked fallback. popsize=10 (cstarpy
default, per Hiroaki's guidance) keeps generate_mra_input's output small
enough (~1M columns) that run_jax_mra should handle it directly.

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
parser.add_argument('--popsize', type=int, default=10,
                    help='generate_mra_input popsize. Multiplies by the number of cells '
                         'per perturbation group, not a flat replicate count -- keep low '
                         'at single-cell scale. cstarpy default is 10.')
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--ckpt_dir', required=True,
                    help='checkpoints/{CONDITION} -- input from 1_prep_stv.py, output of this script too.')
args = parser.parse_args()

CONDITION = args.condition
run_tag = f'{CONDITION}_{args.donors}'
os.makedirs(args.ckpt_dir, exist_ok=True)

log(f'Condition: {CONDITION}')
log(f'run_tag: {run_tag}')
log(f'popsize: {args.popsize}')
log(f'Checkpoint dir (input + output): {args.ckpt_dir}')

# Load checkpoint from 1_prep_stv.py
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

log(f'Core genes: {len(perturbed_gene_list)}')

# DPD_btla is optional and not produced by 1_prep_stv.py - pick it up only if
# it has been provided externally (BTLA is bulk/pseudobulk sourced, not
# single-cell; unchanged gap from before).
dpd_btla_path = os.path.join(args.ckpt_dir, f'dpd_btla_per_cell_{run_tag}.csv')
if os.path.exists(dpd_btla_path):
    dpd_btla_per_cell = pd.read_csv(dpd_btla_path, index_col=0)['DPD_cell']
    dpd_btla_row = pd.Series(0.0, index=combined_df.columns, dtype=np.float32)
    dpd_btla_row.loc[dpd_btla_per_cell.index] = dpd_btla_per_cell.values.astype(np.float32)
    DPD_EPS = 1e-3
    dpd_btla_row.loc[ntc_cell_ids] = DPD_EPS
    combined_df.loc['DPD_btla'] = dpd_btla_row
    log(f'DPD_btla row added (NTC placeholder, DPD_EPS={DPD_EPS}). '
        f'Perturbed-cell range: [{dpd_btla_per_cell.min():.4f}, {dpd_btla_per_cell.max():.4f}]')
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
log(f'After guide_assignments filter: {len(guide_assignments_dict):,} perturbed cells, '
    f'{len(ntc_cell_ids):,} NTC cells, combined_df: {combined_df.shape}')

log('Building PerturbSeq object...')
from cstarpy.preprocessing import PerturbSeq
pert_obj = PerturbSeq(
    data=combined_df,
    perturbed_genes=perturbed_gene_list,
    cells_ntc=ntc_cell_ids,
    guide_assignments=guide_assignments_dict,
    excluded_targets=[],
    dpd_prefix='DPD_')
log('PerturbSeq object created.')
log(f'dpd_modules detected: {pert_obj.dpd_modules}')

import jax
log(f'JAX devices: {jax.devices()}')

# ROLS: generate_mra_input(fptu=False) + run_jax_mra
log(f'Running generate_mra_input (fptu=False, popsize={args.popsize})...')
pert_obj.generate_mra_input(
    fptu=False,
    popsize=args.popsize,
    show_progress_bar=True,
    random_seed=args.random_seed)
R_mat = pert_obj.mra_input.R
pert_mask = pert_obj.mra_input.pert
n_bytes = R_mat.shape[0] * R_mat.shape[1] * 4
log(f'R_mat shape: {R_mat.shape} (~{n_bytes / 1e9:.2f}GB as float32), '
    f'range: [{R_mat.min():.3f}, {R_mat.max():.3f}]')

# ROLS runs on core genes only - exclude DPD rows from the square local-response matrix
R_mat_core = R_mat[:len(perturbed_gene_list), :]
pert_mask_core = pert_mask[:len(perturbed_gene_list), :]

log('Starting run_jax_mra (ROLS, core genes only, cstarpy original function)...')
from cstarpy.inference import run_jax_mra
r_arr = np.asarray(
    run_jax_mra(R_mat_core, pert_mask_core.astype(bool), show_progress_bar=True),
    dtype=np.float32).copy()
log('run_jax_mra complete.')
log(f'r range: [{r_arr.min():.4f}, {r_arr.max():.4f}]')

np.fill_diagonal(r_arr, -1.0)
r_rols = pd.DataFrame(r_arr, index=perturbed_gene_list, columns=perturbed_gene_list)
r_rols.to_csv(os.path.join(args.ckpt_dir, f'r_rols_{run_tag}.csv'))
np.save(os.path.join(args.ckpt_dir, f'r_rols_{run_tag}.npy'), r_arr)
log('r_rols saved.')

# r_arr already has diag=-1 (fixed above), no need to re-copy/re-fix it here
r_minv_arr = -np.linalg.pinv(r_arr)
r_minv_rols = pd.DataFrame(r_minv_arr, index=perturbed_gene_list, columns=perturbed_gene_list)
r_minv_rols.to_csv(os.path.join(args.ckpt_dir, f'r_minv_rols_{run_tag}.csv'))
log('r_minv_rols saved.')

log('2_rols.py complete.')

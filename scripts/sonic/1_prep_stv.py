#!/usr/bin/env python
"""
1_prep_stv.py

Single-cell STV/DPD computation for an externally pre-selected core gene set.
200 core genes are chosen from pseudobulk DE_stats and handed off as 
checkpoints/{CONDITION}/core_genes_{run_tag}.csv,
the same schema as the pseudobulk notebooks (column target_contrast_gene_name).

What this script still does (unchanged from before, independent of which
genes are core): trains the linear SVM STV on NTC cells (Rest vs stimulated),
computes per-cell DPD with re-centring against Rest NTC.

Single merged load: loads each donor's perturbed cells ONCE, with columns =
union(DE genes, core genes), and derives both DPD_cell (projecting onto the
DE-gene columns) and the expression matrix (core-gene columns) from that
single in-memory object.

Coverage diagnostics (new, warn-only): min_cells_per_gene / min_cells_per_donor
/ min_donors are no longer selection filters (the gene list is fixed
externally) -- they are logged per-gene as flags only. No gene is dropped
from the network on this basis. This is deliberate pending confirmation from
Oleksii on whether post-hoc exclusion makes sense here; flip to an actual
filter later if needed. Every flagged gene is logged individually (not just
a count) and also written to coverage_flagged_genes_{run_tag}.csv, so this
can be checked from the log/CSV without re-running anything.

The DPD_stim row is appended directly to the saved combined_expr matrix here
(not left to script 2/3 to rebuild), so 2_rols.py and 3_fptu.py can both load
the same checkpoint and construct an identical PerturbSeq object without
recomputing DPD. DPD_btla is NOT computed here (BTLA data is bulk/pseudobulk
sourced, not single-cell -- unchanged gap from before); 2_rols.py/3_fptu.py
still optionally pick up an externally-provided dpd_btla_per_cell_{run_tag}.csv
if one exists.

Use:
  python 1_prep_stv.py --condition Rest --data_dir ~/cSTAR_rols/Data \
      --ckpt_dir ~/cSTAR_rols/checkpoints/Rest \
      --vstim_ref_dir ~/cSTAR_rols/checkpoints/Stim8hr

For Stim8hr/Stim48hr, v_stim is trained directly (NTC of that condition vs
NTC of Rest, both loaded in this job). For Rest, v_stim is reused from the
Stim8hr run (Rest IS the NTC baseline) -- pass --vstim_ref_dir pointing at
the Stim8hr checkpoint directory.

--ckpt_dir is used as BOTH input (core_genes_{run_tag}.csv, copied here from
SBI's Results/sonic-inputs/ before running) and output (everything this
script saves).
"""
import argparse
import os
import gc
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.sparse
import anndata
import scanpy as sc
from sklearn.svm import LinearSVC


def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)


parser = argparse.ArgumentParser()
parser.add_argument('--condition', required=True, choices=['Rest', 'Stim8hr', 'Stim48hr'])
parser.add_argument('--donors', default='D1_D2_D3_D4')
parser.add_argument('--data_dir', required=True)
parser.add_argument('--ckpt_dir', required=True,
                    help='checkpoints/{CONDITION} -- both the input core_genes.csv '
                         'and all outputs of this script.')
parser.add_argument('--de_padj_threshold', type=float, default=0.05)
parser.add_argument('--de_logfc_threshold', type=float, default=0.25)
parser.add_argument('--max_de_genes', type=int, default=1000,
                    help='Maximum DE genes to use for SVM/STV training (top by |logFC| '
                         'after p_adj and logFC filters).')
parser.add_argument('--vstim_ref_dir', default=None,
                    help='For --condition Rest: path to the Stim8hr checkpoint dir '
                         'to reuse v_stim from. Required when --condition Rest.')
parser.add_argument('--min_cells_per_gene', type=int, default=100,
                    help='Coverage diagnostic threshold, WARN ONLY -- does not exclude '
                         'any gene from the network.')
parser.add_argument('--min_cells_per_donor', type=int, default=5,
                    help='Coverage diagnostic threshold, WARN ONLY.')
parser.add_argument('--min_donors', type=int, default=2,
                    help='Coverage diagnostic threshold, WARN ONLY.')
args = parser.parse_args()

CONDITION = args.condition
DONORS = args.donors.split('_')
run_tag = f'{CONDITION}_{args.donors}'
os.makedirs(args.ckpt_dir, exist_ok=True)

log(f'Condition: {CONDITION}')
log(f'Donors: {DONORS}')
log(f'Data dir: {args.data_dir}')
log(f'Checkpoint dir (input + output): {args.ckpt_dir}')
log(f'Coverage diagnostic thresholds (warn-only): min_cells_per_gene={args.min_cells_per_gene}, '
    f'min_cells_per_donor={args.min_cells_per_donor}, min_donors={args.min_donors}')

if CONDITION == 'Rest' and not args.vstim_ref_dir:
    raise ValueError('--vstim_ref_dir is required when --condition Rest '
                      '(Rest reuses v_stim computed during the Stim8hr run).')


def perturbed_path(donor, condition):
    return os.path.join(args.data_dir, f'{donor}_{condition}.assigned_guide.h5ad')


# gene symbol mapping (read once, cheap)
de_stats_tmp = anndata.read_h5ad(
    os.path.join(args.data_dir, 'GWCD4i.DE_stats.h5ad'), backed='r')
ensembl_to_symbol = dict(zip(
    de_stats_tmp.obs['target_contrast'].values,
    de_stats_tmp.obs['target_contrast_gene_name'].values))
de_stats_tmp.file.close()


def map_symbols(var_names):
    return [ensembl_to_symbol.get(e, e) for e in var_names]


# Load externally-selected core genes (pseudobulk, from SBI)
core_genes_path = os.path.join(args.ckpt_dir, f'core_genes_{run_tag}.csv')
log(f'Loading externally-selected core genes: {core_genes_path}')
core_genes_df = pd.read_csv(core_genes_path)
core_genes = core_genes_df['target_contrast_gene_name'].tolist()
log(f'Core genes (pseudobulk-selected on SBI): {len(core_genes)}')

# Pass 1: load NTC cells only (this condition)
log('Pass 1: loading NTC cells (this condition)...')
ntc_cond_parts = []
for donor in DONORS:
    fpath = perturbed_path(donor, CONDITION)
    tmp = anndata.read_h5ad(fpath, backed='r')
    ntc_mask_sc = tmp.obs['guide_type'] == 'non-targeting'
    cells = tmp[ntc_mask_sc].to_memory()
    tmp.file.close()
    cells.var_names = map_symbols(cells.var_names)
    cells.var_names_make_unique()
    ntc_cond_parts.append(cells)
    log(f'  {donor}: {cells.n_obs:,} NTC ({CONDITION})')
sc_ntc_cond = anndata.concat(ntc_cond_parts, label='donor', keys=DONORS)
del ntc_cond_parts
gc.collect()
log(f'Combined NTC ({CONDITION}): {sc_ntc_cond.shape}')

# STV: train linear SVM on NTC cells (Rest vs stimulated)
if CONDITION != 'Rest':
    log('Loading NTC cells for Rest (needed to train the Rest-vs-stim SVM)...')
    ntc_rest_parts = []
    for donor in DONORS:
        fpath = perturbed_path(donor, 'Rest')
        tmp = anndata.read_h5ad(fpath, backed='r')
        ntc_mask_sc = tmp.obs['guide_type'] == 'non-targeting'
        cells = tmp[ntc_mask_sc].to_memory()
        tmp.file.close()
        cells.var_names = map_symbols(cells.var_names)
        cells.var_names_make_unique()
        ntc_rest_parts.append(cells)
        log(f'  {donor}: {cells.n_obs:,} NTC (Rest)')
    sc_ntc_rest = anndata.concat(ntc_rest_parts, label='donor', keys=DONORS)
    del ntc_rest_parts
    gc.collect()
    log(f'Combined NTC (Rest): {sc_ntc_rest.shape}')

    common_genes_ntc = sc_ntc_rest.var_names.intersection(sc_ntc_cond.var_names)
    log(f'Common genes (NTC Rest vs NTC {CONDITION}): {len(common_genes_ntc):,}')
    sc_ntc_rest = sc_ntc_rest[:, common_genes_ntc].copy()
    sc_ntc_cond_aligned = sc_ntc_cond[:, common_genes_ntc].copy()

    log('Concatenating NTC Rest + NTC stim for DE feature selection...')
    ntc_combined = anndata.concat(
        [sc_ntc_rest, sc_ntc_cond_aligned],
        label='ntc_group', keys=['Rest', CONDITION])
    del sc_ntc_rest, sc_ntc_cond_aligned
    gc.collect()
    sc.pp.normalize_total(ntc_combined, target_sum=1e4)
    sc.pp.log1p(ntc_combined)
    log('Running rank_genes_groups (Wilcoxon, Rest vs stim)...')
    sc.tl.rank_genes_groups(
        ntc_combined, groupby='ntc_group', groups=[CONDITION], reference='Rest',
        method='wilcoxon')
    de_df = sc.get.rank_genes_groups_df(ntc_combined, group=CONDITION)
    de_mask = (de_df['pvals_adj'] < args.de_padj_threshold) & \
              (de_df['logfoldchanges'].abs() > args.de_logfc_threshold)
    de_df_filtered = de_df.loc[de_mask].copy()
    de_df_filtered = de_df_filtered.reindex(
        de_df_filtered['logfoldchanges'].abs().sort_values(ascending=False).index
    ).head(args.max_de_genes)
    de_genes = de_df_filtered['names'].tolist()
    log(f'DE genes (p_adj < {args.de_padj_threshold}, |logFC| > {args.de_logfc_threshold}, '
        f'top {args.max_de_genes} by |logFC|): {len(de_genes):,} / {len(de_df):,}')
    if len(de_genes) < 10:
        raise ValueError(f'Only {len(de_genes)} DE genes found, too few for stable SVM training. '
                          f'Consider relaxing --de_padj_threshold / --de_logfc_threshold.')

    log('Building SVM training matrix (all NTC cells, DE-filtered features)...')
    X_rest = ntc_combined[ntc_combined.obs['ntc_group'] == 'Rest', de_genes].X
    X_stim = ntc_combined[ntc_combined.obs['ntc_group'] == CONDITION, de_genes].X
    if scipy.sparse.issparse(X_rest):
        X_rest = X_rest.toarray()
    if scipy.sparse.issparse(X_stim):
        X_stim = X_stim.toarray()
    X_train = np.vstack([X_rest, X_stim]).astype(np.float32)
    y_train = np.concatenate([np.zeros(X_rest.shape[0]), np.ones(X_stim.shape[0])])
    log(f'SVM training matrix: {X_train.shape}, classes: {np.bincount(y_train.astype(int))}')

    log('Training linear SVM (LinearSVC, liblinear)...')
    svc = LinearSVC(dual='auto', max_iter=100_000, fit_intercept=True)
    svc.fit(X_train, y_train)
    log('SVM training complete.')

    coef = svc.coef_[0]
    norm_vec = coef / np.linalg.norm(coef)
    v_stim = pd.Series(norm_vec, index=de_genes)

    del ntc_combined, X_rest, X_stim, X_train
    gc.collect()

    v_stim.to_csv(os.path.join(args.ckpt_dir, f'v_stim_{run_tag}.csv'), header=['weight'])
    log(f'v_stim saved: {len(v_stim)} genes')
else:
    vstim_ref_path = os.path.join(args.vstim_ref_dir, f'v_stim_Stim8hr_{args.donors}.csv')
    log(f'Rest condition: loading v_stim from {vstim_ref_path}')
    v_stim = pd.read_csv(vstim_ref_path, index_col=0)['weight']
    de_genes = v_stim.index.tolist()

# Re-centring reference: mean DPD of Rest NTC cells
log('Computing DPD re-centring offset (mean DPD of Rest NTC cells)...')
if CONDITION == 'Rest':
    ntc_rest_for_recentre = sc_ntc_cond
    common_de_genes_ntc = [g for g in de_genes if g in ntc_rest_for_recentre.var_names]
else:
    ntc_rest_recentre_parts = []
    for donor in DONORS:
        fpath = perturbed_path(donor, 'Rest')
        tmp = anndata.read_h5ad(fpath, backed='r')
        ntc_mask_sc = tmp.obs['guide_type'] == 'non-targeting'
        tmp_var_symbols = map_symbols(tmp.var_names)
        de_genes_in_tmp = [g for g in de_genes if g in tmp_var_symbols]
        col_idx = [tmp_var_symbols.index(g) for g in de_genes_in_tmp]
        cells = tmp[ntc_mask_sc, col_idx].to_memory()
        cells.var_names = de_genes_in_tmp
        tmp.file.close()
        ntc_rest_recentre_parts.append(cells)
    ntc_rest_for_recentre = anndata.concat(ntc_rest_recentre_parts, label='donor', keys=DONORS)
    del ntc_rest_recentre_parts
    common_de_genes_ntc = [g for g in de_genes if g in ntc_rest_for_recentre.var_names]

X_ntc_rest = ntc_rest_for_recentre[:, common_de_genes_ntc].X
if scipy.sparse.issparse(X_ntc_rest):
    X_ntc_rest = X_ntc_rest.toarray()
X_ntc_rest = X_ntc_rest.astype(np.float32)
totals_ntc = X_ntc_rest.sum(axis=1, keepdims=True)
totals_ntc = np.where(totals_ntc == 0, 1.0, totals_ntc)
X_ntc_rest_norm = np.log1p(X_ntc_rest / totals_ntc * 1e4)
v_stim_for_ntc = v_stim.reindex(common_de_genes_ntc)
dpd_ntc_rest_raw = X_ntc_rest_norm @ v_stim_for_ntc.values
recentre_offset = -float(np.mean(dpd_ntc_rest_raw))
log(f'Re-centring offset: {recentre_offset:.4f} (mean DPD of {len(dpd_ntc_rest_raw):,} Rest NTC cells before centring)')

del X_ntc_rest, X_ntc_rest_norm
if CONDITION != 'Rest':
    del ntc_rest_for_recentre
gc.collect()

# Pass 1b: DPD_cell for this condition's own NTC cells
log("Computing DPD_cell for this condition's own NTC cells...")
common_de_genes_ntc_cond = [g for g in de_genes if g in sc_ntc_cond.var_names]
X_ntc_cond = sc_ntc_cond[:, common_de_genes_ntc_cond].X
if scipy.sparse.issparse(X_ntc_cond):
    X_ntc_cond = X_ntc_cond.toarray()
X_ntc_cond = X_ntc_cond.astype(np.float32)
totals_ntc_cond = X_ntc_cond.sum(axis=1, keepdims=True)
totals_ntc_cond = np.where(totals_ntc_cond == 0, 1.0, totals_ntc_cond)
X_ntc_cond_norm = np.log1p(X_ntc_cond / totals_ntc_cond * 1e4)
v_stim_for_ntc_cond = v_stim.reindex(common_de_genes_ntc_cond)
dpd_ntc_cond_raw = X_ntc_cond_norm @ v_stim_for_ntc_cond.values
dpd_ntc_cond = dpd_ntc_cond_raw + recentre_offset
dpd_ntc_per_cell = pd.Series(dpd_ntc_cond, index=sc_ntc_cond.obs_names, name='DPD_cell')
dpd_ntc_per_cell.to_csv(
    os.path.join(args.ckpt_dir, f'dpd_ntc_per_cell_{run_tag}.csv'), header=['DPD_cell'])
log(f'NTC per-cell DPD saved: {len(dpd_ntc_per_cell):,} cells, '
    f'range [{dpd_ntc_cond.min():.4f}, {dpd_ntc_cond.max():.4f}], '
    f'mean {dpd_ntc_cond.mean():.6f}')

del X_ntc_cond, X_ntc_cond_norm
gc.collect()

# Pass 2: load perturbed cells for core genes ONLY, single read
# Columns = union(DE genes, core genes), subset while still backed. DPD_cell is
# derived by projecting onto the DE-gene columns; the expression matrix is
# derived from the core-gene columns of the SAME loaded object -- no second
# disk read.
log('Pass 2: loading perturbed cells for core genes only (single read, DE + core gene columns)...')
union_genes_target = set(de_genes) | set(core_genes)

dpd_per_cell_parts = []
perturbed_core_parts = []
guide_assignments_dict = {}

for donor in DONORS:
    fpath = perturbed_path(donor, CONDITION)
    tmp = anndata.read_h5ad(fpath, backed='r')
    pert_mask_sc = tmp.obs['perturbed_gene_name'].isin(core_genes)
    tmp_var_symbols = map_symbols(tmp.var_names)
    union_genes_in_tmp = [g for g in union_genes_target if g in tmp_var_symbols]
    col_idx = [tmp_var_symbols.index(g) for g in union_genes_in_tmp]

    cells = tmp[pert_mask_sc, col_idx].to_memory()
    cells.var_names = union_genes_in_tmp
    cell_ids = cells.obs_names.tolist()
    perturbed_gene_names = cells.obs['perturbed_gene_name'].values
    tmp.file.close()

    de_genes_in_cells = [g for g in de_genes if g in cells.var_names]
    X_de = cells[:, de_genes_in_cells].X
    if scipy.sparse.issparse(X_de):
        X_de = X_de.toarray()
    X_de = X_de.astype(np.float32)
    totals = X_de.sum(axis=1, keepdims=True)
    totals = np.where(totals == 0, 1.0, totals)
    X_de_norm = np.log1p(X_de / totals * 1e4)
    v_stim_aligned_donor = v_stim.reindex(de_genes_in_cells)
    dpd_raw_donor = X_de_norm @ v_stim_aligned_donor.values
    dpd_donor = dpd_raw_donor + recentre_offset

    dpd_per_cell_parts.append(pd.DataFrame({
        'cell_id': cell_ids,
        'perturbed_gene': perturbed_gene_names,
        'donor': donor,
        'DPD_cell': dpd_donor,
    }))

    core_genes_in_cells = [g for g in core_genes if g in cells.var_names]
    perturbed_core_parts.append(cells[:, core_genes_in_cells].copy())

    for cell_id, gene_name in zip(cell_ids, perturbed_gene_names):
        guide_assignments_dict[cell_id] = [gene_name]

    log(f'  {donor}: {cells.n_obs:,} perturbed cells, '
        f'{len(de_genes_in_cells)} DE-gene cols, {len(core_genes_in_cells)} core-gene cols, '
        f'DPD range [{dpd_donor.min():.3f}, {dpd_donor.max():.3f}]')

    del cells, X_de, X_de_norm
    gc.collect()

dpd_per_cell_df = pd.concat(dpd_per_cell_parts, axis=0).reset_index(drop=True)
del dpd_per_cell_parts
gc.collect()
log(f'DPD_cell computed for {len(dpd_per_cell_df):,} perturbed cells total (core genes only).')
log(f'DPD_cell range: [{dpd_per_cell_df["DPD_cell"].min():.4f}, '
    f'{dpd_per_cell_df["DPD_cell"].max():.4f}]')

sc_perturbed_core = anndata.concat(perturbed_core_parts, label='donor', keys=DONORS)
del perturbed_core_parts
gc.collect()
log(f'Combined perturbed (core genes only): {sc_perturbed_core.shape}')

# Coverage diagnostics: WARN ONLY, no exclusion applied
# Logged gene-by-gene (not just a count) so this can be checked directly from
# the log file, or from coverage_flagged_genes_{run_tag}.csv, without
# re-running anything. Whether this should become an actual exclusion filter
# is an open question for Oleksii -- currently every core gene stays in the
# network regardless of these flags.
log('Coverage diagnostics for core genes (warn-only, no exclusion applied)...')
donor_gene_agg = dpd_per_cell_df.groupby(['perturbed_gene', 'donor']).agg(
    donor_n_cells=('DPD_cell', 'size')).reset_index()
qualifying_donor_counts = donor_gene_agg[
    donor_gene_agg['donor_n_cells'] >= args.min_cells_per_donor
].groupby('perturbed_gene').size()
total_cells_per_gene = dpd_per_cell_df.groupby('perturbed_gene').size()
ntc_gene_set = set(sc_ntc_cond.var_names)

flagged = []
for gene in core_genes:
    n_cells = int(total_cells_per_gene.get(gene, 0))
    n_donors = int(qualifying_donor_counts.get(gene, 0))
    in_ntc = gene in ntc_gene_set
    reasons = []
    if n_cells < args.min_cells_per_gene:
        reasons.append(f'n_cells={n_cells}<{args.min_cells_per_gene}')
    if n_donors < args.min_donors:
        reasons.append(f'n_donors={n_donors}<{args.min_donors}')
    if not in_ntc:
        reasons.append('absent from NTC data')
    if reasons:
        flagged.append((gene, n_cells, n_donors, in_ntc, '; '.join(reasons)))

log(f'{len(flagged)} / {len(core_genes)} core genes flagged by single-cell coverage thresholds:')
for gene, n_cells, n_donors, in_ntc, reasons in flagged:
    log(f'  FLAGGED {gene}: n_cells={n_cells}, n_donors={n_donors}, in_NTC={in_ntc} -- {reasons}')
if not flagged:
    log('  None -- all core genes meet the coverage thresholds.')

pd.DataFrame(
    flagged, columns=['gene', 'n_cells', 'n_donors', 'in_ntc', 'reasons']
).to_csv(os.path.join(args.ckpt_dir, f'coverage_flagged_genes_{run_tag}.csv'), index=False)

# Build combined_df (core genes + NTC), append DPD_stim row
perturbed_gene_list = [g for g in core_genes if g in sc_perturbed_core.obs['perturbed_gene_name'].values]
all_genes_core = sorted(set(core_genes) & set(sc_perturbed_core.var_names) & set(sc_ntc_cond.var_names))
log(f'Genes retained in checkpoint (core genes present in both perturbed and NTC data): {len(all_genes_core)}')
perturbed_gene_list = [g for g in perturbed_gene_list if g in all_genes_core]
if len(perturbed_gene_list) < len(core_genes):
    dropped = sorted(set(core_genes) - set(perturbed_gene_list))
    log(f'HARD DROP: {len(dropped)} core genes have zero usable data (absent from perturbed '
        f'or NTC space entirely, not a coverage-threshold flag): {dropped}')

perturbed_df = pd.DataFrame(
    sc_perturbed_core[:, all_genes_core].X.toarray()
    if scipy.sparse.issparse(sc_perturbed_core.X) else sc_perturbed_core[:, all_genes_core].X,
    index=sc_perturbed_core.obs_names, columns=all_genes_core).T
ntc_df = pd.DataFrame(
    sc_ntc_cond[:, all_genes_core].X.toarray()
    if scipy.sparse.issparse(sc_ntc_cond.X) else sc_ntc_cond[:, all_genes_core].X,
    index=sc_ntc_cond.obs_names, columns=all_genes_core).T

combined_df = pd.concat([perturbed_df, ntc_df], axis=1)
ntc_cell_ids = sc_ntc_cond.obs_names.tolist()

dpd_per_cell_aligned = dpd_per_cell_df.set_index('cell_id')['DPD_cell'].reindex(
    sc_perturbed_core.obs_names)

# Append DPD_stim row directly here so 2_rols.py/3_fptu.py can both load the
# same checkpoint and build an identical PerturbSeq object without recomputing.
log('Appending DPD_stim row to combined_df...')

dpd_stim_row = pd.Series(0.0, index=combined_df.columns, dtype=np.float32)
dpd_stim_row.loc[dpd_per_cell_aligned.index] = dpd_per_cell_aligned.values.astype(np.float32)
# NTC cells get a small constant placeholder instead of their true (near-zero, by
# construction) DPD value -- cstarpy's generate_mra_input divides by the NTC-averaged
# baseline (x0) for DPD modules, and a baseline landing near exactly zero for some
# pseudoreplica can blow up to extreme values by chance. Same fix already used for
# DPD_btla in 2_rols.py/3_fptu.py.
DPD_EPS = 1e-3
dpd_stim_row.loc[ntc_cell_ids] = DPD_EPS
combined_df.loc['DPD_stim'] = dpd_stim_row
log(f'combined_df shape after DPD_stim row: {combined_df.shape}')

# Save checkpoints
log('Saving checkpoints...')
scipy.sparse.save_npz(
    os.path.join(args.ckpt_dir, f'combined_expr_{run_tag}.npz'),
    scipy.sparse.csr_matrix(combined_df.values))
pd.Series(combined_df.index).to_csv(
    os.path.join(args.ckpt_dir, f'combined_genes_{run_tag}.csv'), index=False, header=['gene'])
pd.Series(combined_df.columns).to_csv(
    os.path.join(args.ckpt_dir, f'combined_cells_{run_tag}.csv'), index=False, header=['cell_id'])
pd.Series(ntc_cell_ids).to_csv(
    os.path.join(args.ckpt_dir, f'ntc_cell_ids_{run_tag}.csv'), index=False, header=['cell_id'])
pd.Series(perturbed_gene_list).to_csv(
    os.path.join(args.ckpt_dir, f'perturbed_gene_list_{run_tag}.csv'), index=False, header=['gene'])
pd.DataFrame(
    [(k, v[0]) for k, v in guide_assignments_dict.items()],
    columns=['cell_id', 'gene']
).to_csv(os.path.join(args.ckpt_dir, f'guide_assignments_{run_tag}.csv'), index=False)

v_stim.to_csv(os.path.join(args.ckpt_dir, f'v_stim_aligned_{run_tag}.csv'), header=['weight'])
dpd_per_cell_aligned.to_csv(
    os.path.join(args.ckpt_dir, f'dpd_per_cell_{run_tag}.csv'), header=['DPD_cell'])

with open(os.path.join(args.ckpt_dir, f'recentre_offset_{run_tag}.txt'), 'w') as f:
    f.write(str(recentre_offset))

log(f'Checkpoints saved to {args.ckpt_dir}/')
log('1_prep_stv.py complete.')

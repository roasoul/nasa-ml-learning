import torch
import numpy as np

data = torch.load('data/kepler_tce_2000.pt', weights_only=False)

fluxes = data['fluxes']        # (N, 200) primary flux
labels = data['labels']        # (N,) 0=FP, 1=confirmed
names  = data['names']         # list of KOI IDs

N = len(labels)
confirmed_mask = labels == 1
fp_mask        = labels == 0

# Flux is baseline-subtracted: baseline=0, dip=negative.
# depth_ppm = magnitude of the dip
depth_ppm = (-fluxes.min(dim=1).values) * 1e6

print("=" * 55)
print("kepler_tce_2000.pt depth distribution")
print("=" * 55)
print(f"Total TCEs   : {N}")
print(f"Confirmed    : {confirmed_mask.sum().item()}")
print(f"FP           : {fp_mask.sum().item()}")
print(f"Depth range  : {depth_ppm.min():.1f} - {depth_ppm.max():.1f} ppm")
print()

BINS = [0, 300, 1000, 3000, 10000, 1e9]
BIN_LABELS = ['<300', '300-1k', '1k-3k', '3k-10k', '10k+']

print(f"{'Bin':<10} {'Range (ppm)':<14} {'Confirmed':>10} {'FP':>8} {'Total':>8} {'Flag'}")
print("-" * 58)

bin_data = {}
for i, (lo, hi) in enumerate(zip(BINS, BINS[1:])):
    in_bin = (depth_ppm >= lo) & (depth_ppm < hi)
    conf   = (in_bin & confirmed_mask).sum().item()
    fp     = (in_bin & fp_mask).sum().item()
    total  = conf + fp
    flag   = " *** SPARSE" if conf < 40 else ""
    print(f"{i+1:<10} {BIN_LABELS[i]:<14} {conf:>10} {fp:>8} {total:>8}{flag}")
    bin_data[BIN_LABELS[i]] = {'confirmed': conf, 'fp': fp,
                                'conf_idx': (in_bin & confirmed_mask).nonzero(as_tuple=True)[0],
                                'fp_idx':   (in_bin & fp_mask).nonzero(as_tuple=True)[0]}

print()
print("Balanced sample size per bin (min of confirmed, fp):")
total_balanced = 0
for label, d in bin_data.items():
    n = min(d['confirmed'], d['fp'])
    total_balanced += n * 2
    print(f"  {label:<8}: {n} conf + {n} FP = {n*2} TCEs")
print(f"  => Stratified total: {total_balanced} TCEs")

print()
print("Depth percentiles (all TCEs):")
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f"  p{p:02d}: {np.percentile(depth_ppm.numpy(), p):.1f} ppm")

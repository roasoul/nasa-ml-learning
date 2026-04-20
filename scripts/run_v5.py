"""Run the V5 training pipeline end-to-end (same code path as notebook 05).

Pretrains on synthetic 3-class data (planet / EB / non-transit), fine-tunes
on 70 real Kepler TCEs, evaluates on the 16-TCE test set. Prints metrics
for comparison with V4.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.data.synthetic import make_synthetic_batch
from src.models.taylor_cnn import TaylorCNN


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load V5 dataset ----
    dataset = torch.load("data/kepler_tce.pt", weights_only=False)
    assert "fluxes_secondary" in dataset, "Dataset is V4 format — rebuild first."

    phases = dataset["phases"]
    fluxes_primary = dataset["fluxes"]
    fluxes_secondary = dataset["fluxes_secondary"]
    labels = dataset["labels"]
    n_conf = int(labels.sum())
    n_fp = int((1 - labels).sum())
    print(f"Dataset: {len(labels)} TCEs ({n_conf} conf, {n_fp} FP)")

    # ---- Same seed-42 stratified split as notebook 04 ----
    torch.manual_seed(42)
    conf_idx = (labels == 1).nonzero(as_tuple=True)[0]
    fp_idx = (labels == 0).nonzero(as_tuple=True)[0]
    conf_perm = conf_idx[torch.randperm(len(conf_idx))]
    fp_perm = fp_idx[torch.randperm(len(fp_idx))]

    def split_indices(indices: torch.Tensor, train_frac: float = 0.7, val_frac: float = 0.15):
        n = len(indices)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        return (
            indices[:n_train],
            indices[n_train : n_train + n_val],
            indices[n_train + n_val :],
        )

    conf_train, conf_val, conf_test = split_indices(conf_perm)
    fp_train, fp_val, fp_test = split_indices(fp_perm)
    train_idx = torch.cat([conf_train, fp_train])
    val_idx = torch.cat([conf_val, fp_val])
    test_idx = torch.cat([conf_test, fp_test])

    def make_split(idx):
        return (
            phases[idx].to(device),
            fluxes_primary[idx].to(device),
            fluxes_secondary[idx].to(device),
            labels[idx].to(device),
        )

    train_ph, train_p, train_s, train_l = make_split(train_idx)
    val_ph, val_p, val_s, val_l = make_split(val_idx)
    test_ph, test_p, test_s, test_l = make_split(test_idx)

    print(
        f"Train: {len(train_l)} ({int(train_l.sum())} conf, {int((1-train_l).sum())} FP)"
    )
    print(f"Val:   {len(val_l)} ({int(val_l.sum())} conf, {int((1-val_l).sum())} FP)")
    print(f"Test:  {len(test_l)} ({int(test_l.sum())} conf, {int((1-test_l).sum())} FP)")

    # ---- Build model (from scratch — synthetic pretraining hurt; see notebook 05) ----
    torch.manual_seed(7)
    model = TaylorCNN(init_amplitude=0.01).to(device)
    criterion = nn.BCELoss()
    print(f"\nModel params: {sum(p.numel() for p in model.parameters())}")

    # ---- Train on real Kepler ----
    optimizer = torch.optim.Adam(
        [
            {"params": model.taylor_gate.parameters(), "lr": 1e-4},
            {"params": model.cnn.parameters(), "lr": 1e-3},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ]
    )

    n_epochs = 200
    batch_size = 16
    n_train = len(train_l)
    patience = 25
    best_val_loss = float("inf")
    best_state = None
    wait = 0
    best_A = None

    print(f"\nFine-tuning on {n_train} real TCEs")
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        epoch_correct = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(train_ph[idx], train_p[idx], train_s[idx]).squeeze(1)
            loss = criterion(pred, train_l[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
            epoch_correct += ((pred > 0.5).float() == train_l[idx]).sum().item()
        train_loss = epoch_loss / n_train
        train_acc = epoch_correct / n_train

        model.eval()
        with torch.no_grad():
            vp = model(val_ph, val_p, val_s).squeeze(1)
            val_loss = criterion(vp, val_l).item()
            val_acc = ((vp > 0.5).float() == val_l).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_A = model.taylor_gate.A.item()
            wait = 0
        else:
            wait += 1

        if epoch == 0 or (epoch + 1) % 10 == 0 or wait == patience:
            marker = " *" if wait == 0 else ""
            print(
                f"  epoch {epoch+1:>3}  "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={train_acc:.1%} val_acc={val_acc:.1%} "
                f"A={model.taylor_gate.A.item():.5f}{marker}"
            )
        if wait >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)

    # ---- Test set evaluation ----
    model.eval()
    with torch.no_grad():
        test_probs = model(test_ph, test_p, test_s).squeeze(1).cpu()
    test_preds = (test_probs > 0.5).float()
    test_labels_cpu = test_l.cpu()

    tp = int(((test_preds == 1) & (test_labels_cpu == 1)).sum())
    tn = int(((test_preds == 0) & (test_labels_cpu == 0)).sum())
    fp_ct = int(((test_preds == 1) & (test_labels_cpu == 0)).sum())
    fn = int(((test_preds == 0) & (test_labels_cpu == 1)).sum())
    accuracy = (tp + tn) / (tp + tn + fp_ct + fn)
    precision = tp / (tp + fp_ct) if (tp + fp_ct) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n=== V5 Test Set Results (16 TCEs, same as V4) ===")
    print(f"Accuracy:  {accuracy:.1%}  (V4: 75.0%)")
    print(f"Precision: {precision:.1%}  (V4: 66.7%)")
    print(f"Recall:    {recall:.1%}     (V4: 100.0%)")
    print(f"F1:        {f1:.3f}        (V4: 0.800)")
    print(f"Best val loss: {best_val_loss:.4f}  |  A (best): {best_A:.5f}")
    print()
    print("Confusion matrix:")
    print("                      Pred Planet  Pred Not")
    print(f"  True Planet             {tp:>2}           {fn:>2}")
    print(f"  True Not-Planet         {fp_ct:>2}           {tn:>2}")
    print()
    print(f"Success criterion: precision > 80% AND recall = 100%")
    print(f"  precision > 80%:  {'PASS' if precision > 0.80 else 'FAIL'} ({precision:.1%})")
    print(f"  recall  = 100%:  {'PASS' if recall >= 1.00 else 'FAIL'} ({recall:.1%})")

    # Per-TCE breakdown so you can see which FPs survived and why
    names = dataset["names"]
    test_idx_list = test_idx.tolist()
    print("\nPer-TCE predictions (s_depth = secondary-view minimum in bins 95–105):")
    print(f"  {'KOI':<12} {'true':<8} {'prob':<6} {'pred':<8} {'s_depth':<10}")
    for pos, orig in enumerate(test_idx_list):
        truth = "PLANET" if test_labels_cpu[pos] == 1 else "FP"
        pred_s = "PLANET" if test_preds[pos] == 1 else "NOT"
        prob = test_probs[pos].item()
        s_depth = fluxes_secondary[orig, 95:105].min().item()
        mark = "" if test_preds[pos] == test_labels_cpu[pos] else "  <-- wrong"
        print(
            f"  {names[orig]:<12} {truth:<8} {prob:.2f}   "
            f"{pred_s:<8} {s_depth:+.4f}{mark}"
        )

    save_path = "src/models/taylor_cnn_v5.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "A": model.taylor_gate.A.item(),
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        },
        save_path,
    )
    print(f"\nSaved V5 model to {save_path}")


if __name__ == "__main__":
    main()

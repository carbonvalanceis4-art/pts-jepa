"""Run the first controlled MUMDMC benchmark locally."""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.splits import add_research_columns, group_split
from data.manifest import resolve_image_path
from models.baselines import make_resnet18, make_vit_tiny
from models.ijepa_tiny import IJEPATiny, random_target_mask


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class MUMD(Dataset):
    def __init__(self, frame, manifest_path, labels, train=False):
        self.frame = frame.reset_index(drop=True)
        self.manifest_path = manifest_path
        self.label_to_id = {x: i for i, x in enumerate(labels)}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.train = train

    def __len__(self): return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = Image.open(resolve_image_path(self.manifest_path, row.path)).convert("RGB")
        return self.transform(image), self.label_to_id[row.label]


def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(device, non_blocking=True)).argmax(1).cpu().numpy()
            ys.extend(y.numpy()); ps.extend(p)
    return {
        "accuracy": float(accuracy_score(ys, ps)),
        "macro_f1": float(f1_score(ys, ps, average="macro")),
        "confusion_matrix": confusion_matrix(ys, ps).tolist(),
    }


def train_supervised(model, name, train_loader, val_loader, test_loader, device, epochs):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    loss_fn = nn.CrossEntropyLoss()
    best_state, best_f1 = None, -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(x), y).backward()
            optimizer.step()
        val = evaluate(model, val_loader, device)
        print(f"{name} epoch={epoch} val_acc={val['accuracy']:.4f} val_macro_f1={val['macro_f1']:.4f}")
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]; best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    result = evaluate(model, test_loader, device)
    result["model"] = name
    return result


def train_ijepa(train_loader, test_loader, labels, device, pretrain_epochs, probe_epochs):
    model = IJEPATiny(image_size=224, patch_size=16, dim=192, depth=4, heads=4).to(device)
    optimizer = torch.optim.AdamW(model.context_encoder.parameters(), lr=1e-3, weight_decay=0.05)
    for epoch in range(1, pretrain_epochs + 1):
        model.train(); losses=[]
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            mask = random_target_mask(x.size(0), model.context_encoder.num_patches, device=x.device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(x, mask)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); model.update_target()
            losses.append(loss.item())
        print(f"ijepa epoch={epoch} loss={np.mean(losses):.5f}")

    class Frozen(nn.Module):
        def __init__(self, encoder, n):
            super().__init__(); self.encoder=encoder; self.head=nn.Linear(encoder.dim,n)
            for p in encoder.parameters(): p.requires_grad=False
        def forward(self, x): return self.head(self.encoder(x))

    probe = Frozen(model.context_encoder, len(labels)).to(device)
    probe_opt = torch.optim.AdamW(probe.head.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(probe_epochs):
        probe.train()
        for x, y in train_loader:
            x,y=x.to(device),y.to(device); probe_opt.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(probe(x), y).backward(); probe_opt.step()
    result = evaluate(probe, test_loader, device); result["model"]="ijepa_tiny"
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("data/manifests/mumdmc2025.csv"))
    p.add_argument("--output", type=Path, default=Path("outputs/e1"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--ijepa-pretrain-epochs", type=int, default=10)
    p.add_argument("--probe-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args=p.parse_args(); seed_all(args.seed)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df=add_research_columns(pd.read_csv(args.manifest))
    labels=sorted(df.label.unique().tolist())
    train,val,test=group_split(df)
    loaders=[DataLoader(MUMD(f,args.manifest,labels,train=(f is train)), batch_size=args.batch_size, shuffle=(f is train), num_workers=2, pin_memory=torch.cuda.is_available()) for f in (train,val,test)]
    train_loader,val_loader,test_loader=loaders
    results=[]; started=time.time()
    results.append(train_supervised(make_resnet18(len(labels), pretrained=True),"resnet18",train_loader,val_loader,test_loader,device,args.epochs))
    results.append(train_supervised(make_vit_tiny(len(labels), pretrained=True),"vit_tiny",train_loader,val_loader,test_loader,device,args.epochs))
    results.append(train_ijepa(train_loader,test_loader,labels,device,args.ijepa_pretrain_epochs,args.probe_epochs))

    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k:v for k,v in r.items() if k!="confusion_matrix"} for r in results]).to_csv(args.output/"metrics.csv",index=False)
    (args.output/"results.json").write_text(json.dumps(results,indent=2))
    (args.output/"split_sizes.json").write_text(json.dumps({"train":len(train),"val":len(val),"test":len(test),"groups":int(df.research_group.nunique()),"elapsed_seconds":time.time()-started},indent=2))
    print(pd.DataFrame([{k:v for k,v in r.items() if k!="confusion_matrix"} for r in results]).to_string(index=False))

if __name__ == "__main__": main()

# audit_indices.py
import argparse, json, hashlib, numpy as np, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_path", required=True, help="FrankaDataset の config.path (torch.saveされた.p)")
    ap.add_argument("--sample_length", type=int, required=True)
    ap.add_argument("--stack_states", type=int, required=True)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--drop_last", action="store_true", help="eval時にdrop_lastしてるなら付ける")
    ap.add_argument("--dump_prefix", default="audit_out", help="出力ファイルの接頭辞")
    args = ap.parse_args()

    print("[AUDIT] loading splits:", args.splits_path)
    splits = torch.load(args.splits_path, map_location="cpu", weights_only=False)

    # 画像は読み込まない（FrankaDatasetのロジックを最小限で再現）
    episode_lengths = [len(d["observations"]) for d in splits]

    flattened_indices = []
    for ep_idx, d in enumerate(splits):
        usable = episode_lengths[ep_idx] - args.sample_length - (args.stack_states - 1)
        for t in range(max(0, usable)):
            flattened_indices.append((ep_idx, t))

    # 指紋（順序＋内容が同じなら一致する）
    fi_json = json.dumps(flattened_indices).encode()
    fi_sha = hashlib.sha256(fi_json).hexdigest()
    print("[AUDIT] FLATTENED_LEN:", len(flattened_indices))
    print("[AUDIT] FLATTENED_SHA:", fi_sha)

    # “shuffle=False の先頭バッチ”に相当するインデックス一覧
    bs = args.batch_size
    eff_len = len(flattened_indices)
    if args.drop_last:
        eff_len = (eff_len // bs) * bs
    first_batch = list(range(0, min(bs, eff_len)))
    first_batch_pairs = [flattened_indices[i] for i in first_batch]

    print("[AUDIT] FIRST_BATCH_SIZE:", len(first_batch_pairs))
    print("[AUDIT] FIRST_BATCH_IDX (first 16):", first_batch[:16])
    print("[AUDIT] FIRST_BATCH_PAIRS (first 8):", first_batch_pairs[:8])

    # 比較用に保存
    np.save(f"{args.dump_prefix}_first_batch_indices.npy", np.array(first_batch, dtype=int))
    with open(f"{args.dump_prefix}_flattened_sha.txt", "w") as f:
        f.write(fi_sha+"\n")
    with open(f"{args.dump_prefix}_first_batch_pairs.json", "w") as f:
        json.dump(first_batch_pairs, f)

if __name__ == "__main__":
    main()

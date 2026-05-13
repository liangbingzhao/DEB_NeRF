"""Generate a markdown report from per-walnut comparison_table*.csv.

Splits each walnut's CSV by view (25v / 50v / 100v), Borda-sorts by PSNR + SSIM
combined rank, prints to stdout AND saves to a single markdown file.

Usage:
  python experiments/report_eval_split.py            # default = comparison_table.csv
  python experiments/report_eval_split.py --clip     # use comparison_table_clip.csv
  python experiments/report_eval_split.py --both     # generate both side-by-side
"""
import argparse
import csv
import os.path as osp
import re
import sys

REPO = "/ibex/project/c2272/liangbing/cs300_project"
WALNUTS = ["walnut_1", "walnut_2", "walnut_3"]
VIEWS = ["50v", "25v", "100v"]


def parse_view(method):
    if re.search(r"\b25\b", method): return "25v"
    if re.search(r"\b50\b", method): return "50v"
    if re.search(r"\b100\b", method): return "100v"
    return "?"


def load_csv(walnut, suffix):
    p = osp.join(REPO, f"experiments/{walnut}/eval_M9_total/comparison_table{suffix}.csv")
    if not osp.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    valid = []
    for r in rows:
        try:
            r["psnr"] = float(r["psnr_vs_total"])
            r["ssim"] = float(r["ssim_vs_total"])
            r["sipsnr"] = float(r["si_psnr_vs_total"])
            r["view"] = parse_view(r["method"])
            valid.append(r)
        except Exception:
            pass
    return valid


def borda_sort(rows):
    if not rows:
        return []
    psnr_rank = {id(r): i for i, r in enumerate(sorted(rows, key=lambda r: -r["psnr"]))}
    ssim_rank = {id(r): i for i, r in enumerate(sorted(rows, key=lambda r: -r["ssim"]))}
    for r in rows:
        r["borda"] = psnr_rank[id(r)] + ssim_rank[id(r)]
    rows.sort(key=lambda r: (r["borda"], -r["ssim"]))
    return rows


def render_table(rows):
    if not rows:
        return "_(no data)_\n"
    out = []
    out.append("| # | section | method | PSNR | SSIM | si-PSNR | Borda |")
    out.append("|--:|---|---|--:|--:|--:|--:|")
    for i, r in enumerate(rows):
        marker = " ⭐" if i == 0 and r["section"] == "method" else ""
        out.append(f"| {i+1} | {r['section']} | {r['method']}{marker} "
                   f"| {r['psnr']:.2f} | {r['ssim']:.4f} | {r['sipsnr']:.2f} | {r['borda']} |")
    return "\n".join(out) + "\n"


def render_walnut(walnut, suffix, label):
    """Returns markdown section for one walnut."""
    rows = load_csv(walnut, suffix)
    if rows is None:
        return f"## {walnut.upper()} {label}\n\n_(file not found: comparison_table{suffix}.csv)_\n\n"
    section = [f"## {walnut.upper()} {label}\n"]
    for view in VIEWS:
        sub = [r for r in rows if r["view"] == view]
        if not sub: continue
        sub = borda_sort(sub)
        section.append(f"### {view}  ({len(sub)} entries)\n")
        section.append(render_table(sub))
    return "\n".join(section) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="store_true", help="use comparison_table_clip.csv")
    ap.add_argument("--both", action="store_true",
                    help="generate side-by-side report (original + clip)")
    ap.add_argument("--out", default=None, help="output markdown path")
    args = ap.parse_args()

    if args.both:
        suffixes = [("", " — original minmax"), ("_clip", " — percentile-clip p99.5")]
    elif args.clip:
        suffixes = [("_clip", " — percentile-clip p99.5")]
    else:
        suffixes = [("", " — original minmax")]

    body = []
    body.append("# M9 vs-Total eval — Borda-sorted by PSNR+SSIM (lower=better)\n")
    body.append(f"_Generated: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}_\n")

    for suffix, label in suffixes:
        body.append(f"\n# Variant: {label.strip(' —')}\n")
        for w in WALNUTS:
            body.append(render_walnut(w, suffix, label))

    md = "\n".join(body)
    print(md)

    if args.out is None:
        suf = "_clip" if args.clip else ("_both" if args.both else "")
        args.out = osp.join(REPO, f"experiments/eval_split_report{suf}.md")
    with open(args.out, "w") as f:
        f.write(md)
    print(f"\n>>> saved to: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

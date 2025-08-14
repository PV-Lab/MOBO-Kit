import argparse, json, os
from .optimize import run_once
from .utils import load_config, set_seeds, select_device
from .design import build_input_spec_list, build_space

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--out", default=None, help="Override output directory")
    args = p.parse_args()

    cfg = load_config(args.config)
    specs = build_input_spec_list(cfg["inputs"])
    space = build_space(specs)
    set_seeds(cfg["seed"])
    device = select_device(cfg.get("device", "cuda"))

    out_dir = args.out or cfg["files"]["save_dir"]
    os.makedirs(out_dir, exist_ok=True)

    summary = run_once(cfg, space=space, device=device, out_dir=out_dir)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

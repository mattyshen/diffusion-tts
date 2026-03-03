"""
This script allows you to generate images using either the Stable Diffusion (SD) or EDM backend,
with a choice of scorer and sampling method

Usage examples:
    python main.py --backend sd --scorer brightness --method naive --prompt "A beautiful landscape"
    python main.py --backend edm --scorer imagenet --method zero_order

Arguments:
    --backend   : 'sd' or 'edm' (required)
    --scorer    : 'brightness', 'compressibility', 'clip', or 'imagenet' (required)
    --method    : Sampling method (available: 'naive', 'rejection', 'beam', 'mcts', 'zero_order', 'eps_greedy', 'adap_eps_greedy') (default: 'naive')
    --prompt    : Prompt for SD (default: 'A beautiful landscape')
    --output    : Output filename
    --N, --lambda_, --eps, --K, --B, --S : sampling parameters (see code for defaults)
    --seed      : Random seed (default: 0)
    --device    : Device (default: 'cuda')
"""

import os
import sys
import argparse
import importlib.util
import torch
import numpy as np
import json
from pathlib import Path
from PIL import Image
import importlib
from search_logging import SearchDataLogger, make_run_path, utc_now_iso

# =========================
# EDM Import Helper
# =========================
def import_edm():
    """Dynamically import EDM modules and scorers."""
    edm_dir = Path(__file__).parent / 'edm'
    sys.path.insert(0, str(edm_dir))
    dnnlib = importlib.import_module('dnnlib')
    dnnlib_util = importlib.import_module('dnnlib.util')
    from scorers import BrightnessScorer, CompressibilityScorer, ImageNetScorer
    return dnnlib, dnnlib_util, BrightnessScorer, CompressibilityScorer, ImageNetScorer

# =========================
# SD Import Helper
# =========================
def import_sd():
    """Dynamically import SD pipeline and scorers."""
    sd_dir = Path(__file__).parent / 'sd'
    diffusers_path = sd_dir / 'diffusers' / 'src' / 'diffusers' / '__init__.py'
    spec = importlib.util.spec_from_file_location('diffusers', str(diffusers_path.resolve()))
    sys.modules['diffusers'] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules['diffusers'])
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    sys.path.insert(0, str(sd_dir))
    from scorers import BrightnessScorer, CompressibilityScorer, CLIPScorer
    return StableDiffusionPipeline, DDIMScheduler, BrightnessScorer, CompressibilityScorer, CLIPScorer

# =========================
# Scorer Factory
# =========================
def get_scorer(backend, scorer_name, BrightnessScorer, CompressibilityScorer, CLIPScorer=None, ImageNetScorer=None):
    """Return the appropriate scorer instance for the backend and scorer name."""
    if scorer_name == 'brightness':
        return BrightnessScorer(dtype=torch.float32)
    elif scorer_name == 'compressibility':
        return CompressibilityScorer(dtype=torch.float32)
    elif scorer_name == 'clip' and backend == 'sd':
        return CLIPScorer(dtype=torch.float32)
    elif scorer_name == 'imagenet' and backend == 'edm':
        return ImageNetScorer(dtype=torch.float32)
    else:
        raise ValueError(f"Unknown or invalid scorer '{scorer_name}' for backend '{backend}'")

# =========================
# Main Logic
# =========================
def main():
    def _read_prompt_file(path):
        prompts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text and not text.startswith("#"):
                    prompts.append(text)
        return prompts

    def _safe_name(text, max_len=48):
        keep = []
        for ch in text.lower():
            if ch.isalnum():
                keep.append(ch)
            elif ch in {" ", "-", "_"}:
                keep.append("_")
        cleaned = "".join(keep).strip("_")
        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")
        return (cleaned[:max_len] if cleaned else "prompt")

    def _load_adap_profile(path):
        if path is None:
            raise ValueError("Adaptive eps-greedy requires --adap_profile_path")
        profile_path = Path(path)
        if not profile_path.exists():
            raise ValueError(f"Adaptive profile not found: {profile_path}")
        suffix = profile_path.suffix.lower()
        if suffix == ".npy":
            values = np.asarray(np.load(profile_path), dtype=float).reshape(-1)
        elif suffix == ".json":
            with open(profile_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                values = np.asarray(payload, dtype=float).reshape(-1)
            elif isinstance(payload, dict):
                if "values" in payload:
                    values = np.asarray(payload["values"], dtype=float).reshape(-1)
                elif "z" in payload:
                    values = np.asarray(payload["z"], dtype=float).reshape(-1)
                else:
                    numeric_keys = [k for k in payload.keys() if str(k).isdigit()]
                    if len(numeric_keys) == len(payload):
                        values = np.asarray([payload[k] for k in sorted(payload.keys(), key=lambda x: int(x))], dtype=float).reshape(-1)
                    else:
                        raise ValueError(f"Unsupported JSON adaptive profile format: {profile_path}")
            else:
                raise ValueError(f"Unsupported JSON adaptive profile format: {profile_path}")
        else:
            values = np.asarray(np.loadtxt(profile_path, dtype=float), dtype=float).reshape(-1)
        if values.size == 0:
            raise ValueError(f"Adaptive profile is empty: {profile_path}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Adaptive profile contains non-finite values: {profile_path}")
        return [float(x) for x in values.tolist()]

    # -----------
    # CLI Arguments
    # -----------
    parser = argparse.ArgumentParser(
        description='Unified Diffusion Image Generator (EDM/SD)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--backend', type=str, choices=['edm', 'sd'], required=True, help='Backend: edm or sd')
    parser.add_argument('--scorer', type=str, choices=['brightness', 'compressibility', 'clip', 'imagenet'], required=True, help='Scorer name')
    parser.add_argument('--method', type=str, default='naive', help='Sampling method (naive, rejection, beam, mcts, zero_order, eps_greedy, adap_eps_greedy)')
    parser.add_argument('--prompt', type=str, default='YOUR PROMPT HERE', help='Prompt for SD')
    parser.add_argument('--prompt_file', type=str, default=None, help='Path to text file with one SD prompt per line')
    parser.add_argument('--class_indices', type=str, default=None, help='Comma-separated ImageNet class indices for EDM (e.g., "0,1,2,3,4")')
    parser.add_argument('--output', type=str, default=None, help='Output filename (default: auto)')
    # Master params (with SD defaults)
    parser.add_argument('--N', type=int, default=4, help='Master param N')
    parser.add_argument('--lambda_', type=float, default=0.15, help='Master param lambda')
    parser.add_argument('--eps', type=float, default=0.4, help='Master param eps')
    parser.add_argument('--K', type=int, default=20, help='Master param K')
    parser.add_argument('--B', type=int, default=2, help='Master param B')
    parser.add_argument('--S', type=int, default=8, help='Master param S')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--log_search_data', action='store_true', help='Enable candidate-level JSONL logging')
    parser.add_argument('--search_log_dir', type=str, default='logs/search_stats', help='Directory for search logs')
    parser.add_argument('--search_log_prefix', type=str, default='search_data', help='Prefix for search log file name')
    parser.add_argument('--adap_profile_path', type=str, default=None, help='Path to adaptive profile vector (txt/json/npy), one value per denoising step')
    parser.add_argument('--adap_alpha', type=float, default=0.0, help='Adaptive threshold alpha: use full search when profile[t] >= alpha')
    parser.add_argument('--adap_Kg', type=int, default=4, help='Adaptive global-only candidate count when profile[t] < alpha')
    args = parser.parse_args()
    args.method = args.method.lower().replace("-", "_")
    valid_methods = {'naive', 'rejection', 'beam', 'mcts', 'zero_order', 'eps_greedy', 'adap_eps_greedy'}
    if args.method not in valid_methods:
        raise ValueError(f"Unknown method: {args.method}")
    if args.adap_Kg < 1:
        raise ValueError("--adap_Kg must be >= 1")

    # -----------
    # Validation
    # -----------
    if args.backend == 'sd' and args.scorer == 'imagenet':
        raise ValueError('imagenet scorer is only available for edm backend')
    if args.backend == 'edm' and args.scorer == 'clip':
        raise ValueError('clip scorer is only available for sd backend')

    # -----------
    # SD Backend
    # -----------
    if args.backend == 'sd':
        StableDiffusionPipeline, DDIMScheduler, BrightnessScorer, CompressibilityScorer, CLIPScorer = import_sd()
        scorer = get_scorer('sd', args.scorer, BrightnessScorer, CompressibilityScorer, CLIPScorer=CLIPScorer)

        model_id = "runwayml/stable-diffusion-v1-5"
        local_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        local_pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=local_scheduler,
            torch_dtype=torch.float16,
        ).to(args.device)

        method = args.method
        num_inference_steps = 50
        MASTER_PARAMS = {
            'N': args.N,
            'lambda': args.lambda_,
            'eps': args.eps,
            'K': args.K,
            'B': args.B,
            'S': args.S,
        }
        if method == "adap_eps_greedy":
            adap_profile = _load_adap_profile(args.adap_profile_path)
            if len(adap_profile) != num_inference_steps:
                raise ValueError(
                    f"Adaptive profile length mismatch for SD: got {len(adap_profile)}, expected {num_inference_steps}"
                )
            MASTER_PARAMS["adap_z_profile"] = adap_profile
            MASTER_PARAMS["adap_alpha"] = float(args.adap_alpha)
            MASTER_PARAMS["adap_Kg"] = int(args.adap_Kg)

        prompts = [args.prompt]
        if args.prompt_file is not None:
            prompts = _read_prompt_file(args.prompt_file)
            if len(prompts) == 0:
                raise ValueError(f"No prompts found in prompt file: {args.prompt_file}")
        print(f"[SD] Running {len(prompts)} prompt(s)")

        for pidx, prompt_text in enumerate(prompts):
            search_logger = None
            if args.log_search_data:
                run_path = make_run_path(args.search_log_dir, f"{args.search_log_prefix}_sd_{args.method}_{args.scorer}_p{pidx:03d}")
                search_logger = SearchDataLogger(
                    run_path,
                    {
                        "created_utc": utc_now_iso(),
                        "backend": "sd",
                        "method": args.method,
                        "scorer": args.scorer,
                        "prompt": prompt_text,
                        "prompt_index": pidx,
                        "seed": args.seed,
                        "device": args.device,
                        "params": MASTER_PARAMS,
                        "model_id": model_id,
                        "prompt_file": args.prompt_file,
                    },
                )
                print(f"[SD] Search logging enabled: {run_path}")

            best_result, best_score = None, float('-inf')
            for _ in range(MASTER_PARAMS['N'] if method == "rejection" else 1):
                result, score = local_pipe(
                    prompt=prompt_text,
                    num_inference_steps=num_inference_steps,
                    score_function=scorer,
                    method=method,
                    params=MASTER_PARAMS,
                    search_data_logger=search_logger,
                )
                if score > best_score:
                    best_result, best_score = result, score

            if args.output and len(prompts) == 1:
                outname = args.output
            elif args.output and len(prompts) > 1:
                base = Path(args.output)
                outname = str(base.with_name(f"{base.stem}_p{pidx:03d}_{_safe_name(prompt_text)}{base.suffix or '.png'}"))
            else:
                outname = f"sd_{method}_{args.scorer}_p{pidx:03d}_{_safe_name(prompt_text)}.png"

            best_result.images[0].save(outname)
            print(f"\n[SD] Saved: {outname}\nPrompt[{pidx}]: {prompt_text}\nBest score: {best_score}\n")
            if search_logger is not None:
                search_logger.log({"event_type": "run_end", "ts_utc": utc_now_iso(), "output": outname, "best_score": float(best_score), "prompt_index": pidx})
                search_logger.close()

    # -----------
    # EDM Backend
    # -----------
    elif args.backend == 'edm':
        dnnlib, dnnlib_util, BrightnessScorer, CompressibilityScorer, ImageNetScorer = import_edm()
        scorer = get_scorer('edm', args.scorer, BrightnessScorer, CompressibilityScorer, ImageNetScorer=ImageNetScorer)

        # EDM defaults
        model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
        network_pkl = f'{model_root}/edm-imagenet-64x64-cond-adm.pkl'
        class_indices = None
        if args.class_indices is not None:
            class_indices = [int(x.strip()) for x in args.class_indices.split(",") if x.strip() != ""]
            if len(class_indices) == 0:
                raise ValueError("--class_indices was provided but no valid indices were parsed")
            for idx in class_indices:
                if idx < 0 or idx > 999:
                    raise ValueError(f"EDM class index out of range [0,999]: {idx}")
        num_images = len(class_indices) if class_indices is not None else 1
        # Keep batch size consistent with requested class indices.
        # EDM generation expects batch_size == gridw * gridh.
        if class_indices is None:
            gridw = gridh = int(np.ceil(np.sqrt(num_images)))
        else:
            gridw, gridh = num_images, 1
        latents = torch.randn([num_images, 3, 64, 64])
        if class_indices is None:
            class_labels = torch.eye(1000)[torch.randint(1000, size=[num_images])]
        else:
            class_labels = torch.eye(1000)[torch.tensor(class_indices, dtype=torch.long)]
        device = torch.device(args.device)
        num_steps = 18

        # EDM method mapping
        from edm.main import SamplingMethod, generate_image_grid
        method_map = {
            'naive': SamplingMethod.NAIVE,
            'rejection': SamplingMethod.REJECTION_SAMPLING,
            'beam': SamplingMethod.BEAM_SEARCH,
            'mcts': SamplingMethod.MCTS,
            'zero_order': SamplingMethod.ZERO_ORDER,
            'eps_greedy': SamplingMethod.EPS_GREEDY,
            'adap_eps_greedy': SamplingMethod.ADAP_EPS_GREEDY,
        }
        if args.method not in method_map:
            raise ValueError(f"Unknown method: {args.method}")
        sampling_method = method_map[args.method]
        sampling_params = {'scorer': scorer}

        # Add master params if relevant for method
        if args.method in ['rejection', 'zero_order', 'eps_greedy', 'adap_eps_greedy', 'beam', 'mcts']:
            if args.N is not None:
                sampling_params['N'] = args.N
            if args.K is not None:
                sampling_params['K'] = args.K
            if args.lambda_ is not None:
                sampling_params['lambda_param'] = args.lambda_
            if args.eps is not None:
                sampling_params['eps'] = args.eps
            if args.B is not None:
                sampling_params['B'] = args.B
            if args.S is not None:
                sampling_params['S'] = args.S
            if args.method == 'adap_eps_greedy':
                adap_profile = _load_adap_profile(args.adap_profile_path)
                if len(adap_profile) != num_steps:
                    raise ValueError(
                        f"Adaptive profile length mismatch for EDM: got {len(adap_profile)}, expected {num_steps}"
                    )
                sampling_params['adap_z'] = adap_profile
                sampling_params['adap_alpha'] = float(args.adap_alpha)
                sampling_params['adap_Kg'] = int(args.adap_Kg)

        search_logger = None
        if args.log_search_data:
            run_path = make_run_path(args.search_log_dir, f"{args.search_log_prefix}_edm_{args.method}_{args.scorer}")
            class_indices_for_log = torch.argmax(class_labels, dim=1).cpu().tolist()
            # `sampling_params` contains a scorer object, which is not JSON-serializable.
            # Keep only scalar/search hyperparameters in run metadata.
            params_for_log = {k: v for k, v in sampling_params.items() if k != "scorer"}
            search_logger = SearchDataLogger(
                run_path,
                {
                    "created_utc": utc_now_iso(),
                    "backend": "edm",
                    "method": args.method,
                    "scorer": args.scorer,
                    "class_indices": class_indices_for_log,
                    "seed": args.seed,
                    "device": args.device,
                    "params": params_for_log,
                    "network_pkl": network_pkl,
                    "num_steps": num_steps,
                },
            )
            print(f"[EDM] Search logging enabled: {run_path}")

        outname = args.output or f"edm_{args.method}_{args.scorer}.png"
        edm_result = generate_image_grid(
            network_pkl,
            outname,
            latents,
            class_labels,
            seed=args.seed,
            gridw=gridw,
            gridh=gridh,
            device=device,
            num_steps=num_steps,
            S_churn=40,
            S_min=0.05,
            S_max=50,
            S_noise=1.003,
            sampling_method=sampling_method,
            sampling_params=sampling_params,
            search_data_logger=search_logger,
        )
        print(f"\n[EDM] Saved: {outname}\n")
        if search_logger is not None:
            run_end_event = {"event_type": "run_end", "ts_utc": utc_now_iso(), "output": outname}
            if isinstance(edm_result, dict):
                if edm_result.get("best_score") is not None:
                    run_end_event["best_score"] = float(edm_result["best_score"])
                if edm_result.get("avg_score") is not None:
                    run_end_event["avg_score"] = float(edm_result["avg_score"])
                if edm_result.get("min_score") is not None:
                    run_end_event["min_score"] = float(edm_result["min_score"])
                if edm_result.get("scores") is not None:
                    run_end_event["per_sample_scores"] = [float(x) for x in edm_result["scores"]]
            search_logger.log(run_end_event)
            search_logger.close()

if __name__ == '__main__':
    main()

"""Explicit-local frozen Alpha-CLIP factory. Importing this file imports no torch.

Only the registered L/14@336 GRIT20M full-tune 4xe model is supported. The
original CLIP checkpoint and alpha visual checkpoint are constituents of it.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re
import sys
import time
from collections.abc import Mapping

import numpy as np

from .artifacts import array_digest, digest, file_digest, readonly
from .encoding import EncodedRegion, validate_encoding

MODEL_VERSION = "object-scope-alpha-clip-L14-336-grit20m-4xe-v1"
MODEL_KEYS = {"schema", "variant", "training", "base_checkpoint", "alpha_checkpoint",
              "package_root", "source_files", "tokenizer_file", "precision"}


def _file_ref(ref):
    if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}:
        raise ValueError("strict local file reference required")
    if not isinstance(ref["path"], str) or not isinstance(ref["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", ref["sha256"]):
        raise ValueError("invalid file reference identity")
    path = Path(ref["path"])
    if not path.is_absolute() or not path.is_file() or file_digest(path) != ref["sha256"]:
        raise ValueError("missing or changed local asset: " + str(path))
    return path.resolve()


def validate_model_spec(spec):
    """CPU filesystem audit, before loading framework/model or acquiring GPU."""
    if not isinstance(spec, Mapping) or set(spec) != MODEL_KEYS:
        raise ValueError("strict Alpha-CLIP specification required")
    if (spec["schema"] != MODEL_VERSION or spec["variant"] != "ViT-L/14@336px"
            or spec["training"] != "GRIT20M-full-tune-4xe" or spec["precision"] not in {"float16", "float32"}):
        raise ValueError("model variant/training/precision is not the frozen experiment")
    root = Path(spec["package_root"])
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("absolute local alpha_clip package directory required")
    root = root.resolve()
    required = {p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".py", ".so", ".pyd"}}
    minimum = {root / p for p in ("__init__.py", "alpha_clip.py", "model.py", "simple_tokenizer.py")}
    if not minimum <= required or not isinstance(spec["source_files"], list):
        raise ValueError("complete official model package sources required")
    listed = [_file_ref(ref) for ref in spec["source_files"]]
    if len(set(listed)) != len(listed) or set(listed) != required:
        raise ValueError("source manifest must exactly cover all package .py/.so/.pyd files")
    tokenizer = _file_ref(spec["tokenizer_file"])
    if tokenizer != (root / "bpe_simple_vocab_16e6.txt.gz").resolve():
        raise ValueError("bind the exact tokenizer read by the official package")
    base = _file_ref(spec["base_checkpoint"])
    alpha = _file_ref(spec["alpha_checkpoint"])
    if base == alpha:
        raise ValueError("CLIP base and alpha visual checkpoint must be separately identified")
    return {"identity_sha256": digest(spec), "package_root": str(root), "base_checkpoint": str(base),
            "alpha_checkpoint": str(alpha), "tokenizer_file": str(tokenizer),
            "source_files": {str(path): file_digest(path) for path in sorted(required)},
            "precision": spec["precision"]}


def _require_lease():
    if not os.environ.get("SAGA_OBJECT_LEASE_TOKEN"):
        raise RuntimeError("real model loading/forward requires the study's reviewed execution lease")


def _import_package(root, identity):
    name = "_saga_object_scope_alpha_" + identity[:20]
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != root / "__init__.py":
            raise RuntimeError("frozen package module identity collision")
        return module, name
    module_spec = importlib.util.spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("cannot load the explicit local package")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    try:
        module_spec.loader.exec_module(module)
    except BaseException:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                sys.modules.pop(key, None)
        raise
    return module, name


def _adapt_clip_base(state, torch):
    """Exact official structural key mapping, followed by STRICT state loading."""
    result = {}
    for key, value in state.items():
        if key in {"input_resolution", "context_length", "vocab_size"}:
            continue
        if key.startswith("visual."):
            key = key.replace("in_proj_weight", "in_proj.weight").replace("in_proj_bias", "in_proj.bias")
        if key in result:
            raise ValueError("duplicate checkpoint key after official attention conversion")
        result[key] = value
    if "visual.conv1_alpha.weight" in result:
        raise ValueError("base checkpoint unexpectedly already has an alpha encoder")
    result["visual.conv1_alpha.weight"] = torch.zeros_like(result["visual.conv1.weight"][:, :1])
    return result


def load_local_alpha_clip(spec, *, device="cuda"):
    """Native model factory, never calls official load(name)/download helpers.

    This is an executable factory but has not itself been exercised on weights
    by the CPU synthetic tests. The owning launcher must gate review and budget.
    """
    assets = validate_model_spec(spec)
    _require_lease()
    if device != "cpu" and not re.fullmatch(r"cuda(?::[0-9]+)?", device):
        raise ValueError("only explicit CPU or CUDA devices are supported")
    import torch
    start = time.perf_counter()
    package, alias = _import_package(Path(assets["package_root"]), assets["identity_sha256"])
    model_module = sys.modules[alias + ".model"]
    try:
        base_archive = torch.jit.load(assets["base_checkpoint"], map_location="cpu")
        base_state = base_archive.state_dict()
    except RuntimeError:
        base_state = torch.load(assets["base_checkpoint"], map_location="cpu", weights_only=True)
    if not isinstance(base_state, Mapping):
        raise ValueError("base checkpoint must expose an unwrapped state dictionary")
    # Registered L/14@336 dimensions. Strict load rejects every other variant.
    model = model_module.CLIP(embed_dim=768, image_resolution=336, vision_layers=24,
        vision_width=1024, vision_patch_size=14, context_length=77, vocab_size=49408,
        transformer_width=768, transformer_heads=12, transformer_layers=12, lora_adapt=False, rank=16)
    model.load_state_dict(_adapt_clip_base(base_state, torch), strict=True)
    alpha_state = torch.load(assets["alpha_checkpoint"], map_location="cpu", weights_only=True)
    if not isinstance(alpha_state, Mapping):
        raise ValueError("alpha checkpoint must be an unwrapped full visual state dictionary")
    model.visual.load_state_dict(alpha_state, strict=True)
    if spec["precision"] == "float16":
        model_module.convert_weights(model)
    else:
        model.float()
    model.eval().requires_grad_(False).to(device)
    loaded = {}
    for key, module in list(sys.modules.items()):
        if (key == alias or key.startswith(alias + ".")) and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if str(path) not in assets["source_files"] or file_digest(path) != assets["source_files"][str(path)]:
                raise ValueError("actually loaded model source differs from frozen manifest")
            loaded[key.replace(alias, "alpha_clip", 1)] = {"path": str(path), "sha256": file_digest(path)}
    runtime_identity = {"model_version": MODEL_VERSION, "assets": assets, "loaded_modules": loaded,
        "factory_source_sha256": file_digest(__file__), "python": sys.version,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "torch_binary_sha256": file_digest(torch._C.__file__), "device": device,
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "precision": spec["precision"], "strict_base_load": True, "strict_alpha_load": True,
        "implicit_downloads": False}
    return LocalAlphaClipRuntime(model, package.tokenize, runtime_identity, torch=torch, device=device,
                                 load_seconds=time.perf_counter() - start)


class LocalAlphaClipRuntime:
    def __init__(self, model, tokenize, runtime_identity, *, torch, device, load_seconds=0.):
        self.model = model
        self.tokenize = tokenize
        self.runtime_identity = runtime_identity
        self.load_seconds = float(load_seconds)
        self.torch = torch
        self.device = device
        self._text_cache = {}

    def encode_pair(self, detail: EncodedRegion, context: EncodedRegion, prompts):
        _require_lease()
        validate_encoding(detail)
        validate_encoding(context)
        if not detail.valid or not context.valid:
            raise ValueError("unknown encoding must not call the model")
        if len(prompts) != 32 or len(set(prompts)) != 32:
            raise ValueError("all 32 frozen class prompts are required")
        torch = self.torch
        tokens_cpu = self.tokenize(list(prompts), context_length=77, truncate=False).cpu()
        tokens_numpy = readonly(tokens_cpu.numpy())
        if tokens_numpy.shape != (32, 77):
            raise ValueError("actual tokenizer shape differs from frozen CLIP")
        text_key = digest({"tokens": array_digest(tokens_numpy), "model": self.runtime_identity["assets"]["identity_sha256"]})
        images = torch.from_numpy(np.stack([detail.rgb_tensor, context.rgb_tensor])).to(self.device, dtype=self.model.dtype)
        alphas = torch.from_numpy(np.stack([detail.alpha_tensor, context.alpha_tensor])).to(self.device, dtype=self.model.dtype)
        # Capture actual dtype/device-converted tensors, BEFORE the forward call.
        actual_images = readonly(images.detach().cpu().numpy())
        actual_alphas = readonly(alphas.detach().cpu().numpy())
        tokens = tokens_cpu.to(self.device)
        started = time.perf_counter()
        text_cached = text_key in self._text_cache
        with torch.inference_mode():
            if not text_cached:
                self._text_cache[text_key] = readonly(self.model.encode_text(tokens).detach().cpu().numpy())
            text_features = self._text_cache[text_key]
            images_out = readonly(self.model.visual(images, alphas).detach().cpu().numpy())
        if images_out.ndim != 2 or images_out.shape[0] != 2:
            raise ValueError("actual visual output is not a pair of feature vectors")
        record = {"identity": self.runtime_identity, "model_load_seconds": self.load_seconds, "prompts": list(prompts),
            "exact_forward_token_ids": tokens_numpy, "token_ids_sha256": array_digest(tokens_numpy),
            "forward_rgb": actual_images, "forward_alpha": actual_alphas,
            "forward_rgb_sha256": array_digest(actual_images), "forward_alpha_sha256": array_digest(actual_alphas),
            "detail_encoding_sha256": detail.trace["encoding_sha256"],
            "context_encoding_sha256": context.trace["encoding_sha256"],
            "text_cache_hit": text_cached, "text_cache_key": text_key,
            "calls": {"visual": 1, "visual_images": 2, "text": 0 if text_cached else 1},
            "seconds": time.perf_counter() - started}
        record["runtime_record_sha256"] = digest(record)
        return {"detail_features": readonly(images_out[0]), "context_features": readonly(images_out[1]),
                "text_features": text_features, "token_ids": tokens_numpy, "runtime_record": record}

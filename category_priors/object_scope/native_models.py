"""Explicit local SAM1 factory and opt-in acquisition of the one new model.

Import/validation has no torch, network or GPU side effects. Acquisition is a
separate explicit command; inference factories never fetch missing assets.
"""
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

from . import EXPERIMENT_VERSION
from .artifacts import digest, file_digest, write_once

SAM_SCHEMA = "scope-v2-local-existing-sam1-vit-h-v1"
SAM_KEYS = {"schema", "model_type", "checkpoint", "package_root", "source_files", "precision"}
ALPHA_REPOSITORY = "https://github.com/SunzeY/AlphaCLIP.git"
# Verified against official model-zoo.md, GRIT20M L14@336 full-tune 4xe row.
ALPHA_URL = "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_336_grit_20m_4xe.pth"
# Official alpha_clip/alpha_clip.py _MODELS[ViT-L/14@336px].
BASE_SHA256 = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
BASE_URL = "https://openaipublic.azureedge.net/clip/models/" + BASE_SHA256 + "/ViT-L-14-336px.pt"
OFFICIAL_REFERENCES = (
    "https://github.com/SunzeY/AlphaCLIP/blob/main/model-zoo.md",
    "https://github.com/SunzeY/AlphaCLIP/blob/main/alpha_clip/alpha_clip.py",
    "https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/build_sam.py",
)


def _file(ref):
    if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}:
        raise ValueError("exact local file/hash reference required")
    if not isinstance(ref["path"], str) or not isinstance(ref["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", ref["sha256"]):
        raise ValueError("invalid local asset reference")
    path = Path(ref["path"])
    if not path.is_absolute() or not path.is_file() or file_digest(path) != ref["sha256"]:
        raise ValueError("missing or changed local model asset")
    return path.resolve()


def validate_sam_spec(spec):
    """CPU-only actual-file validation of the already frozen SAM1 vit_h."""
    if not isinstance(spec, Mapping) or set(spec) != SAM_KEYS:
        raise ValueError("strict local SAM1 specification required")
    if spec["schema"] != SAM_SCHEMA or spec["model_type"] != "vit_h" or spec["precision"] != "float32":
        raise ValueError("only existing SAM1 vit_h float32 is registered")
    root = Path(spec["package_root"])
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("absolute segment_anything package directory required")
    root = root.resolve()
    actual = {p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".py", ".so", ".pyd"}}
    minimum = {root/p for p in ("__init__.py", "build_sam.py", "predictor.py", "modeling/sam.py", "utils/transforms.py")}
    if not minimum <= actual or not isinstance(spec["source_files"], list):
        raise ValueError("complete SAM1 official package sources required")
    listed = [_file(ref) for ref in spec["source_files"]]
    if len(set(listed)) != len(listed) or set(listed) != actual:
        raise ValueError("SAM1 source manifest must exactly cover package .py/.so/.pyd")
    checkpoint = _file(spec["checkpoint"])
    return {"spec_sha256": digest(spec), "package_root": str(root), "checkpoint": str(checkpoint),
        "checkpoint_sha256": spec["checkpoint"]["sha256"],
        "sources": {str(p): file_digest(p) for p in sorted(actual)}, "model_type": "vit_h", "precision": "float32"}


def _lease():
    if not os.environ.get("SAGA_OBJECT_LEASE_TOKEN"):
        raise RuntimeError("native SAM1 loading/forward requires the reviewed GPU lease")


def _import_sam(assets):
    root = Path(assets["package_root"])
    name = "_scope_sam1_" + assets["spec_sha256"][:20]
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, root/"__init__.py", submodule_search_locations=[str(root)])
        if spec is None or spec.loader is None: raise RuntimeError("cannot import local SAM1 package")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try: spec.loader.exec_module(module)
        except BaseException:
            for key in tuple(sys.modules):
                if key == name or key.startswith(name+"."): sys.modules.pop(key, None)
            raise
    loaded = {}
    for key, row in tuple(sys.modules.items()):
        if (key == name or key.startswith(name+".")) and getattr(row, "__file__", None):
            path = Path(row.__file__).resolve()
            if str(path) not in assets["sources"] or file_digest(path) != assets["sources"][str(path)]:
                raise ValueError("actually loaded SAM1 source differs from registered local package")
            loaded[key.replace(name, "segment_anything", 1)] = {"path": str(path), "sha256": file_digest(path)}
    if Path(module.__file__).resolve() != root/"__init__.py":
        raise ValueError("SAM1 alias resolves outside frozen package")
    return module, loaded


@dataclass(frozen=True)
class LoadedSAM:
    predictor: object
    adapter: object
    identity: dict
    load_seconds: float


def load_sam(spec, *, device="cuda"):
    assets = validate_sam_spec(spec)
    _lease()
    import torch
    import torchvision
    import PIL
    import numpy
    from .observations import NativeSAM1Predictor
    start = time.perf_counter()
    module, _ = _import_sam(assets)
    # Explicit local state, strict keys, no registry filename resolution/download.
    model = module.sam_model_registry["vit_h"](checkpoint=None)
    state = torch.load(assets["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.float().eval().requires_grad_(False).to(device)
    predictor = module.SamPredictor(model)
    _, loaded = _import_sam(assets)
    dependencies = {}
    for name, dependency in (("torch",torch),("torchvision",torchvision),("PIL",PIL),("numpy",numpy)):
        path = Path(dependency.__file__).resolve()
        dependencies[name] = {"version": dependency.__version__, "path":str(path),"sha256":file_digest(path)}
    binary_paths = {Path(torch._C.__file__).resolve()}
    binary_paths.update(Path(p).resolve() for p in torch.ops.loaded_libraries if Path(p).is_file())
    identity = {"schema":SAM_SCHEMA,"assets":assets,"loaded_modules":loaded,"dependencies":dependencies,
        "loaded_extension_binaries":{str(p):file_digest(p) for p in sorted(binary_paths)},
        "factory_source_sha256":file_digest(__file__),"python":sys.version,"device":str(device),
        "torch_cuda":torch.version.cuda,"strict_state_load":True,"implicit_downloads":False,
        "cuda_matmul_allow_tf32":bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32":bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms":bool(torch.are_deterministic_algorithms_enabled())}
    # The measurement store consumes .predictor; callers of capture_bank use
    # .adapter. Both reference the same explicitly loaded, immutable model.
    return LoadedSAM(predictor, NativeSAM1Predictor(predictor), identity, time.perf_counter()-start)


def acquire_alpha_assets(destination, *, source_commit):
    """Explicit administrative acquisition, never invoked by a model factory.

    Caller must choose a reviewed exact official commit. Existing/failed trees
    are retained; recovery chooses a new destination, never replaces assets.
    """
    if not isinstance(source_commit,str) or not re.fullmatch("[0-9a-f]{40}",source_commit):
        raise ValueError("reviewed exact official AlphaCLIP commit required")
    destination = Path(destination).resolve()
    if EXPERIMENT_VERSION not in destination.parts or destination.exists():
        raise ValueError("fresh independent scope-v2 acquisition directory required")
    destination.mkdir(parents=True)
    source = destination/"AlphaCLIP"
    source.mkdir()
    for argv in (["git","init",str(source)], ["git","-C",str(source),"remote","add","origin",ALPHA_REPOSITORY],
                 ["git","-C",str(source),"fetch","--depth","1","origin",source_commit],
                 ["git","-C",str(source),"checkout","--detach","FETCH_HEAD"]):
        subprocess.run(argv,check=True,capture_output=True,text=True)
    actual_commit = subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    if actual_commit != source_commit: raise ValueError("official source checkout identity mismatch")
    # Revalidate the allowlisted URLs against the actual pinned official source,
    # so a mutable web page cannot silently redefine the requested model.
    if ALPHA_URL not in (source/"model-zoo.md").read_text(encoding="utf8") or BASE_URL not in (source/"alpha_clip/alpha_clip.py").read_text(encoding="utf8"):
        raise ValueError("pinned official source does not register the exact two model components")
    import urllib.request
    files = {}
    for key,url,name,expected in (("base",BASE_URL,"ViT-L-14-336px.pt",BASE_SHA256),
                                  ("alpha",ALPHA_URL,"clip_l14_336_grit_20m_4xe.pth",None)):
        path = destination/name
        partial = destination/(name+".partial")
        with urllib.request.urlopen(url,timeout=60) as response, partial.open("xb") as output:
            metadata = {"requested_url":url,"resolved_url":response.geturl(),"headers":dict(response.headers)}
            while True:
                chunk=response.read(1024*1024)
                if not chunk:break
                output.write(chunk)
        actual = file_digest(partial)
        if expected is not None and actual != expected:raise ValueError("official CLIP base checksum mismatch")
        with partial.open("rb") as stream:magic=stream.read(4)
        if not (magic.startswith(b"PK\x03\x04") or magic.startswith(b"\x80")):
            raise ValueError("download is not a checkpoint container; preserve failed partial")
        partial.rename(path)
        files[key]={"path":str(path),"sha256":actual,"official_expected_sha256":expected,
                    "checksum_status":"verified_official" if expected else "recorded_local_no_published_checksum",**metadata}
        write_once(destination/(key+"-download.json"),files[key])
    result={"source_commit":source_commit,"repository":ALPHA_REPOSITORY,"official_references":OFFICIAL_REFERENCES,
        "source_files":{str(p.relative_to(source)):file_digest(p) for p in sorted(source.rglob("*"))
                        if p.is_file() and ".git" not in p.relative_to(source).parts},"files":files,
        "model_calls":0,"training_calls":0,"additional_model_count":1}
    write_once(destination/"acquisition.json",result)
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--acquire-alpha-assets",required=True)
    parser.add_argument("--source-commit",required=True)
    args=parser.parse_args()
    print(json.dumps(acquire_alpha_assets(args.acquire_alpha_assets,source_commit=args.source_commit)))


if __name__ == "__main__":main()

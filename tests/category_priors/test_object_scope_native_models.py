from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

from category_priors.object_scope import native_models as nm
from category_priors.object_scope.artifacts import file_digest


def spec(tmp_path):
    root=tmp_path/"segment_anything";root.mkdir()
    for name in ("__init__.py","build_sam.py","predictor.py","modeling/sam.py","utils/transforms.py"):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("# CPU synthetic source\n")
    (root/"__init__.py").write_text('''
class Model:
    mask_threshold=0.
    def load_state_dict(self,state,strict):
        if strict is not True or state != {'expected': 1}: raise ValueError('strict synthetic keys rejected')
    def float(self): return self
    def eval(self): return self
    def requires_grad_(self,value):
        assert value is False
        return self
    def to(self,device):
        assert device == 'cpu'
        return self
def construct(checkpoint):
    assert checkpoint is None
    return Model()
sam_model_registry={'vit_h':construct}
class SamPredictor:
    def __init__(self,model): self.model=model
''')
    checkpoint=tmp_path/"existing_sam.pth";checkpoint.write_bytes(b"CPU synthetic state, no actual model")
    ref=lambda p:{"path":str(p.resolve()),"sha256":file_digest(p)}
    return dict(schema=nm.SAM_SCHEMA,model_type="vit_h",checkpoint=ref(checkpoint),package_root=str(root.resolve()),
                source_files=[ref(p) for p in sorted(root.rglob("*.py"))],precision="float32")


def fake_torch(monkeypatch, state=None):
    calls=[]
    def load(path, **kwargs):
        calls.append((path,kwargs))
        assert kwargs=={"map_location":"cpu","weights_only":True}
        return {"expected":1} if state is None else state
    torch=SimpleNamespace(__file__=__file__,__version__="CPU-synthetic",load=load,
        _C=SimpleNamespace(__file__=__file__),ops=SimpleNamespace(loaded_libraries=set()),
        version=SimpleNamespace(cuda=None),backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=False)),are_deterministic_algorithms_enabled=lambda:True)
    monkeypatch.setitem(sys.modules,"torch",torch)
    monkeypatch.setitem(sys.modules,"torchvision",SimpleNamespace(__file__=__file__,__version__="CPU-synthetic"))
    return calls


def test_validate_sam_is_cpu_files_only(tmp_path,monkeypatch):
    config=spec(tmp_path)
    monkeypatch.setitem(sys.modules,"torch",None)
    assert nm.validate_sam_spec(config)["model_type"]=="vit_h"


@pytest.mark.parametrize("kind",["extra","missing","duplicate","checkpoint"])
def test_actual_source_and_weight_closure_cannot_be_shortened(tmp_path,kind):
    config=spec(tmp_path)
    if kind=="extra": (Path(config["package_root"])/"additional.py").write_text("# new code")
    elif kind=="missing": config["source_files"].pop()
    elif kind=="duplicate": config["source_files"].append(config["source_files"][0])
    else: Path(config["checkpoint"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):nm.validate_sam_spec(config)


def test_loading_requires_lease_before_framework_import(tmp_path,monkeypatch):
    config=spec(tmp_path);monkeypatch.delenv("SAGA_OBJECT_LEASE_TOKEN",raising=False)
    monkeypatch.setitem(sys.modules,"torch",None)
    with pytest.raises(RuntimeError,match="lease"):nm.load_sam(config,device="cpu")


def test_synthetic_local_factory_has_strict_state_and_real_import_identity(tmp_path,monkeypatch):
    config=spec(tmp_path);calls=fake_torch(monkeypatch)
    monkeypatch.setenv("SAGA_OBJECT_LEASE_TOKEN","CPU-synthetic-only")
    result=nm.load_sam(config,device="cpu")
    assert result.adapter.predictor is result.predictor
    assert result.identity["strict_state_load"] and result.identity["implicit_downloads"] is False
    assert len(calls)==1
    assert result.identity["loaded_modules"]["segment_anything"]["path"]==str(Path(config["package_root"])/"__init__.py")


def test_synthetic_checkpoint_mismatch_does_not_become_partial_load(tmp_path,monkeypatch):
    config=spec(tmp_path);fake_torch(monkeypatch,{"unexpected":1})
    monkeypatch.setenv("SAGA_OBJECT_LEASE_TOKEN","CPU-synthetic-only")
    with pytest.raises(ValueError,match="strict synthetic keys"):nm.load_sam(config,device="cpu")


def test_acquisition_never_runs_for_unpinned_or_existing_destination(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError("must not launch acquisition")
    monkeypatch.setattr(nm.subprocess,"run",forbidden)
    with pytest.raises(ValueError,match="commit"):nm.acquire_alpha_assets(tmp_path/"new",source_commit="main")
    with pytest.raises(ValueError,match="fresh"):nm.acquire_alpha_assets(tmp_path,source_commit="a"*40)


def test_only_registered_two_components_are_in_download_constants():
    assert nm.ALPHA_URL.endswith("clip_l14_336_grit_20m_4xe.pth")
    assert nm.BASE_SHA256 in nm.BASE_URL
    assert nm.ALPHA_REPOSITORY=="https://github.com/SunzeY/AlphaCLIP.git"


def test_sam_reload_clock_changes_measurement_not_cache_identity(tmp_path,monkeypatch):
    config=spec(tmp_path);fake_torch(monkeypatch)
    monkeypatch.setenv("SAGA_OBJECT_LEASE_TOKEN","CPU-synthetic-only")
    clock=iter((10.,11.,100.,107.))
    monkeypatch.setattr(nm.time,"perf_counter",lambda:next(clock))
    first=nm.load_sam(config,device="cpu"); second=nm.load_sam(config,device="cpu")
    assert first.load_seconds==1. and second.load_seconds==7.
    assert first.identity==second.identity
    assert "load_seconds" not in first.identity

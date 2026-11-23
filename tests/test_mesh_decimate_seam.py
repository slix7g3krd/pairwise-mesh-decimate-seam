import pytest
import graphics.mesh_decimate_seam as mod

CASES = [
    ("seam_lock", (1)),
    ("edge_cost", (1)),
    ("tangent_rebuild", (1)),
    ("lod_morph", (1)),
    ("async_version", (1)),
    ("collapse_order", (1)),
    ("bounds_update", (1)),
    ("validation", (1)),
    ("audit", (1)),
    ("fallback", (1)),
    ("uv_guard", (1)),
    ("metrics", (1)),
    ("normal_seam", (1)),
    ("import_dedupe", (1)),
    ("determinism", (1)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None

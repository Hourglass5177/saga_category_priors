"""Independent final structural invariants; no model or renderer involved."""
import pytest
from category_priors.object_scope.geometry import PANELS, CameraView, ViewPanel


def test_committable_mid_panel_always_has_new_construction_view():
    # A dependent R1 receiver can enter the retraction set only after Hmid.
    # The suspected missing request_map entry is unreachable with frozen roles.
    for n, roles in PANELS.items():
        cams = tuple(CameraView(str(i), (0., 0., 1.), (i * .2, 0., 0.), 1.) for i in range(n))
        panel = ViewPanel(tuple(zip(roles, cams)), (), ())
        if panel.camera_for('Hmid') is not None:
            assert panel.camera_for('I3') is not None or panel.camera_for('I4') is not None
        if panel.camera_for('I3') is None and panel.camera_for('I4') is None:
            assert panel.camera_for('Hmid') is None
    cams = tuple(CameraView(str(i), (0., 0., 1.), (i * .2, 0., 0.), 1.) for i in range(4))
    with pytest.raises(ValueError, match='unregistered role table'):
        ViewPanel(tuple(zip(('I1', 'I2', 'Hmid', 'Hfinal'), cams)), (), ())

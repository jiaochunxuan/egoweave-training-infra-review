import unittest,tempfile,json
from pathlib import Path
import numpy as np
from human_data import validate_view_pair
from targets import rescale_reference_norms,SCALE

class ControlTests(unittest.TestCase):
    def test_renamed_local_annotations_are_not_a_global_control(self):
        with tempfile.TemporaryDirectory() as d:
            g=Path(d)/'global.json';l=Path(d)/'local.json'
            g.write_text('{}');l.write_text('{}')
            with self.assertRaises(ValueError):validate_view_pair(g,l)
    def test_different_roi_is_rejected(self):
        control={k:'same' for k in ['recordings_sha256','roi_sha256','calibration_sha256','estimator_sha256','fusion_recipe_sha256','smoothing_recipe_sha256']}
        with tempfile.TemporaryDirectory() as d:
            g=Path(d)/'global.json';l=Path(d)/'local.json'
            g.write_text(json.dumps(dict(reconstruction_control=dict(control,view='global_axis'))))
            l.write_text(json.dumps(dict(reconstruction_control=dict(control,view='hand_centered',roi_sha256='other'))))
            with self.assertRaises(ValueError):validate_view_pair(g,l)
    def test_no_scale_uses_one_fixed_conversion_for_all_local_channels(self):
        stats={k:dict(mean=list(np.ones(n)),std=list(np.ones(n))) for k,n in [('state',156),('actions',153)]}
        new=rescale_reference_norms(stats,'no_scale')
        for k in stats:
            np.testing.assert_allclose(new[k]['std'][27:153],SCALE)
            np.testing.assert_allclose(new[k]['mean'][:27],1)
            np.testing.assert_allclose(stats[k]['std'],1)

    def test_actual_pair_provenance_requires_same_policy_rgb_and_frames(self):
        c={k:'a'*64 for k in ['recordings_sha256','roi_sha256','calibration_sha256','estimator_sha256','fusion_recipe_sha256','smoothing_recipe_sha256']}
        common=dict(coordinate_contract={'units':'metres','wrist':'head'},image_manifest_sha256='b'*64,image_config_sha256='c'*64)
        with tempfile.TemporaryDirectory() as d:
            g=Path(d)/'global.json';l=Path(d)/'local.json'
            g.write_text(json.dumps(dict(common,reconstruction_control=dict(c,view='global_axis'))))
            l.write_text(json.dumps(dict(common,reconstruction_control=dict(c,view='hand_centered'))))
            validate_view_pair(g,l)
            changed=dict(common,image_config_sha256='d'*64)
            l.write_text(json.dumps(dict(changed,reconstruction_control=dict(c,view='hand_centered'))))
            with self.assertRaises(ValueError):validate_view_pair(g,l)

if __name__=='__main__':unittest.main()

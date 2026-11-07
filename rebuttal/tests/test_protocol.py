import unittest
import numpy as np
import protocol

class ProtocolTests(unittest.TestCase):
    def test_exact_native_budgets(self):
        for suite,steps,label,batch in [('dexjoco',10001,10000,32),('dmg_a',5001,5000,128),('dmg_b',5001,5000,128),('robotwin',5000,4999,64)]:
            s=protocol.native_spec(suite,'egoweave',123)
            self.assertEqual((s['updates'],s['eval_checkpoint'],s['batch']),(steps,label,batch))
            self.assertEqual(s['seed'],123);self.assertEqual(s['sampler_seed'],123)
    def test_seeds_change_sampling_and_run_identity(self):
        a=protocol.native_spec('dmg_a','egoweave',123);b=protocol.native_spec('dmg_a','egoweave',124)
        self.assertNotEqual(a['run'],b['run'])
        self.assertNotEqual(list(protocol.EpochSampler(200,16,123,3)),list(protocol.EpochSampler(200,16,124,3)))
    def test_epoch_sampler_pads_only_at_end(self):
        a=list(protocol.EpochSampler(10,4,123,3));self.assertEqual(len(a),12)
        self.assertEqual(len(set(a[:10])),10);self.assertEqual(a[10:],a[:2])
    def test_two_epoch_real_budget(self):
        self.assertEqual(protocol.real_steps(110848),6928)
        with self.assertRaises(ValueError):protocol.real_steps(10)
    def test_reject_unsafe_output_name(self):
        with self.assertRaises(ValueError):protocol.native_spec('dmg_a','../legacy',123)

if __name__=='__main__':unittest.main()

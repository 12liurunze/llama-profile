import json
import tempfile
import unittest
from pathlib import Path
from profiler.kernel_audit import audit_captures


class KernelAuditTests(unittest.TestCase):
    def test_rank_windows_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rank in range(8):
                path=Path(tmp)/f'rank_{rank:03d}'/'kernel_profile/status.json'
                path.parent.mkdir(parents=True)
                rows=[dict(step=10+i,phase='puredecode',bs=512,batch_fingerprint='same') for i in range(3)]
                path.write_text(json.dumps(dict(status='complete',expected_bs=512,captured_steps=3,rows=rows)))
            self.assertTrue(audit_captures(tmp,512)['audit_pass'])
            value=json.loads(path.read_text())
            for row in value['rows']:
                row['step']+=1
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError,'different kernel windows'):
                audit_captures(tmp,512)
            value['status']='graph_nodes_unverified'
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError,'Incomplete'):
                audit_captures(tmp,512)


if __name__=='__main__':
    unittest.main()

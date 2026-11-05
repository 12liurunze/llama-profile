"""Audit that separate TP diagnostics captured the same actual full-batch steps."""
import json
from pathlib import Path


def audit_captures(trace_root, expected_bs, world=8):
    expected_signature = None
    for rank in range(world):
        path = Path(trace_root)/f'rank_{rank:03d}'/'kernel_profile/status.json'
        status = json.loads(path.read_text(encoding='utf-8'))
        rows = status.get('rows',[])
        if (status.get('status') != 'complete' or status.get('expected_bs') != expected_bs
                or status.get('captured_steps') != 3 or len(rows) != 3):
            raise RuntimeError(f'Incomplete kernel capture on rank {rank}')
        signature = [(r['step'],r['phase'],r['bs'],r['batch_fingerprint']) for r in rows]
        if any(phase != 'puredecode' or bs != expected_bs or not fingerprint
               for _,phase,bs,fingerprint in signature):
            raise RuntimeError('Kernel capture is not actual full-batch decode')
        if any(signature[i][0] != signature[0][0]+i or signature[i][3] != signature[0][3]
               for i in range(3)):
            raise RuntimeError('Kernel capture is not a consecutive identical-request window')
        if expected_signature is not None and signature != expected_signature:
            raise RuntimeError('TP ranks captured different kernel windows')
        expected_signature = signature
    return dict(audit_pass=True,world_size=world,steps=expected_signature)

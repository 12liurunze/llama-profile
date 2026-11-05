import os
import signal
import subprocess
import time


def snapshot(gpus, own_group=None):
    def smi(scope, fields):
        txt = subprocess.check_output(['nvidia-smi', '--query-'+scope+'='+fields,
            '--format=csv,noheader,nounits'], text=True, timeout=20)
        return [[v.strip() for v in line.split(',')] for line in txt.splitlines() if line.strip()]
    all_devices = {r[0]:dict(index=r[0], uuid=r[1], name=r[2], memory_mib=int(r[3]))
                   for r in smi('gpu','index,uuid,name,memory.total')}
    if any(g not in all_devices for g in gpus):
        raise RuntimeError('Requested GPU indices not present')
    devices = [all_devices[g] for g in gpus]
    uuids = {r['uuid'] for r in devices}
    active = [dict(uuid=r[0],pid=int(r[1]),used_mib=r[2])
              for r in smi('compute-apps','gpu_uuid,pid,used_memory') if r[0] in uuids]
    foreign = []
    for row in active:
        try:
            if own_group is None or os.getpgid(row['pid']) != own_group:
                foreign.append(row)
        except ProcessLookupError:
            pass
    return dict(devices=devices, processes=active, outsiders=foreign, time=time.time())


def stop(child):
    if child is None:
        return
    # Popen was started in a new session. Kill only its saved process group,
    # including surviving TP workers after a failed coordinator.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.2)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

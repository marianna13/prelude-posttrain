"""Print per-step values of chosen keys from offline .wandb files (history records only)."""
import json, sys
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore

keys = sys.argv[1].split(",")
for path in sys.argv[2:]:
    ds = datastore.DataStore(); ds.open_for_scan(path)
    rows = []
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = wandb_internal_pb2.Record(); rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue
        item = {i.key or "/".join(i.nested_key): json.loads(i.value_json) for i in rec.history.item}
        if any(k in item for k in keys):
            rows.append({k: item.get(k) for k in ["_step", *keys]})
    print(path)
    for r in rows[:6]:
        print("  ", r)

"""Read saved W&B protobuf history without uploading a run."""
import argparse
import json
from pathlib import Path
from wandb.sdk.internal.datastore import DataStore
from wandb.proto.wandb_internal_pb2 import Record


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    store=DataStore();store.open_for_scan(str(a.run))
    records=[]
    while True:
        try:
            raw=store.scan_data()
        except (AssertionError,EOFError):
            break
        if raw is None:
            break
        record=Record();record.ParseFromString(raw)
        if record.HasField('history'):
            values={('/'.join(item.nested_key) if item.nested_key else item.key):json.loads(item.value_json)
                    for item in record.history.item}
            records.append(values)
    a.out.write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records[-3:],indent=2))


if __name__=='__main__':
    main()

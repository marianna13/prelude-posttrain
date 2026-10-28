"""Create isolated config variants over immutable weights for a causal RoPE check."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    gate=a.root/'gates_20261018'
    initialized=a.root/'base_hf'
    smoke=a.root/'runs/prelude-longcontext-smoke-16k/hf/step-1'
    def variant(name,source,theta):
        dest=gate/'diagnostic_models'/name;dest.mkdir(parents=True)
        for src in source.iterdir():
            if src.is_file() and src.name!='config.json':(dest/src.name).symlink_to(src)
        cfg=json.loads((source/'config.json').read_text())
        cfg['rope_theta']=theta;cfg['rope_parameters']=dict(rope_type='default',rope_theta=theta)
        cfg['rope_scaling']=None
        (dest/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        return dest
    cases=[('initialized-rope100k',initialized),
           ('initialized-rope6m',variant('initialized-rope6m',initialized,6000000)),
           ('smoke16-rope6m',smoke),
           ('smoke16-rope100k',variant('smoke16-rope100k',smoke,100000))]
    record={name:dict(model=str(path),config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest())
            for name,path in cases}
    (gate/'rope-diagnostic-cases.json').write_text(json.dumps(record,indent=2)+'\n')
    (gate/'rope-diagnostic-cases.tsv').write_text(''.join(f'{name}\t{path}\n' for name,path in cases))


if __name__=='__main__':main()

"""Independent full BF16 readback of a pilot HF/native checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'sft/prelude'))
from verify_midtrain_hf import headers,added_rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--export',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    cfg=json.loads((a.export/'config.json').read_text())
    assert cfg['vocab_size']==262272 and cfg['max_position_embeddings']==65536
    assert cfg['rope_theta']==6000000 and not cfg.get('use_sliding_window',False)
    assert set(cfg['eos_token_id'])=={2,262146}
    contract=json.loads((a.base/'tokenizer_contract.json').read_text())
    for name,expected in contract['tokenizer_files_sha256'].items():
        assert hashlib.sha256((a.export/name).read_bytes()).hexdigest()==expected,name
    before,after=headers(a.base),headers(a.export)
    assert before.keys()==after.keys()
    parameters=0
    for name,(path,start,h) in after.items():
        assert h['shape']==before[name][2]['shape'] and h['dtype']=='BF16',name
        n=int(np.prod(h['shape']));parameters+=n
        data=np.memmap(path,mode='r',dtype='<u2',offset=start+h['data_offsets'][0],shape=(n,))
        for i in range(0,n,8*1024**2):
            assert not np.any((data[i:i+8*1024**2]&0x7f80)==0x7f80),name
        del data
    changes={name:np.max(np.abs(added_rows(after[name])-added_rows(before[name])),axis=1).tolist()
             for name in ('model.embed_tokens.weight','lm_head.weight')}
    assert all(any(x>0 for x in values) for values in changes.values())
    checkpoint=json.loads((a.native/'metadata.json').read_text())
    assert (a.native/'manifest.json').is_file()
    result=dict(status='passed',export=str(a.export),native=str(a.native),config=cfg,
                tensors=len(after),parameters=parameters,all_weights_finite=True,
                tokenizer_contract_matches=True,added_row_max_abs_change=changes,
                native_metadata=checkpoint)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='config'}))


if __name__=='__main__':main()

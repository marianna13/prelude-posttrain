"""Independent HF artifact readback for the MN5 9B synthetic canary."""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np

def headers(folder):
 out={}
 for p in sorted(folder.glob('*.safetensors')):
  with p.open('rb') as f:
   n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n))
  for k,v in h.items():
   if k=='__metadata__':continue
   assert k not in out,k
   out[k]=(p,8+n,v)
 return out

def added_rows(entry):
 p,start,h=entry
 assert h['dtype']=='BF16',h['dtype']
 row_size=h['shape'][1]
 with p.open('rb') as f:
  f.seek(start+h['data_offsets'][0]+262145*row_size*2)
  return (np.frombuffer(f.read(6*row_size*2),dtype='<u2').astype(np.uint32)<<16).view('<f4').reshape(6,row_size)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--step',type=int,required=True);a=parser.parse_args()
 base=a.root/'base_hf';exp=a.root/f'smoke_9b_gpu/hf/step-{a.step}'
 contract=json.loads((base/'tokenizer_contract.json').read_text());cfg=json.loads((exp/'config.json').read_text())
 assert cfg['vocab_size']==262272
 assert set(cfg['eos_token_id'])=={2,262146}
 hashes={}
 for name,expected in contract['tokenizer_files_sha256'].items():
  actual=hashlib.sha256((exp/name).read_bytes()).hexdigest();assert actual==expected,name;hashes[name]=actual
 original=headers(base);trained=headers(exp);assert set(original)==set(trained)
 parameters=0
 for k,(p,start,h) in trained.items():
  assert h['shape']==original[k][2]['shape'],k
  assert h['dtype']=='BF16',(k,h['dtype'])
  n=int(np.prod(h['shape']));parameters+=n
  data=np.memmap(p,mode='r',dtype='<u2',offset=start+h['data_offsets'][0],shape=(n,))
  for i in range(0,n,8*1024**2):assert not np.any((data[i:i+8*1024**2]&0x7f80)==0x7f80),k
  del data
 changes={}
 for k in ('model.embed_tokens.weight','lm_head.weight'):
  delta=np.max(np.abs(added_rows(trained[k])-added_rows(original[k])),axis=1)
  assert np.isfinite(delta).all() and np.any(delta>0),k
  changes[k]=delta.tolist()
 metadata=json.loads((a.root/f'smoke_9b_gpu/checkpoints/step-{a.step}/metadata.json').read_text());assert metadata['step']==a.step
 result=dict(status='passed',step=a.step,path=str(exp),vocab_size=cfg['vocab_size'],eos_token_ids=cfg['eos_token_id'],tensor_names_and_shapes_match_base=True,tensors=len(trained),parameters=parameters,all_exported_weights_finite=True,tokenizer_files_sha256=hashes,added_row_max_abs_change=changes,native_checkpoint=metadata)
 out=a.root/'target_evidence';out.mkdir(exist_ok=True);(out/f'hf_step_{a.step}.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()

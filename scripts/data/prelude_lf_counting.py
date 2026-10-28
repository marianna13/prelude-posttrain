"""Exact document token counts using verified LF boundaries of this tokenizer.

The frozen adapter only needs token lengths for documents. LF is an unmergeable
byte-fallback token in Prelude, so native encoding of bounded LF-terminated
pieces produces the same IDs as whole-document encoding. This avoids enormous
BPE heaps for repository documents. Conversations still need native offsets.
"""
import json
from pathlib import Path


def chunks(text, target=65536):
    start=0
    while start<len(text):
        if len(text)-start<=target:
            yield text[start:]
            return
        end=text.rfind('\n',start,start+target)
        if end<start:
            end=text.find('\n',start+target)
        if end<0:
            yield text[start:]
            return
        end+=1
        yield text[start:end]
        start=end


def verify_barrier(tokenizer, tokenizer_path):
    config=json.loads(Path(tokenizer_path).read_text())
    expected=dict(type='Metaspace',replacement='▁',prepend_scheme='never',split=False)
    if config['normalizer'] is not None or config['pre_tokenizer']!=expected:
        raise ValueError('LF counting requires the frozen independent Metaspace transformation')
    if config.get('truncation') or config.get('padding'):
        raise ValueError('LF counting requires no truncation or padding')
    model=config['model']
    if model['type']!='BPE' or not model['byte_fallback'] or model['dropout'] is not None:
        raise ValueError('LF counting requires deterministic byte-fallback BPE')
    if model.get('continuing_subword_prefix') or model.get('end_of_word_suffix') or model.get('ignore_merges'):
        raise ValueError('Unsupported BPE boundary behavior')
    if any('\n' in token for token in model['vocab']):
        raise ValueError('A vocabulary token spans LF')
    if any('<0x0A>' in pair for pair in model['merges']):
        raise ValueError('A merge crosses the LF fallback token')
    if any('\n' in token['content'] or token['lstrip'] or token['rstrip'] for token in config['added_tokens']):
        raise ValueError('Added-token behavior crosses LF boundaries')
    if tokenizer.encode('\n',add_special_tokens=False).ids!=[model['vocab']['<0x0A>']]:
        raise ValueError('LF does not encode as its unmerged fallback token')


class CountEncoding:
    def __init__(self, length):
        self.ids=range(length)

    @property
    def offsets(self):
        raise RuntimeError('Count-only execution cannot supply conversation offsets')


class LFCountingTokenizer:
    def __init__(self, tokenizer, tokenizer_path):
        verify_barrier(tokenizer,tokenizer_path)
        self.tokenizer=tokenizer

    def encode_batch(self, texts, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError('Special-token insertion is handled by the frozen document adapter')
        counts=[0]*len(texts)
        pending=[];owners=[];chars=0
        def flush():
            nonlocal pending,owners,chars
            if pending:
                encoded=self.tokenizer.encode_batch_fast(pending,add_special_tokens=False)
                for owner,encoding in zip(owners,encoded):counts[owner]+=len(encoding)
                pending=[];owners=[];chars=0
        for owner,text in enumerate(texts):
            for piece in chunks(text):
                pending.append(piece);owners.append(owner);chars+=len(piece)
                if len(pending)>=128 or chars>=8*1024**2:flush()
        flush()
        return [CountEncoding(n) for n in counts]

    def encode(self, text, add_special_tokens=False):
        return self.encode_batch([text],add_special_tokens=add_special_tokens)[0]

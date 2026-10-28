"""Point5 data: reuse all797 prior train inputs, add995 train and512 held-out inputs."""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
from typing import List, Tuple
import time

import numpy as np

from project.pretraining.encoder import load_definitions
from project.reducibility.features import expression_key, validate
from .legal_actions import build_policy_input, content_id, UnsupportedExpression
from .packed32_common import REPO, Ledger, digest, read_json, write_json


def original_config_generator(rng, nrng):
    """Compile only the original function; no data package/import-time writes."""
    ns={'random':rng,'np':SimpleNamespace(random=nrng,where=np.where,argmin=np.argmin,multiply=np.multiply),
        'math':math,'Tuple':Tuple,'List':List}
    load_definitions('data/generation.py',['amplitude_config_gen'],ns)
    return ns['amplitude_config_gen']


def sample_configuration(fn, nrng):
    helicities, dimension=fn(5,int(nrng.randint(2,25)))
    return 5,int(dimension),[int(h) for h in helicities]


def legacy_id(tokens):
    return hashlib.sha256(repr(tokens).encode('utf-8')).hexdigest()


def bin_name(length):
    if not 1 <= length <= 36:
        raise ValueError('Input outside frozen evaluation bins')
    return '1-12' if length<=12 else '13-24' if length<=24 else '25-36'


def distribution(rows):
    return {'count':len(rows),'points':dict(Counter(r['n_point'] for r in rows)),
            'dimensions':dict(Counter(r['dimension'] for r in rows)),
            'point_dimension':dict(Counter(f"{r['n_point']}:{r['dimension']}" for r in rows)),
            'bins':dict(Counter(bin_name(r['terms']) for r in rows)),
            'mean_terms':sum(r['terms'] for r in rows)/len(rows) if rows else None,
            'helicity_configurations':len({(r['n_point'],tuple(r['helicities_twice'])) for r in rows})}


def generate(cfg, out, *, enabled=False):
    if not enabled:
        raise RuntimeError('Point5 generation is off')
    import forsythia_hep as api
    folder=out/'generation'; folder.mkdir()
    dataset=REPO/'datasets'/cfg['dataset_id']; dataset.mkdir()
    settings=cfg['data']; started=time.monotonic()
    ledger=Ledger({'candidates':settings['candidate_cap'],'monomial_calls':settings['monomial_cap'],
                   'random_identity_calls':settings['scramble_cap']})
    excluded_sources=set(); excluded_inputs=set(); seen_keys=set(); origin_keys=set()
    for name in cfg['exclude_inputs']:
        for row in read_json(REPO/name):
            excluded_sources.add(row['source_id']); excluded_inputs.add(legacy_id(row['tokens']))
            seen_keys.add(expression_key(row['tokens'],row['n_point']))
    rng=random.Random(cfg['data_seed']); nrng=np.random.RandomState(cfg['data_seed'])
    fn=original_config_generator(rng,nrng)
    reuse=[dict(r) for r in read_json(REPO/cfg['reuse_train']) if r['n_point']==5]
    if len(reuse)!=settings['reused_five_point_train']:raise ValueError('Unexpected reuse count')
    for audit_path in cfg['origin_audits']:
        with gzip.open(REPO/audit_path,'rt',encoding='utf-8') as stream:
            for line in stream:
                audit=json.loads(line)
                if audit['accepted']:origin_keys.add(audit['source_key'])
    for index,row in enumerate(reuse):
        row.update(index=index,split='train')
        validate(row['tokens'],5,row['dimension'],row['helicities_twice'])
    accepted=list(reuse); rejected=Counter(); candidates=Counter()
    write_json(folder/'started.json',{'limits':ledger.caps,'seed':cfg['data_seed']})
    try:
        with gzip.open(folder/'attempts.jsonl.gz','xt',encoding='utf-8') as attempts:
            for split in ('train','test'):
                split_rows=list(reuse) if split=='train' else []
                while len(split_rows)<settings[split]:
                    if time.monotonic()-started>settings['wall_seconds']:
                        raise RuntimeError('Generation time limit')
                    ledger.hit('candidates'); serial=ledger.counts['candidates']
                    n,dim,hels=sample_configuration(fn,nrng)
                    candidates[f'{n}:{dim}']+=1
                    requested=rng.randint(*settings['source_terms'])
                    source=0*api.abk(1,2)
                    for _ in range(requested):
                        ledger.hit('monomial_calls')
                        mono=api.random_monomial(hels,dim)
                        validate(mono.tokenize(),n,dim,hels)
                        source+=rng.choice((-3,-2,-1,1,2,3))*mono
                    source=source.normal_order().expand(); original=source.tokenize()
                    sid=legacy_id(original); source_key=expression_key(original,n)
                    reason=None
                    if not 1<=len(original)<=5:reason='invalid_source'
                    elif sid in excluded_sources or source_key in origin_keys or source_key in seen_keys:reason='duplicate_source'
                    current=source; changes=0; used=0
                    steps=rng.choice(settings['scramble_attempts'])
                    if reason is None:
                        validate(original,n,dim,hels)
                        for _ in range(steps):
                            before=current.tokenize(); ledger.hit('random_identity_calls'); used+=1
                            current=(api.random_apply_schouten(current,1) if rng.random()<.5
                                     else api.random_apply_momentum_conservation(current,n,1))[1]
                            current=current.normal_order().expand(); tokens=current.tokenize()
                            validate(tokens,n,dim,hels)
                            changes+=int(tokens!=before)
                            # These scramble APIs only accept strict term growth; no later shrink is skipped.
                            if len(tokens)>36 or max(map(len,tokens),default=0)>128:
                                reason='over_capacity';break
                    tokens=current.tokenize(); iid=legacy_id(tokens); input_key=expression_key(tokens,n)
                    if reason is None and not len(original)<len(tokens)<=36:reason='no_growth'
                    if reason is None and (iid in excluded_inputs or input_key in seen_keys or input_key in origin_keys):reason='duplicate_input'
                    if reason is None:
                        batch=build_policy_input([tokens],[n],[1],[1],max_terms=36)
                        if len(batch.actions.action_indices)==0:reason='no_legal_actions'
                    if reason is None:
                        row={'index':len(split_rows),'split':split,'id':iid,'source_id':sid,'n_point':n,
                             'dimension':dim,'helicities_twice':hels,'terms':len(tokens),'tokens':tokens}
                        split_rows.append(row);accepted.append(row)
                        excluded_sources.add(sid);excluded_inputs.add(iid);origin_keys.add(source_key);seen_keys.add(input_key)
                    else:rejected[reason]+=1
                    attempts.write(json.dumps({'candidate':serial,'split':split,'n_point':n,'dimension':dim,
                        'helicities_twice':hels,'accepted':reason is None,'reason':reason,'source_id':sid,'input_id':iid,
                        'source_key':source_key,'input_key':input_key,'source_terms':len(original),'terms':len(tokens),
                        'requested_scrambles':steps,'scramble_calls':used,'changed_scrambles':changes,
                        'counts':dict(ledger.counts)},separators=(',',':'))+'\n')
                    if reason is None and len(split_rows)%128==0:
                        attempts.flush()
                        print('GENERATE '+json.dumps({'split':split,'accepted':len(split_rows),'counts':dict(ledger.counts)}),flush=True)
                write_json(dataset/(split+'.json'),split_rows)
        summary={'status':'completed','dataset_id':cfg['dataset_id'],'counts':dict(ledger.counts),'rejected':dict(rejected),'reused_training_inputs':len(reuse),
                 'candidate_point_dimension':dict(candidates),'elapsed_seconds':time.monotonic()-started,
                 'splits':{s:distribution([r for r in accepted if r['split']==s]) for s in ('train','test')},
                 'files_sha256':{s+'.json':digest(dataset/(s+'.json')) for s in ('train','test')},
                 'input_only':True,'inverse_action_labels':False,'source_disjoint':True,
                 'native_rng_seeded':False,'configuration_generator':'exact function definition data/generation.py::amplitude_config_gen'}
        write_json(dataset/'manifest.json',summary);write_json(folder/'completed.json',summary)
        print('DATASET_COMPLETE '+json.dumps(summary),flush=True)
        return summary
    except BaseException as exc:
        write_json(folder/'failed.json',{'error':repr(exc),'counts':dict(ledger.counts),'accepted':len(accepted)})
        if accepted:write_json(dataset/'partial_inputs.json',accepted)
        raise

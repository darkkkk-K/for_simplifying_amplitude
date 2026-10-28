"""Audited 32x16x24 single-action PPO experiment. Explicit CLI, default off."""
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
from unittest.mock import patch

from .packed32_common import (REPO,read_json,write_json,append,digest,Ledger,key,tensor_digest,
                             expression_from_tokens,reference_scores,source_hashes,check_sources,paired)
from .full32_common import configure_torch

RUN = "single24_shaped32_20261024_v1"
DATA = "single2048_20261024_v1"
SEED = 103
CAPS = {"train_engine":12288,"eval_engine":36864,"encoder_forwards":131072,
        "preparations":98304,"rollout_forwards":98304,"reference_forwards":12288,
        "update_forwards":1536,"autograd":1536,"optimizer":1536,"frozen_feature_reuses":24576,
        "checkpoint_loads":2,"small_state_restores":1,"reconstructions":32768,"generation":0}


def locations():
    from project.paths import run_path,contract_path
    return run_path(RUN),contract_path(RUN)


def seed_for(*parts): return int(key(RUN,*parts)[:15],16)


def inputs(split):
    folder=REPO/'datasets'/DATA; manifest=read_json(folder/'manifest.json'); file=folder/(split+'.json')
    if manifest['status']!='completed' or digest(file)!=manifest['files_sha256'][file.name]: raise ValueError('Dataset changed')
    rows=read_json(file)
    if len(rows)!=(1792 if split=='train' else 256): raise ValueError('Wrong split size')
    for r in rows:
        if set(r)!={'index','split','id','source_id','n_point','terms','tokens'} or r['split']!=split or r['n_point']!=4:
            raise ValueError('Target-bearing or incorrect data record')
        if hashlib.sha256(repr(r['tokens']).encode()).hexdigest()!=r['id'] or not 1<=len(r['tokens'])<=32:
            raise ValueError('Input content mismatch')
    return rows


def plan():
    return {"id":RUN,"dataset":DATA,"seed":SEED,"rounds":32,"lanes":16,"steps_per_lane":24,
        "steps_per_round":384,"optimizer_steps_per_round":48,"optimizer_steps_total":1536,
        "ppo":{"gamma":1.,"gae_lambda":.95,"clip_epsilon":.2,"value_coef":.5,
               "learning_rate":3e-4,"max_grad_norm":.5,"minibatch_size":16,"update_epochs":2,"normalize_advantage":True},
        "rewards":{"segment_enabled":True,"potential_enabled":True,"segment_scale":.5,"potential_scale":.25},
        "checkpoints":[10,16,20,22,23,24],"beta":.4,
        "potential":"mean_i max_j positive_cosine(frozen_encoder_terms)*exp(-L1(exponents_i-exponents_j)/4); same dimension/little-group charges; terminal0",
        "reward_identity":"complete episode sum = best_reduce + final_segment_bonus - .25/L0 * initial_potential",
        "episode_horizon":24,"max_state_terms":128,"max_state_tokens":128,"max_actions":32768,
        "capacity_rule":"oversized child is a paid transition then terminal; retain best feasible expression; no token truncation",
        "sampling":"one categorical draw from packed structural+encoder support; no sibling search; query-history exclusion off",
        "restart":"fresh training-source sample after terminal; 16 streams of24 real transitions; GAE terminal mask; unfinished window tail bootstraps",
        "train_pool":1792,"test_pool":256,"test_sealed_until_training_complete":True,
        "test_methods":["trained_single24","initial_single24","u096_verified_k4_24calls"],"test_seeds":[401,503],
        "comparison":"all methods <=24 identity calls/input; paired by256 held-out original sources; no seed/endpoint selection",
        "success_condition":"trained-initial >=.005 with source-paired95 lower>0; deployment additionally requires same against u096-K4; single seed exploratory",
        "caps":CAPS,"buffer_cap_bytes":768*1024**2,"storage_cap_bytes":1024**3,
        "snapshots":[0,8,16,24,32],"recovery_files":1,"encoder_and_backbone_frozen":True,
        "actor_context":"log initial length, best/initial, first-best-stage/6, paid segment bonus scaled, steps to next checkpoint/24, potential",
        "stopping":"finite/probability/engine/grammar/budget/storage failures only; no outcome-based extension",
        "automatic_retry":False,"automatic_deployment":False,
        "causal_scope":"combined new sampler/expansion/context/reward trial, not a reward-only ablation"}


class SourceDeck:
    def __init__(self,rows,seed):
        self.rows=rows; self.rng=random.Random(seed); self.order=[]; self.visits=[]
    def sample(self,used):
        for _ in range(2*len(self.rows)):
            if not self.order:
                self.order=list(range(len(self.rows))); self.rng.shuffle(self.order)
            row=self.rows[self.order.pop()]
            if row['source_id'] in used: continue
            used.add(row['source_id']); self.visits.append(row['source_id']); return row
        raise RuntimeError('No unused training source for this rollout')


class Runtime:
    def __init__(self,out,ledger,loaded,model,config):
        import forsythia_hep as api
        self.out,self.ledger,self.reference,self.model,self.config=out,ledger,loaded.reference,model,config
        self.phase='train'; self.apply=api.apply_identity_at; self.seen=set(); self.state_count=0
        self.stream=gzip.open(out/'states.jsonl.gz','wt',encoding='utf-8'); self.episode_serial=0
        self.frozen_before=tensor_digest(self.reference.state_dict())
        self.hooks=[self.reference.embdding.register_forward_pre_hook(lambda *_:ledger.hit('encoder_forwards')),
                    self.reference.register_forward_pre_hook(lambda *_:ledger.hit('reference_forwards'))]
        def forbid(*args,**kwargs): ledger.hit('generation'); raise RuntimeError('Generation forbidden in RL')
        self.guards=[patch.object(api,n,forbid) for n in ('random_monomial','random_apply_schouten','random_apply_momentum_conservation','simplify')]
        for p in self.guards:p.start()

    def state(self,expr):
        from .legal_actions import content_id
        tokens=expr.tokenize(); sid=content_id(tokens)
        if sid not in self.seen:
            self.stream.write(json.dumps({'id':sid,'tokens':tokens},separators=(',',':'))+'\n')
            self.seen.add(sid); self.state_count+=1
        return sid

    def frozen_check(self):
        if tensor_digest(self.reference.state_dict())!=self.frozen_before or any(p.grad is not None for p in self.reference.parameters()):
            raise ValueError('Frozen encoder/reference changed')

    def close(self):
        self.stream.close()
        for h in self.hooks:h.remove()
        for p in reversed(self.guards):p.stop()

    def prepare(self,expr,progress):
        import torch
        from .legal_actions import build_policy_input
        from .single24_policy import condition
        tokens=expr.tokenize()
        if not tokens:return None,'zero_expression'
        if len(tokens)>128 or max(map(len,tokens),default=0)>128:return None,'capacity'
        batch=build_policy_input([tokens],[4],[24-progress.step],[24],max_terms=128,device='cuda')
        self.ledger.hit('preparations')
        state=self.model.prepare(batch,progress.initial)
        if state is None:return None,'no_legal_actions'
        return condition(state,progress),None

    def score(self,state):
        import torch
        self.ledger.hit('rollout_forwards')
        with torch.no_grad(): output,value=self.model([state])
        return output,float(value[0])

    def start(self,row):
        from .single24_policy import Progress
        self.ledger.hit('reconstructions'); self.episode_serial+=1
        expr=expression_from_tokens(row['tokens']); progress=Progress(len(row['tokens']),len(row['tokens']))
        state,reason=self.prepare(expr,progress)
        return {'expr':expr,'progress':progress,'state':state,'source':row,'episode':self.episode_serial,
                'initial_state':self.state(expr),'initial_phi':state.potential if state else 0.,'reason':reason}

    def transition(self,episode,generator):
        from .packed_sampling import draw_actions
        from .single24_policy import Progress,condition
        expr,progress,state=episode['expr'],episode['progress'],episode['state']
        output,value=self.score(state); record=draw_actions(output,0,1,generator)
        action=record.selected_actions[0]; before_tokens=expr.tokenize(); parent=self.state(expr)
        self.ledger.hit(self.phase+'_engine'); serial=self.ledger.counts[self.phase+'_engine']
        append(self.out/'calls.jsonl',{'event':'start','phase':self.phase,'serial':serial,'parent':parent,'action':action,'n_point':4})
        try: child=self.apply(expr,action,4)[0]
        except BaseException as exc:
            append(self.out/'calls.jsonl',{'event':'error','phase':self.phase,'serial':serial,'error':repr(exc)});raise
        if expr.tokenize()!=before_tokens:raise ValueError('Engine mutated parent')
        child_id=self.state(child); length=len(child.tokenize())
        append(self.out/'calls.jsonl',{'event':'end','phase':self.phase,'serial':serial,'child':child_id,'terms':length})
        reason='horizon' if progress.step+1==24 else None
        preliminary=Progress(progress.initial,progress.best,progress.step+1,progress.best_stage,progress.paid_bonus)
        next_state=None
        if reason is None:next_state,reason=self.prepare(child,preliminary)
        # Bounds apply even at the last task step; no out-of-cap result is returned as best.
        if length>128 or max(map(len,child.tokenize()),default=0)>128:reason='capacity';next_state=None
        terminal=reason is not None
        feasible_length=progress.best if reason=='capacity' else length
        new_progress,reward=progress.advance(feasible_length,state.potential,next_state.potential if next_state else 0.,terminal,self.config)
        if next_state is not None:next_state=condition(next_state,new_progress)
        episode.update(expr=child,progress=new_progress,state=next_state,reason=reason)
        row={'episode':episode['episode'],'source_id':episode['source']['source_id'],'initial_state':episode['initial_state'],
             'initial_terms':progress.initial,'initial_phi':episode['initial_phi'],'step':new_progress.step,
             'parent':parent,'next':child_id,'before':len(before_tokens),'after':length,'serial':serial,
             'progress_before':asdict(progress),'progress_after':asdict(new_progress),'reward':reward,
             'phi_before':state.potential,'phi_after':next_state.potential if next_state else 0.,
             'terminal':terminal,'reason':reason,'record':record.to_dict(),'actions':output.actions_for(),
             'logits':output.logits.detach().cpu().double().tolist(),'old_value':value}
        return state,record,value,row


def collect(runtime,deck,round_index):
    import torch
    from .packed_ppo import compute_gae,PPOBuffer
    states,records,values,advantages,targets,rows=[],[],[],[],[],[]
    used=set(); streams=[]; empties=[]
    for lane in range(16):
        generator=torch.Generator().manual_seed(seed_for('train',round_index,lane))
        ep=None; local=[]; local_values=[]; rewards=[]; terminals=[]
        for tick in range(24):
            while ep is None or ep['state'] is None:
                ep=runtime.start(deck.sample(used))
                if ep['state'] is None:empties.append({'source_id':ep['source']['source_id'],'episode':ep['episode'],'reason':ep['reason']})
            state,record,value,row=runtime.transition(ep,generator)
            row.update(lane=lane,tick=tick); local.append(row); local_values.append(value)
            states.append(state); records.append(record); rewards.append(row['reward']['total']); terminals.append(row['terminal'])
        bootstrap=runtime.score(ep['state'])[1] if not terminals[-1] else 0.
        a,ret=compute_gae(rewards,local_values,terminals,bootstrap_value=bootstrap,gamma=1.,gae_lambda=.95)
        rows.extend(local); values.extend(local_values); advantages.extend(a.tolist()); targets.extend(ret.tolist())
        streams.append({'lane':lane,'bootstrap':bootstrap,'terminal_at_end':terminals[-1]})
    if len(states)!=384:raise ValueError('Expected384 real transitions')
    size=sum(s.tensor_bytes() for s in states)
    if size>768*1024**2:raise RuntimeError('Frozen feature memory cap exceeded')
    vector=lambda a:torch.tensor(a,dtype=torch.float64)
    buffer=PPOBuffer(states,records,vector([sum(r.conditional_log_probs) for r in records]),vector(values),
                     vector(advantages),vector(targets),rows)
    return buffer,{'rows':rows,'streams':streams,'empty_initial_episodes':empties,
                  'advantages':advantages,'targets':targets,'buffer_bytes':size,'unique_sources':len(used)}


def save_json_gz(path,value):
    with gzip.open(path,'xt',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=True,allow_nan=False,separators=(',',':'))


def save_model(path,model,optimizer=None,**metadata):
    import torch
    value={**model.small_state_dict(),'metadata':metadata}
    if optimizer is not None:value['optimizer']=optimizer.state_dict()
    tmp=path.with_suffix('.tmp')
    with tmp.open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    if path.exists() and path.name!='latest_recovery.pt':raise FileExistsError(path)
    os.replace(tmp,path)


def evaluate_single(runtime,rows,tag):
    import torch
    results=[]; folder=runtime.out/'evaluation'/tag;folder.mkdir(parents=True)
    for i,row in enumerate(rows):
        for seed in (401,503):
            ep=runtime.start(row); trace=[]; generator=torch.Generator().manual_seed(seed_for('test',seed,row['id']))
            while ep['state'] is not None:
                _,_,_,r=runtime.transition(ep,generator);trace.append(r)
                if r['terminal']:break
            result={'model':tag,'source_id':row['source_id'],'index':row['index'],'evaluation_seed':seed,
                'initial_terms':row['terms'],'best_terms':ep['progress'].best,'final_terms':len(ep['expr'].tokenize()),
                'return':(row['terms']-ep['progress'].best)/row['terms'],'calls':len(trace),'reason':ep['reason'],
                'trace':trace}
            path=folder/f"case_{row['index']}_s{seed}.json.gz"; save_json_gz(path,result)
            results.append({k:v for k,v in result.items() if k!='trace'}|{'file':path.relative_to(runtime.out).as_posix(),'sha256':digest(path)})
        if (i+1)%32==0:print(f'test {tag}: {i+1}/256',flush=True)
    return results


def evaluate_k4(runtime,rows):
    import torch
    from .legal_actions import build_policy_input
    from .packed_policy import CandidateOutput,segmented_log_softmax
    from .packed_sampling import draw_actions
    results=[];folder=runtime.out/'evaluation/u096_verified_k4_24calls';folder.mkdir(parents=True)
    for i,row in enumerate(rows):
        for seed in (401,503):
            runtime.ledger.hit('reconstructions'); expr=expression_from_tokens(row['tokens']); calls=0; history=defaultdict(set);trace=[];reason='horizon'
            gen=torch.Generator().manual_seed(seed_for('test',seed,row['id']))
            while calls<24 and expr.tokenize():
                parent=runtime.state(expr);tokens=expr.tokenize();batch=build_policy_input([tokens],[4],[24-calls],[24],[history[parent]],max_terms=128,device='cuda')
                if not bool(batch.actions.query_allowed.any()):reason='no_legal_actions';break
                with torch.no_grad():raw,allowed=reference_scores(runtime.reference,batch)
                keep=allowed&(raw>0);table=batch.actions.select(keep,1);logits=raw[keep].log()
                if not len(logits):reason='no_legal_actions';break
                output=CandidateOutput(logits,segmented_log_softmax(logits,table.action_ptr),logits,table,batch.context_ids,'u096_k4_24_v1')
                record=draw_actions(output,0,min(4,24-calls),gen);lookup=dict(zip(output.actions_for(),logits.exp().cpu().tolist()));children=[]
                for action in record.selected_actions:
                    runtime.ledger.hit('eval_engine');calls+=1;serial=runtime.ledger.counts['eval_engine']
                    append(runtime.out/'calls.jsonl',{'event':'start','phase':'eval','serial':serial,'parent':parent,'action':action,'n_point':4})
                    child=runtime.apply(expr,action,4)[0];cid=runtime.state(child);length=len(child.tokenize());history[parent].add(action)
                    append(runtime.out/'calls.jsonl',{'event':'end','phase':'eval','serial':serial,'child':cid,'terms':length})
                    children.append({'action':action,'state':cid,'terms':length,'tie':lookup[action],'expr':child,'serial':serial})
                win=min(children,key=lambda x:(x['terms'],-x['tie'],x['action']))
                if win['terms']<=len(tokens):expr=win['expr']
                trace.append({'parent':parent,'next':runtime.state(expr),'before':len(tokens),'after':len(expr.tokenize()),'record':record.to_dict(),
                    'actions':output.actions_for(),'logits':logits.cpu().double().tolist(),
                    'children':[{k:v for k,v in c.items() if k!='expr'} for c in children]})
            final=len(expr.tokenize());result={'model':'u096_verified_k4_24calls','source_id':row['source_id'],'index':row['index'],'evaluation_seed':seed,
                'initial_terms':row['terms'],'best_terms':final,'final_terms':final,'return':(row['terms']-final)/row['terms'],'calls':calls,'reason':reason,'trace':trace}
            path=folder/f"case_{row['index']}_s{seed}.json.gz";save_json_gz(path,result)
            results.append({k:v for k,v in result.items() if k!='trace'}|{'file':path.relative_to(runtime.out).as_posix(),'sha256':digest(path)})
        if (i+1)%32==0:print(f'test u096-K4: {i+1}/256',flush=True)
    return results


def summarize(rows):
    sources=sorted({r['source_id'] for r in rows});vectors={}
    for tag in plan()['test_methods']:
        values=[]
        for source in sources:
            subset=[r for r in rows if r['model']==tag and r['source_id']==source]
            if len(subset)!=2 or {r['evaluation_seed'] for r in subset}!={401,503}:raise ValueError('Incomplete paired test')
            values.append(statistics.fmean(r['return'] for r in subset))
        vectors[tag]=values
    if len(sources)!=256:raise ValueError('Wrong test source count')
    comparisons={name:paired(vectors['trained_single24'],vectors[name]) for name in ('initial_single24','u096_verified_k4_24calls')}
    return {'means':{k:statistics.fmean(v) for k,v in vectors.items()},'comparisons':comparisons,
        'learning_condition_passed':comparisons['initial_single24']['mean']>=.005 and comparisons['initial_single24']['ci95'][0]>0,
        'deployment_condition_passed':all(x['mean']>=.005 and x['ci95'][0]>0 for x in comparisons.values()),
        'sources':256,'default_model_changed':False,'single_seed_exploratory':True}


def freeze():
    from project.paths import asset_path
    out,contract=locations()
    if out.exists() or (contract/'protocol.json').exists():raise FileExistsError('Existing experiment; no overwrite/retry')
    training=inputs('train'); data=REPO/'datasets'/DATA
    engine=Path(importlib.util.find_spec('forsythia_hep').origin)
    code=source_hashes(('project/rl/single24_policy.py','project/rl/single24_experiment.py','project/rl/packed_ppo.py',
        'project/rl/full32_common.py','scripts/run_single24.py','tests/test_single24.py'))
    protocol=plan()|{'code_sha256':code,'dataset_sha256':digest(data/'manifest.json'),'input_sha256':read_json(data/'manifest.json')['files_sha256'],
        'encoder_sha256':digest(asset_path('checkpoints/temp15_final_model_sl_256.pth')),
        'baseline_sha256':digest(asset_path('checkpoints_p1_depth2_full160_160u_seed42/grpo_update_0096.pth')),
        'engine_sha256':{str(p):digest(p) for p in [engine,*engine.parent.glob('*.pyd')]},
        'cleanup_completed_sha256':digest(REPO/'artifacts/maintenance/retire_checkpoints_20261024_v1/completed.json')}
    contract.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True);(out/'snapshots').mkdir();(out/'training').mkdir()
    write_json(contract/'protocol.json',protocol);write_json(out/'started.json',{'protocol_sha256':digest(contract/'protocol.json'),'pid':os.getpid()})
    return out,contract,protocol,Ledger(CAPS),training


def run():
    out,contract,protocol,ledger,training=freeze();runtime=None
    try:
        torch=configure_torch()
        from .packed_factory import load_candidate_policy
        from .packed_policy import CandidateConfig
        from .single24_policy import SingleActorCritic,RewardConfig
        from .packed_ppo import PPOConfig,update_ppo
        original=torch.load
        def counted(*a,**kw):ledger.hit('checkpoint_loads');return original(*a,**kw)
        with patch('torch.load',counted):loaded=load_candidate_policy(mode='packed',device='cuda',config=CandidateConfig(symbolic_features=True),seed=SEED)
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(SEED);model=SingleActorCritic(loaded.policy,enabled=True).to('cuda')
        runtime=Runtime(out,ledger,loaded,model,RewardConfig(**protocol['rewards']))
        initial=model.small_state_dict();save_model(out/'snapshots/round_000.pt',model,round=0)
        write_json(out/'model.json',loaded.metadata|{'trainable_parameters':sum(p.numel() for p in model.trainable_parameters())})
        cfg=PPOConfig(**protocol['ppo']);optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=cfg.learning_rate,weight_decay=0.)
        deck=SourceDeck(training,seed_for('source_deck'));rounds=[]
        for u in range(1,33):
            where=out/'training'/f'rollout_{u:03d}';where.mkdir()
            before=tensor_digest({k:v for part,sd in model.small_state_dict().items() for k,v in [(part+'.'+n,t) for n,t in sd.items()]})
            buffer,record=collect(runtime,deck,u)
            after_collection=tensor_digest({part+'.'+n:t for part,sd in model.small_state_dict().items() for n,t in sd.items()})
            if before!=after_collection:raise ValueError('Model changed during rollout')
            save_json_gz(where/'buffer.json.gz',record)
            updates=update_ppo(model,optimizer,buffer,cfg,generator=torch.Generator().manual_seed(seed_for('minibatches',u)),
                               hit=ledger.hit,on_step=lambda r:append(where/'optimizer_steps.jsonl',r))
            if len(updates)!=48 or any(r['batch_size']!=16 for r in updates):raise ValueError('Wrong48x16 PPO updates')
            rewards={k:sum(r['reward'][k] for r in record['rows']) for k in ('main','segment','potential','total')}
            info={'round':u,'real_transitions':384,'optimizer_steps':48,'unique_sources':record['unique_sources'],
                  'episodes_started':len({r['episode'] for r in record['rows']})+len(record['empty_initial_episodes']),
                  'terminated_episodes':sum(r['terminal'] for r in record['rows']),
                  'truncated_streams':sum(not r['terminal_at_end'] for r in record['streams']),
                  'rewards':rewards,'epoch_loss':{str(ep):{k:statistics.fmean(r[k] for r in updates if r['epoch']==ep) for k in ('actor_loss','value_loss','loss','approx_kl','clip_fraction')} for ep in (1,2)},
                  'buffer_sha256':digest(where/'buffer.json.gz'),'counts':dict(ledger.counts)}
            write_json(where/'completed.json',info);rounds.append(info)
            save_model(out/'snapshots/latest_recovery.pt',model,optimizer,round=u,deck_order=deck.order,source_visits=deck.visits,counts=dict(ledger.counts))
            if u in (8,16,24,32):save_model(out/f'snapshots/round_{u:03d}.pt',model,round=u)
            runtime.frozen_check()
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>protocol['storage_cap_bytes']:raise RuntimeError('Storage cap')
            print(f"rollout {u}/32; steps=384; optimizer={ledger.counts['optimizer']}; sources={len(set(deck.visits))}; value_loss={info['epoch_loss']['2']['value_loss']:.5f}",flush=True)
        if ledger.counts['optimizer']!=1536 or ledger.counts['train_engine']!=12288:raise ValueError('Training total mismatch')
        write_json(out/'training_completed.json',{'rounds':rounds,'unique_training_sources':len(set(deck.visits)),'source_visits':deck.visits,'counts':dict(ledger.counts)})
        # Test contents become available to the experiment only after all32 rounds.
        test=inputs('test')
        if {r['source_id'] for r in test}&{r['source_id'] for r in training}:raise ValueError('Source overlap')
        runtime.phase='eval';results=evaluate_single(runtime,test,'trained_single24')
        ledger.hit('small_state_restores');model.load_small_state(initial)
        results+=evaluate_single(runtime,test,'initial_single24');results+=evaluate_k4(runtime,test)
        write_json(out/'evaluation.json',results);runtime.frozen_check();runtime.close();runtime=None
        report=summarize(results)|{'counts':dict(ledger.counts),'rounds':rounds,'training_unique_sources':len(set(deck.visits)),
                                  'frozen_unchanged':True,'test_access_after_training':True}
        check_sources(protocol);write_json(out/'report.json',report)
        files=[]
        for directory,_,names in os.walk(out):
            for name in sorted(names):
                p=Path(directory)/name
                with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
                files.append({'path':p.relative_to(REPO).as_posix(),'sha256':sha})
        write_json(contract/'result_index.json',{'files':files})
        write_json(out/'completed.json',{'report_sha256':digest(out/'report.json'),'protocol_sha256':digest(contract/'protocol.json'),'counts':dict(ledger.counts)})
        print(json.dumps({k:report[k] for k in ('means','comparisons','counts')},indent=2),flush=True)
    except BaseException as exc:
        write_json(out/'failed.json',{'error':repr(exc),'counts':dict(ledger.counts),'automatic_retry':False});raise
    finally:
        if runtime is not None:runtime.close()

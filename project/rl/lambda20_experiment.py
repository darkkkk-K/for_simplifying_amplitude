"""Four matched GAE-lambda arms,480 transitions/round, sealed512-source test."""
from dataclasses import asdict
import gzip
import importlib.util
import json
import math
from pathlib import Path
import statistics
from unittest.mock import patch

from .packed32_common import REPO, Ledger, append, check_sources, digest, expression_from_tokens, key, paired, read_json, source_hashes, tensor_digest, write_json
from .early_fusion_experiment import Runtime as PreviousRuntime, progress_report
from .single24_experiment import SourceDeck, inputs, save_json_gz, save_model
from data.generation_eval1024 import DATA_ID

RUN = 'early_fusion_lambda20_20261024_v1'
ARMS = (('lambda_095', .95), ('lambda_096', .96), ('lambda_098', .98), ('lambda_094', .94))
CAPS = {'checkpoint_loads': 1, 'old_rl_checkpoint_loads': 0, 'train_engine': 69120,
    'dev_engine': 102400, 'test_engine': 61440, 'encoder_forwards': 250000, 'policy_forwards': 250000,
    'preparations': 250000, 'sampling': 232960, 'reconstructions': 76000, 'generation': 0,
    'update_forwards': 8640, 'autograd': 8640, 'optimizer': 8640, 'encoder_feature_reuses': 138240,
    'small_state_restores': 12, 'cache_hits': 0}


def locations(): return REPO/'artifacts/runs'/RUN, REPO/'experiments'/RUN
def seed_for(*parts): return int(key(RUN, *parts)[:15], 16)


def plan():
    return {'id': RUN, 'arms': dict(ARMS), 'seed': 103, 'horizon': 20, 'beta': .4,
        'checkpoints': [8,13,16,18,19,20], 'rounds_per_arm': 36, 'lanes': 24, 'steps_per_lane': 20,
        'transitions_per_round': 480, 'minibatch': 16, 'epochs': 2, 'updates_per_round': 60,
        'updates_per_arm': 2160, 'updates_total': 8640, 'train_calls_per_arm': 17280,
        'initialization': 'one freshly initialized seed103 policy; identical small-state tensors restored per arm; new AdamW per arm',
        'paired_seeds': 'source deck,rollout,minibatch seeds independent of arm; first behavior rollout must match exactly',
        'encoder_only_frozen': True, 'top_k': 16, 'point_features': 'early32 algebra+3 true n_point onehot',
        'support': 'all structural legal actions; dynamic top16; no encoder threshold or query-history exclusions',
        'ppo': {'gamma': 1., 'clip': .2, 'value_coef': .5, 'grad_clip': .5, 'lr': .0003, 'weight_decay': 0., 'entropy_coef': 0.},
        'rewards': {'segment_enabled': True, 'potential_enabled': True, 'segment_scale': .5, 'potential_scale': .25},
        'max_terms': 128, 'max_tokens': 128, 'max_actions': 32768,
        'training_data': 'single2048_20261024_v1/train.json', 'evaluation_data': DATA_ID,
        'evaluation_seeds': [401,503], 'dev_sources': 512, 'test_sources': 512,
        'selection': 'max dev mean actual final reduce; exact ties closest to .95 then smallest lambda',
        'test_models': 'selected,lambda_095,initial; deduplicate; read test only after sealed selection.json',
        'confirmation': 'selected-.95 final gain>=.005 and source-paired95 lower>0; if selected=.95 only learning-vs-initial, no lambda-gain claim',
        'report_every_rounds': 4, 'report_metric': 'completed episodes actual final reduce; unfinished segments separate',
        'caps': CAPS, 'buffer_cap_bytes': 2*1024**3, 'storage_cap_bytes': 4*1024**3,
        'snapshots': 'one initial,each arm endpoint36 and one latest recovery; no encoder duplicates',
        'automatic_retry': False, 'automatic_deployment': False, 'training_seed_count': 1,
        'stopping': 'all4x36 fixed rounds except mechanical failures; no outcome-based extension'}


def read_eval_inputs(split, out):
    if split == 'test' and not (out/'selection.json').is_file(): raise RuntimeError('Test sealed until selection')
    folder = REPO/'datasets'/DATA_ID; manifest = read_json(folder/'manifest.json'); path = folder/(split+'.json')
    frozen = read_json(locations()[1]/'protocol.json')
    if digest(folder/'manifest.json') != frozen['evaluation_manifest_sha256'] or digest(path) != frozen['input_sha256'][path.name]:
        raise ValueError('Frozen evaluation data changed')
    if manifest['status'] != 'completed' or digest(path) != manifest['files_sha256'][path.name]: raise ValueError('Evaluation data changed')
    rows = read_json(path)
    if len(rows) != 512 or len({r['source_id'] for r in rows}) != 512: raise ValueError('Wrong source count')
    from data.generation_full32 import token_id
    for row in rows:
        if set(row) != {'index','split','id','source_id','n_point','terms','tokens'} or row['split'] != split or row['n_point'] != 4:
            raise ValueError('Wrong/target-bearing record')
        if token_id(row['tokens']) != row['id'] or row['terms'] != len(row['tokens']) or not 2 <= row['terms'] <= 32:
            raise ValueError('Bad input record')
    return rows


def choose_arm(means):
    return min(dict(ARMS), key=lambda tag: (-means[tag]['final_reduce'], abs(dict(ARMS)[tag]-.95), dict(ARMS)[tag]))


def freeze():
    from project.paths import asset_path
    out, contract = locations()
    if out.exists() or (contract/'protocol.json').exists(): raise FileExistsError('Existing run; no overwrite/retry')
    training = inputs('train'); folder = REPO/'datasets'/DATA_ID; manifest = read_json(folder/'manifest.json')
    if manifest['status'] != 'completed': raise ValueError('Generation incomplete')
    retention = read_json(REPO/'project/layout/model_retention.json')
    for model in retention['protected_models']:
        if digest(REPO/model['path']) != model['sha256']: raise ValueError('Protected model changed')
    encoder = asset_path('checkpoints/temp15_final_model_sl_256.pth')
    engine = Path(importlib.util.find_spec('forsythia_hep').origin)
    files = ('project/rl/lambda20_policy.py','project/rl/lambda20_experiment.py','scripts/run_lambda20.py',
        'tests/test_lambda20.py','project/rl/early_fusion_policy.py','project/rl/early_fusion_experiment.py',
        'project/rl/single24_policy.py','project/rl/single24_experiment.py','project/rl/packed_ppo.py',
        'project/rl/full32_common.py','data/generation_eval1024.py','data/generation_full32.py')
    protocol = plan() | {'code_sha256': source_hashes(files), 'encoder_path': str(encoder), 'encoder_sha256': digest(encoder),
        'input_sha256': {'train': digest(REPO/'datasets/single2048_20261024_v1/train.json'), **manifest['files_sha256']},
        'evaluation_manifest_sha256': digest(folder/'manifest.json'), 'protected_models': retention['protected_models'],
        'git_index_sha256': digest(REPO/'.git/index'),
        'engine_sha256': {str(p): digest(p) for p in [engine,*engine.parent.glob('*.pyd')]}}
    contract.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True)
    for name in ('arms','snapshots','evaluation'): (out/name).mkdir()
    write_json(contract/'protocol.json', protocol); write_json(out/'started.json', {'protocol_sha256': digest(contract/'protocol.json')})
    return out, contract, protocol, Ledger(CAPS), training


class Runtime20(PreviousRuntime):
    def __init__(self, out, model, ledger, rewards, gae_lambda):
        super().__init__(out, model, ledger, rewards)
        self.lanes = [None]*24; self.gae_lambda = gae_lambda

    def prepare(self, expr, progress, n_point):
        from .legal_actions import build_policy_input
        tokens = expr.tokenize()
        if not tokens: return None, 'zero_expression'
        if len(tokens) > 128 or max(map(len,tokens), default=0) > 128: return None, 'capacity'
        self.ledger.hit('preparations')
        state = self.model.prepare(build_policy_input([tokens],[n_point],[20-progress.step],[20]), progress)
        return (state, None) if state is not None else (None, 'no_legal_actions')

    def start(self, source):
        from .lambda20_policy import Progress20
        self.ledger.hit('reconstructions'); self.serial += 1
        expr = expression_from_tokens(source['tokens']); sid = self.state(expr)
        progress = Progress20(len(source['tokens']),len(source['tokens']))
        state, reason = self.prepare(expr,progress,source['n_point'])
        ep = {'source': source, 'expr': expr, 'episode': self.serial, 'progress': progress,
            'initial_state': sid, 'initial_phi': state.potential if state else 0., 'state': state, 'reason': reason, 'best_state': sid}
        if state is None: self.finish(ep)
        return ep

    def transition(self, ep, generator):
        from .early_fusion_policy import sample_one
        from .lambda20_policy import Progress20, condition20
        state, progress, expr = ep['state'], ep['progress'], ep['expr']
        logits, mask, values = self.score(state)
        self.ledger.hit('sampling'); index, logp, top = sample_one(logits,mask,16,generator)
        t,i,j = state.actions[index].tolist(); action = (t,(i,j)); tokens = expr.tokenize()
        parent, n = self.state(expr), ep['source']['n_point']
        self.ledger.hit(self.phase+'_engine'); serial = self.ledger.counts[self.phase+'_engine']
        append(self.out/'calls.jsonl', {'event':'start','phase':self.phase,'serial':serial,'parent':parent,'action':action,'n_point':n})
        try: child = self.apply(expr,action,n)[0]
        except BaseException as exc:
            append(self.out/'calls.jsonl', {'event':'error','phase':self.phase,'serial':serial,'error':repr(exc)}); raise
        if expr.tokenize() != tokens: raise ValueError('Parent mutated')
        cid = self.state(child); length = len(child.tokenize())
        append(self.out/'calls.jsonl', {'event':'end','phase':self.phase,'serial':serial,'child':cid,'terms':length})
        reason = 'horizon' if progress.step+1 == 20 else None; next_state = None
        if reason is None:
            preliminary = Progress20(progress.initial,progress.best,progress.step+1,progress.best_stage,progress.paid_bonus)
            next_state, reason = self.prepare(child,preliminary,n)
        if length > 128 or max(map(len,child.tokenize()),default=0) > 128: reason, next_state = 'capacity', None
        effective = progress.best if reason == 'capacity' else length
        new_progress, reward = progress.advance(effective,state.potential,next_state.potential if next_state else 0.,reason is not None,self.rewards)
        if next_state is not None: next_state = condition20(next_state,new_progress)
        if new_progress.best < progress.best: ep['best_state'] = cid
        ep.update(expr=child,progress=new_progress,state=next_state,reason=reason)
        row = {'episode':ep['episode'],'source_id':ep['source']['source_id'],'n_point':n,
            'initial_state':ep['initial_state'],'initial_terms':progress.initial,'initial_phi':ep['initial_phi'],
            'step':new_progress.step,'parent':parent,'next':cid,'before':len(tokens),'after':length,'serial':serial,
            'actions':state.actions.tolist(),'logits':logits[0,mask[0]].detach().cpu().double().tolist(),
            'context_id':state.context_id,'selected_index':index,'top_indices':top,'old_logp':logp,'old_value':float(values[0]),
            'progress_before':asdict(progress),'progress_after':asdict(new_progress),'phi_before':state.potential,
            'phi_after':next_state.potential if next_state else 0.,'reward':reward,'terminal':reason is not None,'reason':reason}
        if reason is not None: self.finish(ep)
        return state, row


def collect20(runtime, deck, u):
    import torch
    from .packed_ppo import compute_gae
    runtime.round = u; states, rows, advantages, targets, streams = [], [], [], [], []; size = 0
    used = {ep['source']['source_id'] for ep in runtime.lanes if ep is not None and ep['state'] is not None}
    for lane in range(24):
        generator = torch.Generator().manual_seed(seed_for('train',u,lane)); ep = runtime.lanes[lane]; local = []
        for tick in range(20):
            while ep is None or ep['state'] is None: ep = runtime.start(deck.sample(used))
            state, row = runtime.transition(ep,generator); row.update(lane=lane,tick=tick,round=u)
            size += state.tensor_bytes()
            if size > 2*1024**3: raise RuntimeError('Buffer cap')
            states.append(state); local.append(row)
        runtime.lanes[lane] = ep
        bootstrap = float(runtime.score(ep['state'])[2][0]) if ep['state'] is not None else 0.
        a, target = compute_gae([r['reward']['total'] for r in local],[r['old_value'] for r in local],
            [r['terminal'] for r in local],bootstrap_value=bootstrap,gamma=1.,gae_lambda=runtime.gae_lambda)
        rows.extend(local); advantages.extend(a.tolist()); targets.extend(target.tolist())
        streams.append({'lane':lane,'bootstrap':bootstrap,'terminal_at_end':local[-1]['terminal']})
    if len(states) != 480: raise ValueError('Not480 transitions')
    return states, {'rows':rows,'streams':streams,'advantages':advantages,'targets':targets,'buffer_bytes':size,'gae_lambda':runtime.gae_lambda}


def behavior_digest(rows):
    from .legal_actions import content_id
    return content_id([{k:v for k,v in row.items() if k != 'serial'} for row in rows])


def evaluate20(runtime, rows, split, tag):
    import torch
    results = []; folder = runtime.out/'evaluation'/split/tag; folder.mkdir(parents=True)
    runtime.phase = split
    for i, source in enumerate(rows):
        for seed in (401,503):
            ep = runtime.start(source); trace = []; generator = torch.Generator().manual_seed(seed_for('evaluation',seed,source['id']))
            while ep['state'] is not None:
                _, row = runtime.transition(ep,generator); trace.append(row)
            result = dict(runtime.finished[-1], model=tag, split=split, index=source['index'], evaluation_seed=seed, trace=trace)
            path = folder/f"case_{source['index']}_s{seed}.json.gz"; save_json_gz(path,result)
            results.append({k:v for k,v in result.items() if k != 'trace'}|{'file':path.relative_to(runtime.out).as_posix(),'sha256':digest(path)})
        if (i+1)%64 == 0: print(f'EVAL {split} {tag}: {i+1}/512',flush=True)
    return results


def summarize(rows):
    sources = sorted({r['source_id'] for r in rows}); models = sorted({r['model'] for r in rows})
    if len(sources) != 512: raise ValueError('Expected512 eval sources')
    vectors = {}; means = {}
    for tag in models:
        vectors[tag] = {metric: [] for metric in ('final_reduce','best_reduce')}
        for sid in sources:
            group = [r for r in rows if r['model'] == tag and r['source_id'] == sid]
            if len(group) != 2 or {r['evaluation_seed'] for r in group} != {401,503}: raise ValueError('Incomplete source/seed pairing')
            for metric in vectors[tag]: vectors[tag][metric].append(statistics.fmean(r[metric] for r in group))
        means[tag] = {metric: statistics.fmean(v) for metric,v in vectors[tag].items()}
    comparisons = {}
    for tag in models:
        if tag in ('lambda_095','initial'): continue
        comparisons[tag] = {metric: paired(vectors[tag][metric],vectors['lambda_095'][metric]) for metric in ('final_reduce','best_reduce')}
    return {'means':means,'versus_lambda095':comparisons}, vectors


def run():
    out, contract, protocol, ledger, training = freeze(); runtime = None
    try:
        from .full32_common import configure_torch
        from project.pretraining.encoder import read_checkpoint, construct_encoder
        from .lambda20_policy import Horizon20ActorCritic, update_ppo480
        from .single24_policy import RewardConfig
        torch = configure_torch(); original_load = torch.load
        def counted(path,*args,**kwargs):
            if Path(path).resolve() != Path(protocol['encoder_path']).resolve():
                ledger.hit('old_rl_checkpoint_loads'); raise RuntimeError('Only standalone encoder loading permitted')
            ledger.hit('checkpoint_loads'); return original_load(path,*args,**kwargs)
        with patch('torch.load',counted): state, config = read_checkpoint(protocol['encoder_path'],num_heads=8)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(103); encoder = construct_encoder(state,config,'cpu')
            model = Horizon20ActorCritic(encoder,enabled=True,top_k=16).to('cuda').train()
        del state
        initial = model.small_state_dict(); initial_hash = tensor_digest(initial); endpoints = {}; endpoint_files = {}; arm_reports = {}; first_behavior = None
        write_json(out/'model.json',{'encoder_config':config,'trainable_parameters':sum(p.numel() for p in model.trainable_parameters()),
            'initial_tensor_sha256':initial_hash,'old_rl_checkpoint_loads':0,'trainable_groups':sorted({n.split('.')[0] for n,p in model.named_parameters() if p.requires_grad})})
        save_model(out/'snapshots/initial.pt',model,round=0)
        for tag, lam in ARMS:
            ledger.hit('small_state_restores'); model.load_small_state(initial)
            if tensor_digest(model.small_state_dict()) != initial_hash: raise ValueError('Different arm initialization')
            optimizer = torch.optim.AdamW(model.trainable_parameters(),lr=.0003,weight_decay=0.)
            folder = out/'arms'/tag; folder.mkdir()
            for name in ('training','progress','snapshots'): (folder/name).mkdir()
            count_before = dict(ledger.counts)
            write_json(folder/'started.json',{'gae_lambda':lam,'initial_tensor_sha256':initial_hash,'counts_before':count_before})
            runtime = Runtime20(folder,model,ledger,RewardConfig(**protocol['rewards']),lam)
            deck = SourceDeck(training,seed_for('source_deck')); summaries = []; reports = []
            for u in range(1,37):
                where = folder/'training'/f'rollout_{u:03d}'; where.mkdir(); before = model.small_state_dict()
                states, buffer = collect20(runtime,deck,u)
                if tensor_digest(before) != tensor_digest(model.small_state_dict()): raise ValueError('Parameters changed in rollout')
                if u == 1:
                    behavior = behavior_digest(buffer['rows'])
                    if first_behavior is None: first_behavior = behavior
                    elif behavior != first_behavior: raise ValueError('Initial behavior differs between lambda arms')
                    write_json(folder/'initial_behavior.json',{'sha256':behavior})
                save_json_gz(where/'buffer.json.gz',buffer)
                with gzip.open(where/'optimizer_steps.jsonl.gz','xt',encoding='utf-8') as stream:
                    updates = update_ppo480(model,optimizer,states,buffer['rows'],buffer['advantages'],buffer['targets'],
                        generator=torch.Generator().manual_seed(seed_for('update',u)),hit=ledger.hit,
                        on_step=lambda row:stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n'))
                if len(updates) != 60: raise ValueError('Not60 optimizer updates')
                after = model.small_state_dict(); groups = sorted({n.split('.')[0] for n in before if n != 'charges'})
                changes = {g:math.sqrt(sum(float((after[n].double()-before[n].double()).square().sum()) for n in before if n.split('.')[0] == g)) for g in groups}
                info = {'round':u,'gae_lambda':lam,'real_transitions':480,'optimizer_steps':60,'buffer_bytes':buffer['buffer_bytes'],
                    'buffer_sha256':digest(where/'buffer.json.gz'),'weight_changes':changes,
                    'epoch_loss':{str(ep):{metric:statistics.fmean(r[metric] for r in updates if r['epoch'] == ep)
                        for metric in ('actor_loss','value_loss','loss','support_changed_fraction','selected_action_excluded_fraction','clip_fraction')} for ep in (1,2)},
                    'counts':dict(ledger.counts)}
                write_json(where/'completed.json',info); summaries.append(info); runtime.frozen_check()
                save_model(folder/'snapshots/latest_recovery.pt',model,optimizer,round=u,gae_lambda=lam,
                    deck_order=deck.order,deck_rng=deck.rng.getstate(),source_visits=deck.visits,
                    lane_context=[{'tokens':ep['expr'].tokenize(),'progress':asdict(ep['progress']),'source':ep['source'],'reason':ep['reason'],'episode':ep['episode']} for ep in runtime.lanes],counts=dict(ledger.counts))
                print(f'ROLLOUT lambda={lam} {u}/36; optimizer_arm={u*60}; optimizer_total={ledger.counts["optimizer"]}',flush=True)
                if u%4 == 0:
                    progress = progress_report(runtime.finished,runtime.lanes,u)|{'gae_lambda':lam,'arm':tag}
                    reports.append(progress); write_json(folder/'progress'/f'round_{u:03d}.json',progress)
                    print('PROGRESS '+json.dumps(progress),flush=True)
                if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > protocol['storage_cap_bytes']: raise RuntimeError('Storage cap')
                del states,buffer,before,after,updates
            save_model(folder/'snapshots/round_036.pt',model,round=36,gae_lambda=lam)
            endpoints[tag] = model.small_state_dict(); endpoint_files[tag] = {'path':(folder/'snapshots/round_036.pt').relative_to(REPO).as_posix(),'sha256':digest(folder/'snapshots/round_036.pt')}
            delta_counts = {k:ledger.counts[k]-count_before.get(k,0) for k in CAPS}
            if delta_counts['optimizer'] != 2160 or delta_counts['train_engine'] != 17280: raise ValueError('Arm totals differ')
            arm_report = {'gae_lambda':lam,'rounds':summaries,'progress':reports,'unique_training_sources':len(set(deck.visits)),
                'source_visits':deck.visits,'counts':delta_counts,'initial_tensor_sha256':initial_hash,'initial_behavior_sha256':first_behavior}
            write_json(folder/'training_completed.json',arm_report); arm_reports[tag] = arm_report
            runtime.close(); runtime = None; del optimizer
            print(f'ARM_COMPLETE lambda={lam}; optimizer=2160; train_calls=17280',flush=True)
        write_json(out/'training_completed.json',{'arms':{tag:{k:v for k,v in info.items() if k != 'rounds'} for tag,info in arm_reports.items()},'endpoint_files':endpoint_files,'counts':dict(ledger.counts)})
        if ledger.counts['optimizer'] != 8640 or ledger.counts['train_engine'] != 69120: raise ValueError('Four-arm totals')
        development = read_eval_inputs('dev',out)
        if {r['source_id'] for r in development} & {r['source_id'] for r in training}: raise ValueError('Train/dev overlap')
        runtime = Runtime20(out,model,ledger,RewardConfig(**protocol['rewards']),.95); runtime.round = 36
        dev_results = []
        for tag in [t for t,_ in ARMS]+['initial']:
            ledger.hit('small_state_restores'); model.load_small_state(initial if tag == 'initial' else endpoints[tag])
            dev_results += evaluate20(runtime,development,'dev',tag)
        write_json(out/'dev_evaluation.json',dev_results)
        dev_summary,_ = summarize(dev_results); write_json(out/'dev_completed.json',dev_summary)
        selected = choose_arm(dev_summary['means']); selection = {'selected':selected,'gae_lambda':dict(ARMS)[selected],
            'rule':protocol['selection'],'development_summary_sha256':digest(out/'dev_completed.json'),
            'endpoint':endpoint_files[selected],'test_data_read':False}
        write_json(out/'selection.json',selection)
        print('SELECTION '+json.dumps(selection),flush=True)
        # This is the first semantic read of the sealed test by the RL experiment.
        test = read_eval_inputs('test',out)
        if {r['source_id'] for r in test} & ({r['source_id'] for r in training}|{r['source_id'] for r in development}): raise ValueError('Test overlap')
        write_json(out/'test_consumption.json',{'selection_sha256':digest(out/'selection.json'),'test_sha256':protocol['input_sha256']['test.json'],
            'models':list(dict.fromkeys([selected,'lambda_095','initial']))})
        test_results = []
        for tag in dict.fromkeys([selected,'lambda_095','initial']):
            ledger.hit('small_state_restores'); model.load_small_state(initial if tag == 'initial' else endpoints[tag])
            test_results += evaluate20(runtime,test,'test',tag)
        write_json(out/'test_evaluation.json',test_results); test_summary,vectors = summarize(test_results)
        versus_initial = {metric:paired(vectors[selected][metric],vectors['initial'][metric]) for metric in ('final_reduce','best_reduce')}
        gain = test_summary['versus_lambda095'].get(selected,{}).get('final_reduce')
        report = {'development':dev_summary,'selection':selection,'test':test_summary,'selected_minus_initial':versus_initial,
            'lambda_gain_confirmed':None if selected == 'lambda_095' else gain['mean'] >= .005 and gain['ci95'][0] > 0,
            'learning_confirmed':versus_initial['final_reduce']['mean'] >= .005 and versus_initial['final_reduce']['ci95'][0] > 0,
            'counts':{k:ledger.counts[k] for k in CAPS},'single_training_seed':True,'default_model_changed':False}
        runtime.frozen_check(); runtime.close(); runtime = None; check_sources(protocol)
        for item in protocol['protected_models']:
            if digest(REPO/item['path']) != item['sha256']: raise ValueError('Protected model changed')
        for split in ('dev','test'):
            if digest(REPO/'datasets'/DATA_ID/(split+'.json')) != protocol['input_sha256'][split+'.json']:
                raise ValueError('Evaluation input changed during run')
        if digest(REPO/'datasets/single2048_20261024_v1/train.json') != protocol['input_sha256']['train']:
            raise ValueError('Training input changed during run')
        for endpoint in endpoint_files.values():
            if digest(REPO/endpoint['path']) != endpoint['sha256']: raise ValueError('Saved endpoint changed')
        if digest(REPO/'.git/index') != protocol['git_index_sha256']: raise ValueError('Git index changed')
        write_json(out/'report.json',report)
        write_json(contract/'result_index.json',{'files':[{'path':p.relative_to(REPO).as_posix(),'sha256':digest(p)} for p in sorted(out.rglob('*')) if p.is_file()]})
        write_json(out/'completed.json',{'report_sha256':digest(out/'report.json'),'protocol_sha256':digest(contract/'protocol.json'),'counts':report['counts']})
        print(json.dumps(report,indent=2),flush=True)
    except BaseException as exc:
        write_json(out/'failed.json',{'error':repr(exc),'counts':dict(ledger.counts),'automatic_retry':False}); raise
    finally:
        if runtime is not None: runtime.close()

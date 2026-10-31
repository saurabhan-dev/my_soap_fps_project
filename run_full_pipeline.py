from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run SOAP-based structural redundancy filtering for an MLIP CFG file.'
    )
    parser.add_argument('--input', type=Path, default=Path('all.cfg'),
                        help='Input MLIP CFG file (default: all.cfg).')
    parser.add_argument('--output', type=Path, default=Path('static_filtering_output'),
                        help='Output directory (default: static_filtering_output).')
    parser.add_argument('--test-n-configs', type=int, default=30,
                        help='Number of configurations used by the validation test (default: 30).')
    parser.add_argument('--skip-test', action='store_true',
                        help='Skip the small test run and execute only the full pipeline.')
    parser.add_argument('--seed', type=int, default=20261001,
                        help='Random seed used for diagnostics (default: 20261001).')
    return parser.parse_args()


ARGS = parse_args()
INPUT_CFG = ARGS.input
OUTPUT_DIR = ARGS.output
TEST_MODE = not ARGS.skip_test
TEST_N_CONFIGS = ARGS.test_n_configs
RANDOM_SEED = ARGS.seed
ELEMENTS=['Zn','O']; TYPE_TO_ELEMENT={0:'Zn',1:'O'}
SOAP_PARAMS=dict(r_cut=5.5,n_max=8,l_max=6,sigma=0.4,periodic=True)
OUTPUT_DIR.mkdir(exist_ok=True); (OUTPUT_DIR/'plots').mkdir(exist_ok=True); (OUTPUT_DIR/'soap_cache').mkdir(exist_ok=True)
print(INPUT_CFG.resolve(),OUTPUT_DIR.resolve())

import sys,re,json,hashlib,importlib.metadata
from collections import Counter
from datetime import datetime,timezone
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from ase import Atoms
from dscribe.descriptors import SOAP
import ase,dscribe,scipy,sklearn
DSCRIBE_VERSION=importlib.metadata.version('dscribe')
print('python',sys.executable,sys.version.split()[0],'ASE',ase.__version__,'DScribe',DSCRIBE_VERSION,'NumPy',np.__version__,'SciPy',scipy.__version__,'sklearn',sklearn.__version__)
def parse_mlip_cfg(path):
    text=Path(path).read_text(); starts=[m.start() for m in re.finditer(r'(?m)^BEGIN_CFG\s*$',text)]; ends=[m.start() for m in re.finditer(r'(?m)^END_CFG\s*$',text)]
    if len(starts)!=len(ends): raise ValueError('BEGIN/END mismatch')
    bs=[]; errs=[]
    for idx,start in enumerate(starts):
        stop=text.find('\n',ends[idx]); stop=len(text) if stop<0 else stop+1; raw=text[start:stop]; lines=raw.splitlines()
        try:
            si=next(i for i,l in enumerate(lines) if l.strip()=='Size'); n=int(lines[si+1].strip())
            ci=next(i for i,l in enumerate(lines) if l.strip()=='Supercell'); cell=np.array([[float(x) for x in lines[ci+j+1].split()[:3]] for j in range(3)])
            ai=next(i for i,l in enumerate(lines) if l.strip().startswith('AtomData:')); atoms=[]
            for l in lines[ai+1:]:
                s=l.strip()
                if s in ('Energy','PlusStress:','END_CFG') or s.startswith('Feature'): break
                t=l.split()
                if len(t)>=5 and t[0].lstrip('+-').isdigit(): atoms.append((int(t[0]),int(t[1]),np.array([float(x) for x in t[2:5]])))
            if len(atoms)!=n: raise ValueError(f'Size={n}, parsed={len(atoms)}')
            symbols=[TYPE_TO_ELEMENT[a[1]] for a in atoms]; pos=np.array([a[2] for a in atoms])
            if not np.isfinite(pos).all() or not np.isfinite(cell).all(): raise ValueError('NaN/Inf')
            energy=None; stress=None; features=[]
            for j,l in enumerate(lines):
                s=l.strip()
                if s=='Energy' and j+1<len(lines):
                    try: energy=float(lines[j+1].split()[0])
                    except: pass
                if s.startswith('PlusStress:') and j+1<len(lines):
                    try: stress=[float(x) for x in lines[j+1].split()]
                    except: pass
                if s.startswith('Feature'): features.append(s)
            a=Atoms(symbols=symbols,positions=pos,cell=cell,pbc=True); a.info['original_config_index']=idx
            if energy is not None: a.info['energy_metadata']=energy
            bs.append(dict(index=idx,raw=raw,n_atoms=n,cell=cell,atoms=atoms,symbols=symbols,positions=pos,metadata={'energy':energy,'stress':stress,'feature_lines':features},ase=a))
        except Exception as e: errs.append((idx,repr(e)))
    return bs,errs,len(starts),len(ends)
blocks,errors,n_begin,n_end=parse_mlip_cfg(INPUT_CFG)
print('BEGIN_CFG',n_begin,'END_CFG',n_end,'parsed',len(blocks),'errors',len(errors))
if errors: print(errors[:10]); raise RuntimeError('parser validation failed')
if not blocks: raise ValueError('Input CFG contains no configurations')
assert (n_begin,n_end,len(blocks))==(len(blocks),len(blocks),len(blocks))
assert all(set(b['symbols']).issubset(set(ELEMENTS)) for b in blocks)
# Limit SOAP species and all downstream species loops to elements present in
# this input. TYPE_TO_ELEMENT remains the supported CFG type mapping.
ELEMENTS=sorted({symbol for b in blocks for symbol in b['symbols']})
print('species counts first',Counter(blocks[0]['symbols']),'cell',blocks[0]['cell'],'pbc',blocks[0]['ase'].pbc)
for i in sorted(set([0,len(blocks)//2,len(blocks)-1])):
    b=blocks[i]; assert b['raw'].startswith('BEGIN_CFG') and b['raw'].rstrip().endswith('END_CFG')
    print('spot-check',i,len(b['ase']),b['symbols'][:5],b['positions'][:2],b['metadata'])


def run_case(bs,case_dir,name):
    case_dir=Path(case_dir); (case_dir/'plots').mkdir(parents=True,exist_ok=True); (case_dir/'soap_cache').mkdir(parents=True,exist_ok=True)
    fingerprint={'sha256':hashlib.sha256(INPUT_CFG.read_bytes()).hexdigest(),'params':SOAP_PARAMS,'elements':ELEMENTS,'ase':ase.__version__,'dscribe':DSCRIBE_VERSION,'n_configs':len(bs),'atom_counts':[b['n_atoms'] for b in bs]}
    soap=SOAP(species=ELEMENTS,**SOAP_PARAMS); dim=soap.get_number_of_features(); total_atoms=sum(b['n_atoms'] for b in bs); print(name,len(bs),'configs',len(bs),'atoms',total_atoms,'envs',dim,'SOAP features',total_atoms*dim*4/1024**2,'MiB')
    cp=case_dir/'soap_cache'/f'{name}.npz'; mp=case_dir/'soap_cache'/f'{name}.json'; good=False
    if cp.exists() and mp.exists():
        try: good=json.loads(mp.read_text())==fingerprint
        except: good=False
    if good: desc=[x.astype(np.float32) for x in np.load(cp,allow_pickle=True)['d']]; print('compatible cache reused')
    else:
        desc=[]
        for j,b in enumerate(bs):
            n_atoms=b['n_atoms']; x=np.asarray(soap.create(b['ase'],centers=np.arange(n_atoms),n_jobs=1),np.float32); x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12); assert x.shape==(n_atoms,dim); desc.append(x)
            if (j+1)%25==0 or j==len(bs)-1: print('SOAP',j+1,'/',len(bs))
        np.savez_compressed(cp,d=np.array(desc,dtype=object)); mp.write_text(json.dumps(fingerprint,indent=2))
    owners={s:[] for s in ELEMENTS}
    for ci,b in enumerate(bs):
        for ai,s in enumerate(b['symbols']): owners[s].append((desc[ci][ai],ci))
    def cross_configuration_nn(arr, owners, max_distance_entries=2_000_000):
        """Return exact nearest distances after excluding same-configuration points.

        Same-configuration environments must not define epsilon.  An arbitrary
        truncated neighbor list (such as n_neighbors=8) can hide the nearest
        valid cross-configuration environment, so search all candidates in
        bounded-memory chunks instead.
        """
        owners=np.asarray(owners)
        # The dot-product form avoids materializing a query-by-dataset-by-feature
        # tensor.  The chunk size is derived from the dataset size, rather than
        # from an arbitrary neighbor count, to keep memory bounded.
        chunk_size=max(1,min(1024,max_distance_entries//max(1,len(arr))))
        result=[]
        for start in range(0,len(arr),chunk_size):
            stop=min(start+chunk_size,len(arr)); q=arr[start:stop]
            distances=2.0-2.0*np.matmul(q,arr.T)
            distances=np.maximum(distances,0.0)
            distances[owners[start:stop,None]==owners[None,:]]=np.inf
            result.append(np.sqrt(np.min(distances,axis=1)))
        return np.concatenate(result) if result else np.empty(0,dtype=np.float32)

    nn_dist=[]; nn_rows=[]
    for s in ELEMENTS:
        arr=np.asarray([x[0] for x in owners[s]],np.float32); own=np.array([x[1] for x in owners[s]])
        if len(arr)<2 or len(np.unique(own))<2: continue
        cross_dist=cross_configuration_nn(arr,own)
        for owner,distance in zip(own,cross_dist):
            if np.isfinite(distance): nn_dist.append(float(distance)); nn_rows.append((int(owner),float(distance)))
    if not nn_dist:
        raise ValueError('At least two configurations sharing a chemical species are required for cross-configuration epsilon calculation')
    nn_dist=np.array(nn_dist); eps=float(np.quantile(nn_dist,.95)); print('NN quantiles',np.quantile(nn_dist,[.1,.5,.9,.95,.99]),'epsilon',eps)
    def savefig(name,xlabel,ylabel,title):
        plt.gca().set(xlabel=xlabel,ylabel=ylabel,title=title); plt.savefig(case_dir/'plots'/name,dpi=130,bbox_inches='tight'); plt.close()
    plt.figure(); plt.hist(nn_dist,bins=60); plt.axvline(eps,color='r'); savefig('soap_nn_histogram.png','distance','density',name+' SOAP NN histogram')
    ss=np.sort(nn_dist); plt.figure(); plt.plot(np.linspace(0,1,len(ss)),ss); plt.axhline(eps,color='r'); savefig('soap_nn_cdf.png','cumulative fraction','distance',name+' SOAP NN CDF')
    plt.figure(); plt.plot(ss); plt.axhline(eps,color='r'); savefig('soap_nn_sorted.png','sorted environment','distance',name+' sorted NN')
    def all_nov(candidates,bank):
        models={s:(NearestNeighbors(n_neighbors=1,metric='euclidean',n_jobs=-1).fit(bank[s]) if len(bank[s]) else None) for s in ELEMENTS}
        result={}
        for ci in candidates:
            out=np.full(bs[ci]['n_atoms'],np.inf)
            for s in ELEMENTS:
                ais=[a for a,x in enumerate(bs[ci]['symbols']) if x==s]
                if models[s] is not None: out[ais]=models[s].kneighbors(desc[ci][ais],return_distance=True)[0][:,0]
            result[ci]=out
        return result
    seed=int(np.argmax([max([x[1] for x in nn_rows if x[0]==ci],default=0) for ci in range(len(bs))]))
    selected=[seed]; bank={s:np.asarray([desc[seed][a] for a,x in enumerate(bs[seed]['symbols']) if x==s],np.float32) for s in ELEMENTS}
    while True:
        candidates=[ci for ci in range(len(bs)) if ci not in selected]
        novelties=all_nov(candidates,bank); best=max(((float(novelties[ci].max()),ci) for ci in candidates),default=(-1,None))
        if best[1] is None or best[0]<eps: break
        selected.append(best[1]); ci=best[1]
        for s in ELEMENTS:
            ais=[a for a,x in enumerate(bs[ci]['symbols']) if x==s]; bank[s]=np.vstack([bank[s],desc[ci][ais]])
        if len(selected)%10==0 or len(selected)==2: print('selected',len(selected),'last max',best[0])
    selected=sorted(selected); removed=[i for i in range(len(bs)) if i not in set(selected)]
    rows=[]; selected_set=set(selected)
    for ci,b in enumerate(bs):
        vals=[]; sp={}
        for s in ELEMENTS:
            ais=[a for a,x in enumerate(b['symbols']) if x==s]; other=[x for x in selected if x!=ci] if ci in selected_set else selected
            use=np.vstack([desc[cj][[a for a,x in enumerate(bs[cj]['symbols']) if x==s]] for cj in (other if len(other) else selected)])
            ds=NearestNeighbors(n_neighbors=1,metric='euclidean',n_jobs=-1).fit(use).kneighbors(desc[ci][ais],return_distance=True)[0][:,0]; vals+=ds.tolist(); sp[s]=ds
        v=np.array(vals); row={'original_config_index':ci,'selected':ci in selected_set,'max_local_novelty':float(v.max()),'mean_local_novelty':float(v.mean()),'median_local_novelty':float(np.median(v)),'p90_local_novelty':float(np.quantile(v,.9)),'p95_local_novelty':float(np.quantile(v,.95)),'p99_local_novelty':float(np.quantile(v,.99)),'fraction_above_epsilon':float(np.mean(v>=eps))}
        for s in ELEMENTS: row[s+'_max_novelty']=float(sp[s].max()); row[s+'_mean_novelty']=float(sp[s].mean())
        rows.append(row)
    scores=pd.DataFrame(rows)
    ep=[]
    for e in np.unique(np.clip(eps*np.array([.5,.75,.9,1,1.1,1.25,1.5]),0,np.sqrt(2))):
        ss=[seed]; bb={s:np.asarray([desc[seed][a] for a,x in enumerate(bs[seed]['symbols']) if x==s],np.float32) for s in ELEMENTS}
        while True:
            bv=-1; bi=None
            for ci in range(len(bs)):
                if ci in ss: continue
                pass
            candidates=[ci for ci in range(len(bs)) if ci not in ss]
            nall=all_nov(candidates,bb)
            if candidates:
                bi=max(candidates,key=lambda ci:float(nall[ci].max())); bv=float(nall[bi].max())
            if bi is None or bv<e: break
            ss.append(bi)
            for s in ELEMENTS:
                ais=[a for a,x in enumerate(bs[bi]['symbols']) if x==s]; bb[s]=np.vstack([bb[s],desc[bi][ais]])
        ep.append({'epsilon':float(e),'retained':len(ss),'removed':len(bs)-len(ss),'retention_percent':100*len(ss)/len(bs)})
    ep=pd.DataFrame(ep); ep.to_csv(case_dir/'epsilon_analysis.csv',index=False)
    for col in ['retained','retention_percent']:
        plt.figure(); plt.plot(ep.epsilon,ep[col],'o-'); savefig('epsilon_'+col+'.png','epsilon',col,name+' epsilon sensitivity')
    xy=PCA(2,random_state=RANDOM_SEED).fit_transform(np.array([x.mean(0) for x in desc])); plt.figure(); plt.scatter(xy[:,0],xy[:,1],s=10); savefig('pca_original.png','PC1','PC2',name+' SOAP PCA diagnostic only')
    plt.figure(); plt.scatter(xy[removed,0],xy[removed,1],s=10,label='removed'); plt.scatter(xy[selected,0],xy[selected,1],s=18,label='retained'); plt.legend(); savefig('pca_retained_removed.png','PC1','PC2',name+' retained/removed PCA diagnostic only')
    scores.to_csv(case_dir/'novelty_scores.csv',index=False)
    return {'desc':desc,'nn_dist':nn_dist,'epsilon':eps,'selected':selected,'removed':removed,'scores':scores,'eps':ep,'dim':dim}


def export_blocks(bs,inds,path):
    with open(path,'w') as f:
        for i in inds: f.write(bs[i]['raw'] if bs[i]['raw'].endswith('\n') else bs[i]['raw']+'\n')
def validate_output(path,source,inds):
    out,err,nb,ne=parse_mlip_cfg(path); assert not err and len(out)==len(inds)==nb==ne
    for x,i in zip(out,inds):
        y=source[i]; assert x['symbols']==y['symbols'] and x['n_atoms']==y['n_atoms']
        np.testing.assert_array_equal(x['positions'],y['positions']); np.testing.assert_array_equal(x['cell'],y['cell'])
    return len(out)
if TEST_MODE:
    test_blocks=blocks[:min(TEST_N_CONFIGS,len(blocks))]
    testdir=OUTPUT_DIR/'test_run'; test=run_case(test_blocks,testdir,'test')
    export_blocks(test_blocks,test['selected'],testdir/'filtered_static_dataset.cfg'); export_blocks(test_blocks,test['removed'],testdir/'removed_static_dataset.cfg')
    assert validate_output(testdir/'filtered_static_dataset.cfg',test_blocks,test['selected'])==len(test['selected'])
    assert validate_output(testdir/'removed_static_dataset.cfg',test_blocks,test['removed'])==len(test['removed'])
    assert np.linalg.norm(test['desc'][0]-test['desc'][0])==0
    print('TEST_MODE PASSED: parser, SOAP self-distance, NN, FPS, epsilon, plots, CFG round-trip')
TEST_MODE=False
full=run_case(blocks,OUTPUT_DIR,'full')
sel=full['selected']; rem=full['removed']
export_blocks(blocks,sel,OUTPUT_DIR/'filtered_static_dataset.cfg'); export_blocks(blocks,rem,OUTPUT_DIR/'removed_static_dataset.cfg')
assert validate_output(OUTPUT_DIR/'filtered_static_dataset.cfg',blocks,sel)==len(sel)
assert validate_output(OUTPUT_DIR/'removed_static_dataset.cfg',blocks,rem)==len(rem)
scores=full['scores']; scores.to_csv(OUTPUT_DIR/'novelty_scores.csv',index=False)
pd.DataFrame({'original_config_index':sel}).to_csv(OUTPUT_DIR/'selected_indices.csv',index=False)
pd.DataFrame({'original_config_index':rem}).to_csv(OUTPUT_DIR/'removed_indices.csv',index=False)
pd.DataFrame([{'original_config_index':i,'selected':i in set(sel),'temperature':np.nan,'energy_metadata':b['metadata']['energy'],'n_atoms':b['n_atoms'],'species_counts':json.dumps(Counter(b['symbols']),sort_keys=True),'pbc':'111'} for i,b in enumerate(blocks)]).to_csv(OUTPUT_DIR/'configuration_metadata.csv',index=False)
pd.DataFrame([{'temperature':'missing','original':len(blocks),'retained':len(sel),'removed':len(rem),'retention_percent':100*len(sel)/len(blocks)}]).to_csv(OUTPUT_DIR/'temperature_statistics.csv',index=False)
full['eps'].to_csv(OUTPUT_DIR/'epsilon_analysis.csv',index=False)
plt.figure(); plt.hist(full['nn_dist'],bins=60,density=True,alpha=.7,label='original local NN'); plt.hist(scores['mean_local_novelty'],bins=40,density=True,alpha=.6,label='config mean novelty'); plt.legend(); plt.xlabel('SOAP distance'); plt.ylabel('density'); plt.title('Before/after structural coverage'); plt.savefig(OUTPUT_DIR/'plots'/'before_after_nn_distributions.png',dpi=130,bbox_inches='tight'); plt.close()
plt.figure(); scores['max_local_novelty'].hist(bins=40); plt.axvline(full['epsilon'],color='r',label='epsilon'); plt.legend(); plt.xlabel('maximum local novelty'); plt.ylabel('count'); plt.savefig(OUTPUT_DIR/'plots'/'novelty_distribution.png',dpi=130,bbox_inches='tight'); plt.close()
plt.figure(); plt.bar(ELEMENTS,[scores[s+'_mean_novelty'].mean() for s in ELEMENTS]); plt.xlabel('central species'); plt.ylabel('mean nearest SOAP distance'); plt.savefig(OUTPUT_DIR/'plots'/'species_specific_novelty.png',dpi=130,bbox_inches='tight'); plt.close()
params={'input_file':str(INPUT_CFG.resolve()),'input_sha256':hashlib.sha256(INPUT_CFG.read_bytes()).hexdigest(),'n_configurations':len(blocks),'atom_counts':[b['n_atoms'] for b in blocks],'species':ELEMENTS,'type_to_element':TYPE_TO_ELEMENT,'soap_params':SOAP_PARAMS,'distance_definition':'Euclidean distance between unit-normalized local SOAP vectors; same central species only','epsilon':float(full['epsilon']),'epsilon_method':'95th percentile cross-configuration local NN distances','random_seed':RANDOM_SEED,'energy_used_for_selection':False,'pca_used_for_selection':False,'temperature_metadata_present':False,'software':{'python':sys.version,'ase':ase.__version__,'dscribe':DSCRIBE_VERSION,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'run_utc':datetime.now(timezone.utc).isoformat()}
(OUTPUT_DIR/'selection_parameters.json').write_text(json.dumps(params,indent=2,default=str))
(OUTPUT_DIR/'README.txt').write_text(f'Purpose: local structural redundancy removal for later MTP training.\nInput: {INPUT_CFG.name}, {len(blocks)} configurations, per-configuration atom counts recorded in configuration_metadata.csv.\nSOAP: periodic DScribe SOAP parameters {SOAP_PARAMS}; local atom descriptors; same central species only.\nDistance: normalized SOAP Euclidean distance, equivalent to sqrt(2-2*k), range [0,sqrt(2)].\nEpsilon: {full["epsilon"]:.8f}, data-derived 95th percentile of cross-configuration local nearest-neighbour distances; not universal.\nSelection: global species-aware greedy FPS; no temperature balancing. Energy is metadata only and was excluded. PCA is diagnostic only. Rare/high-novelty environments are reported, not auto-deleted. Filtering does not guarantee better MTP accuracy or MD stability; retain active learning and extrapolation validation.\n')
report=f'''STRUCTURAL REDUNDANCY FILTERING — FINAL REPORT
Input configurations: {len(blocks)}
Retained configurations: {len(sel)}
Removed configurations: {len(rem)}
Retention percentage: {100*len(sel)/len(blocks):.3f}
Atom counts: per-configuration values are preserved from the input CFG
Species: {ELEMENTS}
SOAP parameters: {SOAP_PARAMS}
Selected epsilon: {full["epsilon"]:.8f}
Temperature metadata: not available in input CFG; no temperature-based filtering performed
Energy used for selection: NO
Validation: custom parser round-trip passed for filtered ({len(sel)}) and removed ({len(rem)}) CFG files; coordinates/cells/species unchanged
Output directory: {OUTPUT_DIR.resolve()}
The filtering criterion was based on structural/local atomic-environment similarity only. Energy was not used for selection.
'''
(OUTPUT_DIR/'validation_report.txt').write_text(report); print(report)

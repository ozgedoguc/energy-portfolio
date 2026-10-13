"""Generate a fresh illustrative dataset and compute every reported result.

Run with Python 3 and NumPy: python synthetic_experiment.py
Inputs are generated before selection; no historical manuscript results are targets.
"""
from pathlib import Path
import json
import platform
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'synthetic_reproduction'
OUT.mkdir(exist_ok=True)
SEED = 20261013
rng = np.random.default_rng(SEED)
technology = ['Solar','Battery','Efficiency','Wind','PPA','Demand response',
              'Solar','Battery','Efficiency','Wind','PPA','Demand response',
              'Solar','Battery','Efficiency']
countries = ['DE','FR','IT','ES','DE','IT','FR','DE','ES','DE','FR','IT','DE','ES','FR']
projects = []
for i, (tech,country) in enumerate(zip(technology,countries)):
    cost = int(rng.integers(25,91))
    projects.append(dict(project_id=f'S{i+1:02d}', technology=tech,
                         jurisdiction=country, commitment_years=int(rng.integers(2,16)),
                         capex=cost, npv=round(cost*float(rng.uniform(.4,.85)),2),
                         carbon=round(cost*float(rng.uniform(.10,.25)),2),
                         resilience=round(cost*float(rng.uniform(.05,.12)),2)))
inputs = np.array([[p[k] for k in ['capex','npv','carbon','resilience']] for p in projects])
masks = np.arange(1<<15,dtype=np.int32)
bits = ((masks[:,None] >> np.arange(15)) & 1).astype(np.float64)
totals = bits @ inputs
budget, carbon_floor, resilience_floor = 300.,42.,13.
excluded = np.array([p['jurisdiction']=='DE' and p['commitment_years']>3 for p in projects])
weight=np.array([.5,.3,.2])
equal=np.array([1/3]*3)

def feasible(t, floor=carbon_floor, institution=False):
    f=(t[:,0]<=budget+1e-9)&(t[:,2]>=floor-1e-9)&(t[:,3]>=resilience_floor-1e-9)
    if institution:
        f &= (bits[:,excluded].sum(axis=1)==0)
    return f

base_feasible=feasible(totals)
assert base_feasible.any()
lower=totals[base_feasible,1:].min(axis=0)
upper=totals[base_feasible,1:].max(axis=0)
assert np.all(upper>lower)

def scores(t):
    # Bounds remain fixed across shocks and institutional constraint changes.
    return (t[:,1:]-lower)/(upper-lower)

base_scores=scores(totals)

def select(f, sc, criterion, w=weight):
    ix=np.flatnonzero(f)
    primary=sc[ix].min(axis=1) if criterion=='max_min' else sc[ix]@w
    # Descending primary objective, mean satisfaction, NPV score; then lowest mask.
    order=np.lexsort((ix,-sc[ix,0],-sc[ix].mean(axis=1),-primary))
    return int(ix[order[0]])

def pareto_indices(t,f):
    ix=np.flatnonzero(f)
    nondominated=[]
    for j in ix:
        others=t[ix,1:]
        dom=np.all(others>=t[j,1:]-1e-9,axis=1)&np.any(others>t[j,1:]+1e-9,axis=1)
        if not dom.any(): nondominated.append(int(j))
    return np.array(nondominated,dtype=int)

pareto=pareto_indices(totals,base_feasible)
def record(mask,t=totals,sc=base_scores,f=base_feasible):
    pp=pareto if t is totals and f is base_feasible else pareto_indices(t,f)
    return dict(mask=mask, project_ids=[p['project_id'] for i,p in enumerate(projects) if bits[mask,i]],
                capex=float(t[mask,0]), npv=float(t[mask,1]), carbon=float(t[mask,2]),
                resilience=float(t[mask,3]), satisfaction=sc[mask].tolist(),
                minimum_satisfaction=float(sc[mask].min()),mean_satisfaction=float(sc[mask].mean()),
                pareto_distance=float(np.linalg.norm(sc[pp]-sc[mask],axis=1).min()),
                institution_exclusions_selected=int(bits[mask,excluded].sum()))

chosen={name:select(base_feasible,base_scores,rule,w) for name,rule,w in
        [('max_min','max_min',weight),('weighted','weighted',weight),('equal_weight','weighted',equal)]}
static={name:record(m) for name,m in chosen.items()}
assert all(r['pareto_distance']<1e-9 for r in static.values())
institution_f=feasible(totals,institution=True)
assert institution_f.any(), 'Institutional scenario infeasible; report infeasibility instead of retuning seed.'
institution={name:record(select(institution_f,base_scores,rule),f=institution_f)
             for name,rule in [('max_min','max_min'),('weighted','weighted')]}

def scan_weights(ws):
    ix=np.flatnonzero(base_feasible)
    vals=[]
    for w in ws:
        m=select(base_feasible,base_scores,'weighted',w)
        vals.append(float(base_scores[m].min()))
    return np.array(vals)

named_weights={'Reference':[.5,.3,.2],'Equal':[1/3]*3,'Financial':[.7,.2,.1],
               'Carbon':[.2,.7,.1],'Resilience':[.2,.1,.7]}
wrng=np.random.default_rng(SEED+1)
random_weights=wrng.dirichlet(np.ones(3),size=1000)
grid=np.array([[i/100,j/100,(100-i-j)/100] for i in range(101) for j in range(101-i)])
random_min=scan_weights(random_weights)
grid_min=scan_weights(grid)
search=dict(named={n:dict(weights=w,**record(select(base_feasible,base_scores,'weighted',np.array(w)))) for n,w in named_weights.items()},
            random=dict(count=1000,seed=SEED+1,distribution='Dirichlet(1,1,1)',
                        mean=float(random_min.mean()),p05=float(np.quantile(random_min,.05)),
                        p95=float(np.quantile(random_min,.95)),maximum=float(random_min.max())),
            grid=dict(count=len(grid),step=.01,maximum=float(grid_min.max())))

scenarios=[]
for label, md, wd in [('Battery cost decline',2,12),('Carbon target tightening',3,18),
                      ('PPA basis shock',2,14),('Grid reliability shock',3,15)]:
    a=inputs.copy(); floor=carbon_floor
    if label.startswith('Battery'):
        ids=[i for i,p in enumerate(projects) if p['technology']=='Battery']; a[ids,0]*=.75
        desc='Reduce battery CapEx by 25%; hold NPV and other attributes fixed.'
    elif label.startswith('Carbon'):
        ids=[];floor=50.;desc='Increase portfolio carbon floor from 42 to 50.'
    elif label.startswith('PPA'):
        ids=[i for i,p in enumerate(projects) if p['technology']=='PPA'];a[ids,1]*=.70
        desc='Reduce PPA NPV by 30%; hold all other attributes fixed.'
    else:
        ids=[i for i,p in enumerate(projects) if p['technology'] in ['Solar','Wind','PPA']];a[ids,3]*=.70
        desc='Reduce resilience of Solar, Wind and PPA projects by 30%; hold other attributes fixed.'
    t=bits@a;sc=scores(t);f=feasible(t,floor)
    assert f.any()
    opt=select(f,sc,'weighted');u=sc@weight
    penalty=np.maximum(t[:,0]-budget,0)/budget + np.maximum(floor-t[:,2],0)/floor + np.maximum(resilience_floor-t[:,3],0)/resilience_floor
    regrets={name:float(max(0.,u[opt]-u[m]+penalty[m])) for name,m in chosen.items()}
    loss={name:regrets[name]*(wd if name=='weighted' else md) for name in chosen}
    sensitivity={str(lam):{name:float(max(0.,u[opt]-u[m]+lam*penalty[m])) for name,m in chosen.items()} for lam in [0,.25,.5,.75,1]}
    scenarios.append(dict(name=label, transformation=desc, affected_project_ids=[projects[i]['project_id'] for i in ids],
                          max_min_days=md,weighted_days=wd,carbon_floor=floor,reference_optimum_ids=record(opt,t,sc,f)['project_ids'],
                          daily_regret=regrets,cumulative_loss=loss,penalty_sensitivity_regrets=sensitivity))
reference_loss={n:sum(s['cumulative_loss'][n] for s in scenarios) for n in chosen}
equal_delay={n:sum(s['daily_regret'][n]*s['max_min_days'] for s in scenarios) for n in chosen}
long_delay={n:sum(s['daily_regret'][n]*30 for s in scenarios) for n in chosen}
qstar=equal_delay['max_min']/equal_delay['weighted'] if equal_delay['weighted']>1e-12 else None
sensitivity={str(lam):{n:sum(s['penalty_sensitivity_regrets'][str(lam)][n]*(s['weighted_days'] if n=='weighted' else s['max_min_days']) for s in scenarios) for n in chosen} for lam in [0,.25,.5,.75,1]}

dataset=dict(provenance='Fresh synthetic scenario generated for the manuscript revision. Not a recovery of the original data and not fitted to prior published aggregates.',
             seed=SEED,generator='numpy.random.default_rng; PCG64',
             generation='Fixed technology/country lists. Per row draw CapEx as integer 25..90 inclusive, duration as integer 2..15 inclusive, then independent uniform multipliers for NPV (0.40..0.85), carbon (0.10..0.25), resilience (0.05..0.12). Round derived attributes to two decimal places.',
             units={'capex':'synthetic capital units (SCU)','npv':'SCU, assumed already discounted; no cash-flow model',
                    'carbon':'synthetic carbon-reduction score units; not measured tonnes',
                    'resilience':'synthetic resilience score units; not measured reliability'},
             assumptions='Binary independent projects, additive attributes, no dependencies, synergies, cash-flow model, or jurisdictional economic effects. Countries and durations are illustrative labels used only for the commitment boundary.',projects=projects)
results=dict(seed=SEED,python=platform.python_version(),numpy=np.__version__,
             portfolio_count=len(masks),feasible_count=int(base_feasible.sum()),pareto_count=len(pareto),
             institutional_feasible_count=int(institution_f.sum()),excluded_project_ids=[p['project_id'] for i,p in enumerate(projects) if excluded[i]],
             constraints=dict(budget=budget,carbon_floor=carbon_floor,resilience_floor=resilience_floor),
             normalization=dict(lower=lower.tolist(),upper=upper.tolist(),rule='s_k=(objective_k-lower_k)/(upper_k-lower_k). Bounds from initial feasible portfolios, fixed thereafter, no clipping.'),
             selection_tie_break='Highest primary objective, highest mean satisfaction, highest financial satisfaction, then lowest integer portfolio mask.',
             regret_definition='max(0, U_s_star - U_s_stale + lambda * V_s_stale), U=0.5*s_financial+0.3*s_carbon+0.2*s_resilience; V=sum normalized budget excess, carbon shortfall, resilience shortfall. lambda=1. Each shock is independently applied to the initial dataset.',
             static=static,institution=institution,weight_search=search,scenarios=scenarios,
             reference_loss=reference_loss,equal_delay_loss=equal_delay,thirty_day_loss=long_delay,
             break_even_q=qstar,penalty_sensitivity=sensitivity,
             checks={'binary_portfolios_enumerated':32768,'all_static_selected_portfolios_pareto_efficient':True,
                     'post_constraint_portfolios_exclude_long_german_commitments':all(r['institution_exclusions_selected']==0 for r in institution.values()),
                     'max_min_objective_at_least_each_weighted_search_minimum':bool(static['max_min']['minimum_satisfaction']+1e-9>=max(random_min.max(),grid_min.max()))})
(OUT/'synthetic_projects.json').write_text(json.dumps(dataset,indent=2)+'\n')
(OUT/'computed_results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({k:results[k] for k in ['feasible_count','pareto_count','excluded_project_ids','normalization','static','institution','reference_loss','equal_delay_loss','thirty_day_loss','break_even_q','checks']},indent=2))

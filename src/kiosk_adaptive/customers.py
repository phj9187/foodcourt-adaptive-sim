"""Synthetic heterogeneous customers; assumptions, not fitted Korean coefficients.

Stable profiles and potential-visit randomness are shared across experimental
arms. Experiences and actual attendance evolve separately in each arm.
"""
from dataclasses import dataclass, field
import numpy as np
from .models import rng_for, CATEGORIES

WORLDS = ('mixed_logit', 'satisficing', 'lexicographic')


@dataclass
class CustomerMemory:
    experiences: dict = field(default_factory=dict)

    def observe(self, customer, experience, category, strength):
        previous = self.experiences.get(customer, (.5, None))
        self.experiences[customer] = ((1-strength)*previous[0]+strength*experience,
                                      category or previous[1])


def generate_customers(settings, court, day, namespace, count, expected, memory):
    """One row per potential visit; repeat IDs have stable, correlated traits.

    Population budgets are fixed in court provenance when first generated, not
    recomputed from an intervention's menu prices. No objective enters here.
    """
    s=settings
    rng=rng_for(s.seed,court.id,namespace,day,'customer_opportunities')
    panel=max(count,int(expected*3)+1)
    regular_ids=rng.permutation(panel)[:count]
    repeat=rng.random(count)<s.repeat_share
    visit_uniform=rng.random(count)
    result=[]
    low,high=court.provenance.get('customer_budget_bounds',[s.budget_low,s.budget_high])
    for i in range(count):
        cid=f'regular_{regular_ids[i]}' if repeat[i] else f'new_{day}_{i}'
        profile=rng_for(s.seed,court.id,namespace,'profile',cid)
        economy,urgency=profile.normal(size=2)
        budget=low+(high-low)/(1+np.exp(-economy))
        # Joint synthetic distribution: a tight budget tends to raise price
        # sensitivity; urgency raises waiting sensitivity and shortens time.
        ps=float(np.clip(.7-.22*economy+profile.normal(0,.12),.1,1.4))
        ws=float(np.clip(.7+.24*urgency+profile.normal(0,.12),.1,1.4))
        tastes=np.clip(profile.normal(.5,.23,3),0,1)
        tastes=np.clip(tastes+rng_for(s.seed,court.id,namespace,day,cid,'daily_taste').normal(0,.05,3),0,1)
        gap=(40. if s.venue_type=='campus' else 100.)*np.exp(-.22*urgency)
        gap=float(np.clip(gap,12.,180.))
        patience=float(profile.lognormal(np.log(s.patience_minutes)-.3*urgency,.45))
        taste_weight=float(profile.lognormal(np.log(2.5),.2))
        previous,habit=memory.experiences.get(cid,(.5,None))
        visit_probability=float(np.clip(s.regular_visit_probability+s.memory_strength*(previous-.5),.05,1.)) if repeat[i] else 1.
        input_noise=rng_for(s.seed,court.id,namespace,day,cid,'input_noise')
        measured=np.clip(tastes+input_noise.normal(0,s.input_noise,3),0,1)
        ordered_worlds=[s.behaviour_world]+[w for w in WORLDS if w!=s.behaviour_world]
        choice_world=str(profile.choice(ordered_worlds,p=[s.dominant_choice_share,(1-s.dominant_choice_share)/2,(1-s.dominant_choice_share)/2]))
        # Kiosk-enterable inputs only. A separate stream keeps all earlier
        # behavioural draws unchanged. Budget is entered as a whole-1000 KRW
        # ceiling; hurry is a noisy 5-level self-report, never the hidden gap.
        extra=rng_for(s.seed,court.id,namespace,day,cid,'kiosk_inputs')
        hurry=float(np.clip(1/(1+np.exp(-urgency))+extra.normal(0,s.input_noise),0,1))
        taste_input=np.round(measured*4)/4
        price_input=round(float(np.clip((ps-.1)/1.3,0,1))*4)/4
        wait_input=round(float(np.clip((ws-.1)/1.3,0,1))*4)/4
        hurry_input=round(hurry*4)/4
        missing=bool(extra.random()<s.input_missing_probability)
        if missing:
            # Skipped preference questions: neutral values, hard constraints still apply.
            taste_input=np.full(3,.5);price_input=wait_input=hurry_input=.5
        result.append(dict(id=cid,repeat=bool(repeat[i]),visits=bool(visit_uniform[i]<visit_probability),
            choice_world=choice_world,
            budget=float(budget),ps=ps,ws=ws,tastes=tastes,taste_input=taste_input,
            price_input=price_input,wait_input=wait_input,hurry_input=hurry_input,
            budget_input=float(np.floor(budget/1000.)*1000.),inputs_missing=missing,
            patience=patience,class_gap=gap,taste_weight=taste_weight,
            allergy=bool(profile.random()<.05),dislikes=bool(profile.random()<.1),
            excluded=int(profile.integers(len(CATEGORIES))),habit=habit))
    return result


def choose_menu(considered, utility, shocks, prices, waits, fits, world, threshold):
    """Alternative plausible choice laws, kept separate from reward weights."""
    if world=='mixed_logit':
        return max(considered,key=lambda j:utility[j]+shocks[j]),list(considered)
    if world=='satisficing':
        seen=[]
        for j in considered:
            seen.append(j)
            if utility[j]+shocks[j]>=threshold:return j,seen
        return max(seen,key=lambda j:utility[j]+shocks[j]),seen
    if world=='lexicographic':
        # The caller supplies an urgency-dependent primary criterion. Near ties
        # use taste, then price; no composite recommendation score is imposed.
        if threshold<0:key=lambda j:(-round(waits[j]/3),fits[j],-prices[j])
        elif threshold==0:key=lambda j:(-round(prices[j]/500),fits[j],-waits[j])
        else:key=lambda j:(round(fits[j]*5),-prices[j],-waits[j])
        return max(considered,key=key),list(considered)
    raise ValueError('unknown behaviour world')

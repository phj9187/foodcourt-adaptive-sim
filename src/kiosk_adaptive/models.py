from dataclasses import dataclass,field,asdict
from pathlib import Path
import hashlib,json,math
import numpy as np

CATEGORIES=('분식','중식','일식','양식','한식','동남아식')
POLICIES=('none','neural')

def seed_for(seed,*parts):return int.from_bytes(hashlib.sha256(repr((seed,*parts)).encode()).digest()[:8],'big')
def rng_for(seed,*parts):return np.random.default_rng(seed_for(seed,*parts))
def data_file(name):return Path(__file__).parent/'data'/name

def softmax(x):
 x=np.asarray(x,dtype=float);v=np.exp(np.clip(x-x.max(),-60,0));return v/v.sum()

@dataclass(frozen=True)
class Dish:
 id:str
 name:str
 counter:int
 category:str
 price:float
 recipe:dict
 taste:tuple
 cook_minutes:float
 popularity:float
 vegetarian:bool
 allergens:tuple
 source_id:str
 observed_name:bool
 price_status:str

@dataclass
class Court:
 id:str
 dishes:list
 servers:list
 kiosks:int
 seats:int
 duration:float
 initial_stock:list
 ingredient_costs:dict
 provenance:dict

@dataclass
class Settings:
 seed:int=20261105
 duration:float=180.
 kiosks:int=2
 servers_per_counter:int=2
 seats:int=200
 attention_limit:int=6
 recommendation_count:int=1
 decision_base:float=.4
 per_item_minutes:float=.04
 recommendation_overhead:float=.12
 payment_minutes:float=.25
 patience_minutes:float=15.
 taste_noise:float=.2
 choice_noise:float=.12
 openness:float=.65
 outside_utility:float=.15
 budget_low:float=4500.
 budget_high:float=11000.
 demand_cv:float=.15
 cook_cv:float=.25
 stock_cover:float=1.1
 expected_purchase_rate:float=.75
 wait_scale:float=20.
 profit_scale:float=6000.
 venue_types:tuple=('campus',)
 venue_samples:int=24
 venue_lhs_candidates:int=24
 dining_minutes:float=20.
 peak_fraction:float=.65
 venue_type:str='campus'
 def validate(self):
  if self.duration<=0 or self.stock_cover<=0 or self.wait_scale<=0:raise ValueError('positive operating scales required')
  if self.servers_per_counter<1 or self.kiosks<1 or self.seats<0:raise ValueError('invalid capacity')
  if not 0<=self.openness<=1:raise ValueError('invalid probability')
  return self
 def to_dict(self):return asdict(self)


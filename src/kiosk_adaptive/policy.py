"""Direct-menu PPO with shared candidate encoder and invariant value pooling."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

FEATURES = 49
# Named positions in the candidate vector built by environment.simulate.
INDEX = dict(taste_input=slice(0, 3), budget_input=3, price_input=4, wait_input=5, hurry_input=6,
             dish_taste=slice(7, 10), price=10, cook=11, margin=12, eta=13, stock_fraction=14,
             popularity=-2, position=-1)


class ActorCritic(nn.Module):
    """prior: optional RuleController. With a prior the policy is residual:
    logits = prior_scale * standardized rule score + learned correction, the
    correction starts at zero, so the untrained policy already ranks menus by
    the known objective formula and PPO only learns what the formula misses."""
    def __init__(self, hidden=48, prior=None, prior_scale=3.):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(FEATURES, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden*2, 1)
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.prior = prior
        if prior is not None:
            self.prior_scale = nn.Parameter(torch.tensor(float(prior_scale)))
            nn.init.zeros_(self.actor.weight); nn.init.zeros_(self.actor.bias)

    def forward(self, candidates):
        z = self.encoder(candidates)
        pooled = z.mean(0)
        logits = self.actor(torch.cat((z, pooled.expand_as(z)), -1)).squeeze(-1)
        if self.prior is not None:
            with torch.no_grad():
                score = self.prior.score(candidates).float()
                score = (score-score.mean())/(score.std(unbiased=False)+1e-6) if len(score)>1 else score*0
            logits = logits+self.prior_scale*score
        return Categorical(logits=logits), self.critic(pooled).squeeze(-1)


class Collector:
    def __init__(self, model, deterministic=False):
        self.model, self.deterministic, self.samples = model, deterministic, []

    def __call__(self, candidates, rng):
        x = torch.as_tensor(candidates)
        with torch.no_grad():
            dist, value = self.model(x)
            probs = dist.probs.cpu().numpy().astype(float)
            probs /= probs.sum()
            action = int(probs.argmax()) if self.deterministic else int(rng.choice(len(probs), p=probs))
            self.samples.append(dict(x=x, action=action, logp=float(dist.log_prob(torch.tensor(action))),
                                     value=float(value), reward=0.))
        return action


class RuleController:
    """Deterministic non-learning recommender over the same observable candidate features.

    Used as an upper-bound style control: a learned policy that cannot beat the
    best simple rule has not learned anything the rule did not already know.
    """
    def __init__(self, kind, objective=None, wait_scale=20., profit_scale=6000.):
        if kind not in ('rule_fit', 'rule_fast', 'rule_margin', 'rule_objective'):
            raise ValueError('unknown rule')
        if kind == 'rule_objective' and objective is None:
            raise ValueError('objective rule needs an objective')
        self.kind, self.objective, self.wait_scale = kind, objective, wait_scale

    def score(self, x):
        fit = 1-(x[:, INDEX['taste_input']]-x[:, INDEX['dish_taste']]).abs().mean(1)
        eta = x[:, INDEX['eta']]*self.wait_scale
        if self.kind == 'rule_fit': return fit
        if self.kind == 'rule_fast': return -eta
        if self.kind == 'rule_margin': return x[:, INDEX['margin']]
        o = self.objective
        budget = (x[:, INDEX['budget_input']]*11000.).clamp(min=1.)
        afford = (1-x[:, INDEX['price']]*11000./budget).clamp(min=0.)
        wait = self.wait_scale/(self.wait_scale+eta)
        customer = o.taste*fit+o.affordability*afford+o.waiting*wait
        return o.customer_share*customer+(1-o.customer_share)*o.profit_share*x[:, INDEX['margin']]

    def __call__(self, candidates):
        v = self.score(candidates)
        return Categorical(logits=(v-v.max())*1e4), torch.tensor(0.)


def targets(samples, gae_lambda=None):
    """Critic targets and raw advantages; gae_lambda=1 equals the Monte Carlo return."""
    values = torch.tensor([s['value'] for s in samples], dtype=torch.float32)
    if gae_lambda is None:
        # Legacy: undiscounted Monte Carlo return over the whole block.
        returns, cumulative = [], 0.
        for sample in reversed(samples):
            cumulative += sample['reward']; returns.append(cumulative)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
        advantages = returns-values
    else:
        # GAE(lambda) with gamma=1 over the finite block; terminal value 0.
        # The critic bootstraps the effect of each decision on later customers
        # (queues, stock, return visits) instead of sharing one day-level return.
        advantages = torch.zeros(len(samples)); running = 0.
        for t in reversed(range(len(samples))):
            following = float(values[t+1]) if t+1 < len(samples) else 0.
            delta = samples[t]['reward']+following-float(values[t])
            running = delta+gae_lambda*running; advantages[t] = running
        returns = advantages+values
    return returns, advantages


def update_ppo(model, optimizer, samples, epochs=4, clip=.2, entropy=.01, batch_size=64, gae_lambda=None):
    if not samples:
        return dict(samples=0)
    returns, advantages = targets(samples, gae_lambda)
    if len(samples)>1:
        advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    losses = []
    for _ in range(epochs):
        for indices in torch.randperm(len(samples)).split(batch_size):
            terms = []
            for i in indices.tolist():
                s = samples[i]; dist, value = model(s['x'])
                ratio = (dist.log_prob(torch.tensor(s['action']))-s['logp']).exp()
                surrogate = torch.minimum(ratio*advantages[i], ratio.clamp(1-clip,1+clip)*advantages[i])
                terms.append(-surrogate+.5*(value-returns[i]).square()-entropy*dist.entropy())
            loss = torch.stack(terms).mean()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step(); losses.append(float(loss.detach()))
    return dict(samples=len(samples), loss=float(np.mean(losses)))

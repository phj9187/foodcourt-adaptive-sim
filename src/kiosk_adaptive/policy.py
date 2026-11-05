"""Direct-menu PPO with shared candidate encoder and invariant value pooling."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

FEATURES = 47


class ActorCritic(nn.Module):
    def __init__(self, hidden=48):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(FEATURES, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden*2, 1)
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, candidates):
        z = self.encoder(candidates)
        pooled = z.mean(0)
        logits = self.actor(torch.cat((z, pooled.expand_as(z)), -1)).squeeze(-1)
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


def update_ppo(model, optimizer, samples, epochs=4, clip=.2, entropy=.01, batch_size=64):
    if not samples:
        return dict(samples=0)
    # Undiscounted finite multi-day return, terminal reward at each day's final
    # decision. Earlier decisions receive all later rewards in the same block.
    returns, cumulative = [], 0.
    for sample in reversed(samples):
        cumulative += sample['reward']; returns.append(cumulative)
    returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
    advantages = returns-torch.tensor([s['value'] for s in samples])
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

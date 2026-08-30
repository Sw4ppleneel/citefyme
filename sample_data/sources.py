from citefyme.ingest import build_source
from citefyme.models import Source

_DREAMER = """
## 1. Introduction
Dreamer learns a latent dynamics model from past experience that predicts
outcomes for imagined action sequences. This allows the agent to plan by
imagination rather than acting in the real environment for every rollout.

## 3.2 Latent Dynamics
The world model compresses high dimensional observations into compact
latent states and predicts forward transitions in latent space. Rewards
and values are also predicted directly from the latent state, avoiding
the need to decode back to pixels during planning.
"""

_DREAMERV3 = """
## 4. Experiments
DreamerV3 solves a wide range of tasks across multiple domains using a
single fixed set of hyperparameters, without domain-specific tuning. This
includes both continuous and discrete control tasks evaluated across more
than twenty benchmark suites.

## 6. Limitations
Performance degrades on tasks requiring planning over very long horizons
under distribution shift from the training environment. The latent model
accumulates compounding prediction error as the imagined rollout lengthens.
"""

_LONGHORIZON_EVAL = """
## 5. Findings
Across the surveyed benchmarks, only a small number of studies evaluate
planning beyond 100 steps under out-of-distribution dynamics, leaving
long-horizon robustness largely underexplored. Most reported results are
limited to in-distribution short-horizon tasks under 50 steps.
"""

_VJEPA2 = """
## 2. Architecture
V-JEPA 2 learns representations by predicting masked regions of video in
latent space rather than predicting raw pixels, avoiding pretraining
objectives that waste capacity on unpredictable low-level texture.

## 5. Planning Results
When paired with a lightweight planner, V-JEPA 2 achieves competitive
performance on short-horizon robotic manipulation tasks without any
action-labeled pretraining data for the manipulation domain itself.
"""


def get_sample_sources() -> list[Source]:
    docs = [
        ("paper_dreamer", "Dream to Control: Learning Behaviors by Latent Imagination",
         ["Hafner et al."], 2020, _DREAMER),
        ("paper_dreamerv3", "Mastering Diverse Domains through World Models",
         ["Hafner et al."], 2023, _DREAMERV3),
        ("paper_longhorizon_eval", "Evaluating Long-Horizon Planning in Learned World Models",
         ["Chen et al."], 2025, _LONGHORIZON_EVAL),
        ("paper_vjepa2", "V-JEPA 2: Self-Supervised Video Representations for Planning",
         ["Assran et al."], 2025, _VJEPA2),
    ]
    return [build_source(sid, title, authors, year, text) for sid, title, authors, year, text in docs]

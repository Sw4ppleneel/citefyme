from pydantic import BaseModel


class EvalQuestion(BaseModel):
    question: str
    relevant_chunk_ids: list[str] = []
    expected_keywords: list[str] = []
    answerable: bool = True


GOLD_QUESTIONS: list[EvalQuestion] = [
    EvalQuestion(
        question="How does Dreamer plan using imagined trajectories?",
        relevant_chunk_ids=["paper_dreamer_c0"],
        expected_keywords=["latent", "imagin"],
    ),
    EvalQuestion(
        question="How does the world model represent observations in latent space?",
        relevant_chunk_ids=["paper_dreamer_c1"],
        expected_keywords=["compress", "latent"],
    ),
    EvalQuestion(
        question="Does DreamerV3 need domain-specific hyperparameter tuning?",
        relevant_chunk_ids=["paper_dreamerv3_c0"],
        expected_keywords=["fixed", "hyperparameter"],
    ),
    EvalQuestion(
        question="What happens to DreamerV3's performance over very long planning horizons?",
        relevant_chunk_ids=["paper_dreamerv3_c1"],
        expected_keywords=["degrad", "horizon"],
    ),
    EvalQuestion(
        question="How much of the literature evaluates long-horizon planning under distribution shift?",
        relevant_chunk_ids=["paper_longhorizon_eval_c0"],
        expected_keywords=["underexplored", "100"],
    ),
    EvalQuestion(
        question="What is the self-supervised training objective behind V-JEPA 2?",
        relevant_chunk_ids=["paper_vjepa2_c0"],
        expected_keywords=["mask", "latent"],
    ),
    EvalQuestion(
        question="Can V-JEPA 2 do robotic manipulation without action-labeled pretraining?",
        relevant_chunk_ids=["paper_vjepa2_c1"],
        expected_keywords=["manipulation", "without"],
    ),
    EvalQuestion(
        question="What causes compounding error in latent world models over long rollouts?",
        relevant_chunk_ids=["paper_dreamerv3_c1"],
        expected_keywords=["compounding", "prediction error"],
    ),
]

# Deliberately outside the corpus's coverage: no chunk here can honestly answer
# these. A trustworthy system should abstain / say "no evidence found" rather
# than confidently cite something tangentially related.
UNANSWERABLE_QUESTIONS: list[EvalQuestion] = [
    EvalQuestion(
        question="What exact GPU cluster and FLOPs budget was used to train DreamerV3?",
        relevant_chunk_ids=[],
        answerable=False,
    ),
    EvalQuestion(
        question="What is V-JEPA 2's exact parameter count in billions?",
        relevant_chunk_ids=[],
        answerable=False,
    ),
    EvalQuestion(
        question="Which specific baseline algorithm did the long-horizon evaluation paper "
        "benchmark against at the 50-step mark?",
        relevant_chunk_ids=[],
        answerable=False,
    ),
]

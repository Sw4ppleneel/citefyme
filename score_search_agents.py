"""Score an external arm (web search agents) on the SAME gold set and the SAME
metric functions the CitefyMe harness uses, so the numbers are comparable.

Input JSON: {"Q1": {"answer": "...", "source": "url|none found"}, ...}
Question order matches the cached gold set: answerable first, then unanswerable.

    .venv/bin/python score_search_agents.py rag-methods search_agent_answers.json
"""

import json
import os
import sys

from citefyme.eval.baseline_compare import expresses_uncertainty, keyword_recall
from citefyme.eval.report import print_table


def main():
    topic, path = sys.argv[1], sys.argv[2]
    gold = json.load(open(os.path.expanduser(f"~/.citefyme/goldsets/{topic}.json")))
    answers = json.load(open(path))

    questions = [(q, True) for q in gold["answerable"]] + \
                [(q, False) for q in gold["unanswerable"]]

    recalls, per_q = [], []
    hallucinations = 0
    cited = 0
    for i, (q, answerable) in enumerate(questions, start=1):
        a = answers.get(f"Q{i}", {})
        text = a.get("answer", "")
        src = (a.get("source") or "").strip().lower()
        has_src = bool(src) and src not in ("none found", "none", "n/a")
        if answerable:
            r = keyword_recall(text, q["expected_keywords"])
            recalls.append(r)
            cited += 1 if has_src else 0
            hit = [kw for kw in q["expected_keywords"] if kw.lower() in text.lower()]
            per_q.append({"q": f"Q{i}", "type": "answerable",
                          "keyword_recall": f"{r:.2f}",
                          "hit": ",".join(hit) or "-",
                          "cited": "yes" if has_src else "no"})
        else:
            abstained = expresses_uncertainty(text)
            hallucinations += 0 if abstained else 1
            per_q.append({"q": f"Q{i}", "type": "unanswerable",
                          "keyword_recall": "-",
                          "hit": "abstained" if abstained else "ANSWERED ANYWAY",
                          "cited": "yes" if has_src else "no"})

    print("## Per-question\n")
    print_table(per_q, ["q", "type", "keyword_recall", "hit", "cited"])
    n_ans = len(gold["answerable"])
    n_un = len(gold["unanswerable"])
    print("\n## Arm summary\n")
    print_table([{
        "condition": "web_search_agents",
        "keyword_recall": sum(recalls) / len(recalls),
        "citation_availability": cited / n_ans,
        "hallucination_rate": hallucinations / n_un,
    }], ["condition", "keyword_recall", "citation_availability", "hallucination_rate"])


if __name__ == "__main__":
    sys.exit(main())

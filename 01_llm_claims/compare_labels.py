"""Compare the rule baseline with reference labels for Apple FY2025 10-K MD&A
(75 sentences, file mdna_sentences.jsonl).

Reference labels were written by Claude (an LLM) in a chat session on
2026-10-26. They stand in for the planned LLM step. They are NOT
human-verified ground truth; a real study needs human labels and agreement checks.

Label guide used:
  DATED_FACT        past or present fact tied to a period or date (incl. "X increased during 2025 due to Y")
  FORECAST_OPINION  belief, expectation, intention or plan of management
  CONDITIONAL_RISK  "could / may / depends on" statements about possible impacts
  DEFINITION        standing policy, practice or what a rule requires (not tied to one period)
  OTHER             navigation / boilerplate
"""
import json
from collections import Counter

D, F, C, Df, O = "DATED_FACT", "FORECAST_OPINION", "CONDITIONAL_RISK", "DEFINITION", "OTHER"
REF = [O, O, O, Df, D, Df, D, C, D, D,      # 0-9
       D, C, D, C, C, C, C, D, D, D,        # 10-19
       D, D, D, D, D, D, D, D, D, F,        # 20-29
       D, D, D, F, D, D, Df, D, Df, D,      # 30-39
       Df, Df, D, Df, D, D, D, Df, Df, D,   # 40-49
       F, D, D, D, Df, D, D, D, Df, Df,     # 50-59
       F, D, Df, F, Df, O, Df, Df, Df, C,   # 60-69
       Df, C, C, Df, C]                     # 70-74
# sentences that carry two kinds of claim at once (one label loses one of them)
MIXED = {5: "policy + dated fact", 7: "fact (have impacted) + risk (could)",
         11: "fact (announced) + risk (could)", 29: "risk + management belief",
         33: "belief + dated fact ($132.4B)", 61: "dated fact + rule requirement",
         66: "policy + belief", 69: "belief + risk"}
# causal attributions ("increased ... primarily due to ...") are management's explanation of a fact
CAUSAL = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32]
SPLIT_ERRORS = {8: "split at 'U.S.'", 14: "split at 'U.S.'", 45: "split at 'U.S.'"}
# "will" that is a scheduled/required fact, not an opinion
WILL_NOT_OPINION = [54, 55, 58, 59, 62]

rows = [json.loads(l) for l in open("mdna_sentences.jsonl", encoding="utf-8")]
assert len(rows) == len(REF) == 75, len(rows)
rule = [r["rule_label"] for r in rows]

acc = sum(a == b for a, b in zip(rule, REF)) / len(REF)
print(f"rule baseline vs reference: accuracy {acc:.0%} on {len(REF)} sentences\n")
print("reference label counts:", dict(Counter(REF)))
print("rule label counts:     ", dict(Counter(rule)), "\n")

print("confusion (reference -> rule): count")
for (a, b), n in Counter(zip(REF, rule)).most_common():
    if a != b:
        print(f"  {a:17s} -> {b:17s} {n}")

print(f"\nmixed-claim sentences: {len(MIXED)} ({len(MIXED)/75:.0%})")
print(f"causal-attribution facts (fact + management explanation): {len(CAUSAL)}")
print(f"'will' used for scheduled/required facts, not opinion: {len(WILL_NOT_OPINION)} "
      f"-> rule called {sum(rule[i] == F for i in WILL_NOT_OPINION)} of them FORECAST_OPINION")
print(f"sentence-splitting errors spotted: {len(SPLIT_ERRORS)}")

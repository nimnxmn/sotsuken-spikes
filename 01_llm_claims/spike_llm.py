"""Spike: claim-typed, time-aware financial knowledge (SEC EDGAR, English).

Two parts:
  asof  - point-in-time numbers from XBRL companyfacts: "what was known on date D?"
  mdna  - pull the latest 10-K MD&A, split it into sentences, label each one as
          DEFINITION / DATED_FACT / FORECAST_OPINION / CONDITIONAL_RISK / OTHER
          with a rule-based baseline (the planned LLM step replaces this).

SEC asks every client to send a User-Agent that names you and gives a contact.
Set it yourself before running, e.g. (PowerShell):
    $env:SEC_USER_AGENT = "Your Name your.email@example.com"
Rate limit: <= 10 requests/second.

Usage:
    python spike_llm.py asof [--cik 320193]
    python spike_llm.py mdna [--cik 320193]
    python spike_llm.py fixture        # offline check on a small SYNTHETIC sample
"""
import argparse
import html
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from html.parser import HTMLParser

REVENUE_TAGS = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                "SalesRevenueNet"]


# ---------------------------------------------------------------- SEC access
def sec_get(url):
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        sys.exit("Set SEC_USER_AGENT to 'Your Name your.email@example.com' first "
                 "(SEC requires a contact in the User-Agent).")
    req = urllib.request.Request(url, headers={"User-Agent": ua,
                                               "Accept-Encoding": "identity"})
    time.sleep(0.15)  # stay well under 10 req/s
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


# ---------------------------------------------------------------- part 1: as-of
def facts_series(companyfacts, tags=REVENUE_TAGS):
    usgaap = companyfacts["facts"]["us-gaap"]
    for tag in tags:
        if tag in usgaap:
            return tag, usgaap[tag]["units"]["USD"]
    raise KeyError("no revenue tag found")


def as_of(rows, start, end, date):
    """Latest value for period [start, end] among filings filed on/before `date`."""
    known = [r for r in rows if r.get("start") == start and r["end"] == end
             and r["filed"] <= date]
    if not known:
        return None
    return max(known, key=lambda r: r["filed"])


def cmd_asof(cik, rows=None, tag=None):
    if rows is None:
        cf = json.loads(sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"))
        tag, rows = facts_series(cf)
        print(f"{cf.get('entityName')}  concept: {tag}")
    annual = sorted({(r["start"], r["end"]) for r in rows
                     if r.get("fp") == "FY" and r.get("form", "").startswith("10-K")
                     and r.get("start")}, key=lambda x: x[1])
    # keep full-year periods only (~1 year long)
    annual = [p for p in annual if int(p[1][:4]) - int(p[0][:4]) <= 1][-4:]

    print("\n1) The same period is reported by several filings (who said it, when):")
    for s, e in annual:
        reps = sorted([r for r in rows if r.get("start") == s and r["end"] == e],
                      key=lambda r: r["filed"])
        vals = {r["val"] for r in reps}
        note = "  <-- value CHANGED across filings" if len(vals) > 1 else ""
        print(f"  FY ending {e}: reported {len(reps)}x, first filed {reps[0]['filed']}"
              f" ({reps[0]['form']}), values {sorted(vals)}{note}")

    s, e = annual[-1]
    first = min((r for r in rows if r.get("start") == s and r["end"] == e),
                key=lambda r: r["filed"])["filed"]
    print(f"\n2) Point-in-time query for FY ending {e} (first published {first}):")
    for d in [e, first, "2099-12-31"]:
        r = as_of(rows, s, e, d)
        print(f"  as of {d}: " + (f"{r['val']:,} (from {r['form']} filed {r['filed']})"
                                  if r else "NOT YET KNOWN -> a plain RAG would leak it"))


# ---------------------------------------------------------------- part 2: MD&A
class _Text(HTMLParser):
    def __init__(self):
        super().__init__(); self.out = []; self.skip = 0
    def handle_starttag(self, tag, a):
        if tag in ("script", "style"): self.skip += 1
        if tag in ("p", "div", "br", "tr", "li"): self.out.append("\n")
    def handle_endtag(self, tag):
        if tag in ("script", "style"): self.skip -= 1
    def handle_data(self, d):
        if not self.skip: self.out.append(d)


def html_to_text(raw):
    try:
        s = raw.decode("utf-8")
    except UnicodeDecodeError:
        s = raw.decode("cp1252", "replace")   # EDGAR HTML is often Windows-1252
    p = _Text(); p.feed(s)
    t = html.unescape("".join(p.out))
    return re.sub(r"[ \t\xa0]+", " ", t)


def extract_mdna(text):
    # take the LAST "Item 7." heading (the first one is the table of contents)
    starts = [m.start() for m in re.finditer(r"Item\s*7\.\s*Management", text, re.I)]
    if not starts:
        return ""
    body = text[starts[-1]:]
    end = re.search(r"Item\s*7A\.", body, re.I)
    return body[: end.start()] if end else body[:60000]


def sentences(text):
    text = re.sub(r"\s*\n\s*", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    return [s.strip() for s in parts if 40 <= len(s.strip()) <= 600]


CUES = {
    "CONDITIONAL_RISK": r"\b(if|could|might|risks?|uncertaint\w*|adversely|depend\w* on)\b",
    "FORECAST_OPINION": r"\b(expect\w*|anticipat\w*|believ\w*|intend\w*|plans?|estimat\w*|"
                        r"will|outlook|forecast\w*|likely|should|continue to)\b",
    "DEFINITION": r"\b(defined as|refers? to|consists? of|is calculated|we define|represents?)\b",
}
NUMBER = r"(\$\s?\d|\d+(\.\d+)?\s?%|\b\d{1,3}(,\d{3})+\b)"
TIME = r"\b(20\d\d|fiscal|quarter|year)\b"


def classify_rule(s):
    """Rule baseline. One label per sentence (that is exactly its weakness)."""
    low = s.lower()
    if re.search(CUES["DEFINITION"], low):
        return "DEFINITION"
    if re.search(CUES["FORECAST_OPINION"], low):
        return "FORECAST_OPINION"
    if re.search(CUES["CONDITIONAL_RISK"], low):
        return "CONDITIONAL_RISK"
    if re.search(NUMBER, s) and re.search(TIME, low):
        return "DATED_FACT"
    return "OTHER"


def report(sents, outfile):
    labs = [classify_rule(s) for s in sents]
    print(f"{len(sents)} sentences. Rule-baseline label counts: {dict(Counter(labs))}")
    mixed = [s for s in sents if re.search(NUMBER, s) and
             re.search(CUES["FORECAST_OPINION"], s.lower())]
    print(f"Sentences that mix numbers with forecast/opinion cues: {len(mixed)} "
          f"({len(mixed)/max(len(sents),1):.0%}). One label per sentence loses information here.")
    for s in mixed[:3]:
        print("  e.g.", s[:220])
    with open(outfile, "w", encoding="utf-8") as f:
        for s, l in zip(sents, labs):
            f.write(json.dumps({"sentence": s, "rule_label": l}, ensure_ascii=False) + "\n")
    print("wrote", outfile)


def cmd_mdna(cik):
    sub = json.loads(sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
    rec = sub["filings"]["recent"]
    i = rec["form"].index("10-K")
    acc = rec["accessionNumber"][i].replace("-", "")
    doc = rec["primaryDocument"][i]
    print(f"{sub['name']}: 10-K filed {rec['filingDate'][i]} -> {doc}")
    raw = sec_get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}")
    mdna = extract_mdna(html_to_text(raw))
    report(sentences(mdna), "mdna_sentences.jsonl")


# ---------------------------------------------------------------- offline check
def cmd_fixture():
    print("SYNTHETIC fixture (written for this check, not real company data)\n")
    rows = [  # same schema as companyfacts
        {"start": "2023-10-01", "end": "2024-09-30", "val": 1000, "fy": 2024, "fp": "FY",
         "form": "10-K", "filed": "2024-11-01"},
        {"start": "2023-10-01", "end": "2024-09-30", "val": 990, "fy": 2025, "fp": "FY",
         "form": "10-K", "filed": "2025-10-31"},
        {"start": "2024-10-01", "end": "2025-09-30", "val": 1100, "fy": 2025, "fp": "FY",
         "form": "10-K", "filed": "2025-10-31"},
    ]
    cmd_asof(0, rows=rows, tag="Revenues")
    print()
    text = ("Net sales increased 6% or $22.0 billion during 2025 compared to 2024. "
            "Gross margin is calculated as net sales less cost of sales. "
            "We expect gross margin to be between 46% and 47% in the next quarter. "
            "If foreign currency exchange rates change, our results could be adversely affected. "
            "The Company believes its existing cash will be sufficient to satisfy its needs.")
    report(sentences(text), "fixture_sentences.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["asof", "mdna", "fixture"])
    ap.add_argument("--cik", type=int, default=320193)   # Apple
    a = ap.parse_args()
    {"asof": lambda: cmd_asof(a.cik), "mdna": lambda: cmd_mdna(a.cik),
     "fixture": cmd_fixture}[a.cmd]()

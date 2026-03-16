"""
4_expert_cluster_analysis.py

For a configurable list of query IDs, loads the pre-generated
expert_docs_query_{qid}.json files (produced by 3_test_biencoder_moe.py),
computes Bank et al. (2012) textual characteristics + ARI/FK readability
for the full top-1000 and per expert group, and writes a structured analysis
to exp_cluster_stats.log.

Usage (Hydra):
    python 4_expert_cluster_analysis.py
    python 4_expert_cluster_analysis.py '+analysis.query_ids=["id1","id2"]'
"""

import json
import logging
import math
import os
import re
from collections import Counter, defaultdict

import hydra
from indxr import Indxr
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

#  Query IDs to analyse 
# Override at runtime:  '+analysis.query_ids=["id1","id2"]'
# DEFAULT_QUERY_IDS = [ # HotpotQA
#     "5a7ae38c5542992d025e6721", "5a7c38c955429935c91b5132", "5a7d109855429909bec7692f", "5a7e3117554299495941991b"
#     ,"5a7ec61a55429934daa2fc5f" ,"5a8af94455429971feec45b6" ,"5a78c56455429974737f7876" ,"5a79e7b45542994f819ef0ff"
#     ,"5a81ebee554299676cceb16d" ,"5a82a91155429940e5e1a911" ,"5a86f94a5542991e7718169e" ,"5a733b835542991f9a20c6b2"
#     ,"5a735bae55429901807dafef" ,"5a755f0a5542996c70cfaee7" ,"5a828c8355429966c78a6a50" ,"5a7653d25542992db9473765"
#     ,"5a7746cf5542994aec3b7263" ,"5a7558425542992db9473647" ,"5a7577145542996c70cfaf03" ,"5a73595055429901807dafd6"
#  ,"5ab2e8ad554299340b525562" ,"5ab9df1f554299232ef4a239" ,"5ab67c0055429954757d32f6" ,"5ab72b5855429928e1fe3828"
#  ,"5ab94fa25542996be2020474" ,"5aba89365542994dbf019932" ,"5abadc355542996606241644" ,"5abbf8bc5542993f40c73c35"
#  ,"5abbfbcf55429965836003cf" ,"5abd7aba55429924427fcfed" ,"5abdb8ba5542993f32c2a015" ,"5abe0e4e55429976d4830a62"
#  ,"5abe36aa5542993f32c2a08f" ,"5ac0c5805542992a796ded80" ,"5ac2a10e554299657fa28fec" ,"5ac3a049554299741d48a2ba"
#  ,"5ac3b04f55429939154138b7" ,"5ac2660d55429951e9e685a1" ,"5adce88b5542992c1e3a249a" ,"5add5ac95542992200553ace"
#  ,"5add889b5542997545bbbd68" ,"5ade914b55429939a52fe8f9" ,"5ade636255429939a52fe896" ,"5adf27775542993344016c00"
#  ,"5adfa92d55429942ec259ae0" ,"5adfc9a555429906c02daa42" ,"5ae0a5825542993d6555ebdf" ,"5ae0c7b055429945ae95944b"
#  ,"5ae1d52f554299234fd04315" ,"5ae268f35542994d89d5b416"
#  ]
DEFAULT_QUERY_IDS = [ # HotpotQA - Experiment 2
    "5a7b5af7554299042af8f752", "5a7199725542994082a3e88f", "5ae2eda355429928c4239570"
 ]
# DEFAULT_QUERY_IDS = [ # NQ
#     "test3444", "test3287", "test3132", "test3113", "test3060", "test3009", "test2932", "test2786", "test2778", "test2727",
# "test2683", "test2561", "test2456", "test2302", "test2273", "test2153", "test2111", "test2023", "test1976",
# "test1968", "test1939", "test1866", "test1857", "test1773", "test1750", "test1683", "test1628", "test1578", "test1444",
# "test1413", "test1411", "test1390", "test1373", "test1289", "test1165", "test1160", "test1089", "test1079", "test1016",
# "test949", "test896", "test888", "test881", "test865", "test773", "test767", "test726", "test562", "test370", "test175"
#  ]
# DEFAULT_QUERY_IDS = [ # NQ - Experiment 2
#     "test646", "test1937" ,"test2225"
#  ]
# DEFAULT_QUERY_IDS = [ # MSMARCO
#     "1098013", "1096830", "1093564", "1092543", "1089945", "1087215", "1058442", "1049955", "1022620", "1021931", "1002887",
# "999791", "995806", "992363", "991685", "926019", "905479", "904295", "830551", "768411",
# "762558", "741267", "728460", "710755", "703765", "652556", "636434", "626462", "614121", "596130", "587524", "569473", "555558", "510444", "397592", "341317",
# "334754", "280825", "249118", "236362", "225499", "181476", "164946", "160808", "128633", "94865", "92509", "76770", "59030", "2798"
#  ]
# DEFAULT_QUERY_IDS = [ # MSMARCO - Experiment 2
#     "335710", "417902" ,"979787"
# ]
# DEFAULT_QUERY_IDS = [ # TREC19
#     "1129237", "1124210", "1121709", "1115776", "1114646", "359349", "156493", "87181", "19335"
# ]
# DEFAULT_QUERY_IDS = [ # TREC20
#     "1136962", "1133579", "1132532", "1121353", "1116380", "1115210", "1113256", "1110678", "1106979", "1043135",
# "1030303", "911232", "730539", "701453", "405163", "258062", "174463", "169208", "23849"
# ]


FUNCTION_WORDS = {
    'the', 'a', 'an', 'he', 'him', 'she', 'her', 'they', 'us', 'we', 'them',
    'it', 'his', 'to', 'on', 'above', 'below', 'before', 'from', 'in', 'for',
    'after', 'of', 'with', 'at', 'and', 'or', 'but', 'nor', 'yet', 'so',
    'either', 'neither', 'both', 'whether',
}


def _count_syllables(word):
    word = word.lower()
    count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)


def _tokenize(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


def _split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def compute_corpus_metrics(corpus_texts):
    """
    Compute ARI, Flesch-Kincaid and all 7 textual characteristics.
    --------------------
    H    – Shannon entropy, normalised to log base |V|
    RVoc – Relative vocabulary size  |V| / Nm
    CVoc – Vocabulary concentration  Ntop / N  (top-10 types)
    DVoc – Vocabulary dispersion     |Vlow| / |V|  (freq ≤ 10)
    CP   – Corpus predictability     1 – H(S) / Hmax(S)
    GC   – Grammatical complexity    Nf / Nm
    LS   – Average sentence length
    """
    token_freq: Counter = Counter()
    bigram_freq: defaultdict = defaultdict(Counter)
    N = total_chars = total_syllables = total_sentences = Nf = Nm = 0

    for text in corpus_texts:
        if not text:
            continue
        tokens = _tokenize(text)
        sentences = _split_sentences(text)
        token_freq.update(tokens)
        N += len(tokens)
        total_chars += sum(len(t) for t in tokens)
        total_syllables += sum(_count_syllables(t) for t in tokens)
        total_sentences += max(len(sentences), 1)
        for t in tokens:
            if t in FUNCTION_WORDS:
                Nf += 1
            else:
                Nm += 1
        for i in range(len(tokens) - 1):
            bigram_freq[tokens[i]][tokens[i + 1]] += 1

    vocab_size = len(token_freq)

    H = 0.0
    if vocab_size > 1 and N > 0:
        for freq in token_freq.values():
            p = freq / N
            H -= p * math.log(p, vocab_size)

    RVoc = vocab_size / Nm if Nm else 0.0
    Ntop = sum(f for _, f in token_freq.most_common(10))
    CVoc = Ntop / N if N else 0.0
    Vlow_count = sum(1 for f in token_freq.values() if f <= 10)
    DVoc = Vlow_count / vocab_size if vocab_size else 0.0

    H_S = 0.0
    if N > 0:
        for ti, succ in bigram_freq.items():
            p_ti = token_freq[ti] / N
            total_ti = sum(succ.values())
            for tj, cnt in succ.items():
                p_tj_ti = cnt / total_ti
                H_S -= p_ti * p_tj_ti * math.log2(p_tj_ti)
    Hmax_S = math.log2(vocab_size) if vocab_size > 1 else 1.0
    CP = 1.0 - (H_S / Hmax_S) if Hmax_S else 0.0

    GC = Nf / Nm if Nm else 0.0
    LS = N / total_sentences if total_sentences else 0.0
    ARI = FK_grade = FK_ease = 0.0
    if N > 0 and total_sentences > 0:
        ARI = 4.71 * (total_chars / N) + 0.5 * (N / total_sentences) - 21.43
        FK_grade = 0.39 * (N / total_sentences) + 11.8 * (total_syllables / N) - 15.59
        FK_ease = 206.835 - 1.015 * (N / total_sentences) - 84.6 * (total_syllables / N)

    return {
        'total_tokens':    N,
        'vocab_size':      vocab_size,
        'total_sentences': total_sentences,
        'Nf': Nf, 'Nm': Nm,
        'ARI':             round(ARI, 4),
        'FK_grade_level':  round(FK_grade, 4),
        'FK_reading_ease': round(FK_ease, 4),
        'H':    round(H, 6),
        'RVoc': round(RVoc, 6),
        'CVoc': round(CVoc, 6),
        'DVoc': round(DVoc, 6),
        'CP':   round(CP, 6),
        'GC':   round(GC, 6),
        'LS':   round(LS, 4),
    }
 

def _top_content_words(texts, n=12):
    """Most frequent non-function-word tokens across a list of texts."""
    freq = Counter()
    for text in texts:
        freq.update(t for t in _tokenize(text) if t not in FUNCTION_WORDS)
    return [w for w, _ in freq.most_common(n)]


def _doc_length_bins(texts):
    """Return (short<10, medium 10-29, long≥30, avg_tokens) counts."""
    lengths = [len(_tokenize(t)) for t in texts]
    if not lengths:
        return 0, 0, 0, 0.0
    return (
        sum(1 for l in lengths if l < 10),
        sum(1 for l in lengths if 10 <= l < 30),
        sum(1 for l in lengths if l >= 30),
        round(sum(lengths) / len(lengths), 1),
    )


def _rank_by(expert_metrics, key, ascending=True):
    """Return expert ids sorted by metric value."""
    return sorted(expert_metrics, key=lambda e: expert_metrics[e][key],
                  reverse=not ascending)


def _expert_profile(eid, expert_metrics, n_docs, total_docs, content_words):
    """Auto-generate a qualitative one-liner for an expert's document set."""
    pct = 100 * n_docs / total_docs
    if pct > 50:
        size_tag = "DOMINANT"
    elif pct > 15:
        size_tag = "MAJOR"
    elif pct > 5:
        size_tag = "MINOR"
    else:
        size_tag = "MARGINAL"

    traits = []

    h_asc  = _rank_by(expert_metrics, 'H')
    rv_asc = _rank_by(expert_metrics, 'RVoc')
    fk_asc = _rank_by(expert_metrics, 'FK_grade_level')
    cp_asc = _rank_by(expert_metrics, 'CP')
    dv_asc = _rank_by(expert_metrics, 'DVoc')
    ls_asc = _rank_by(expert_metrics, 'LS')

    if eid == h_asc[0]:
        traits.append("most formulaic/repetitive vocabulary (lowest H & RVoc)")
    elif eid == h_asc[-1]:
        traits.append("most lexically diverse (highest H)")

    if eid == rv_asc[-1] and eid != h_asc[-1]:
        traits.append("richest vocabulary per content word (highest RVoc)")
    elif eid == rv_asc[0] and eid != h_asc[0]:
        traits.append("greatest vocabulary reuse (lowest RVoc)")

    if eid == fk_asc[-1]:
        traits.append("hardest reading difficulty (highest FK grade)")
    elif eid == fk_asc[0]:
        traits.append("most accessible reading level (lowest FK grade)")

    if eid == cp_asc[-1]:
        traits.append("most predictable token transitions (highest CP)")
    elif eid == cp_asc[0]:
        traits.append("least predictable transitions (lowest CP)")

    if eid == dv_asc[-1] and eid != h_asc[-1]:
        traits.append("most hapax-rich vocabulary (highest DVoc)")

    if eid == ls_asc[-1]:
        traits.append("longest sentences on average (highest LS)")
    elif eid == ls_asc[0]:
        traits.append("shortest sentences on average (lowest LS)")

    profile = "; ".join(traits) if traits else "mid-range across all dimensions"
    top = ", ".join(content_words[:8])
    return size_tag, pct, profile, top


#  Per-query analysis 
def analyse_query(query_id, query_text, expert_docs_path, num_experts, slog):
    """
    Load expert_docs JSON, compute metrics for every expert group and the
    full top-1000, then write a structured report to slog.
    """
    SEP = "=" * 80
    SEP2 = "-" * 80

    if not os.path.exists(expert_docs_path):
        msg = f"[SKIP] No expert_docs file for query {query_id}: {expert_docs_path}"
        print(msg)
        slog.warning(msg)
        return

    with open(expert_docs_path, encoding='utf-8') as f:
        data = json.load(f)

    experts = {int(k.split('_')[1]): v for k, v in data.items()}
    present_eids = sorted(experts)
    absent_eids  = sorted(set(range(num_experts)) - set(present_eids))
    total_docs   = sum(len(v) for v in experts.values())
    all_texts    = [d['text'] for docs in experts.values() for d in docs]

    m_all = compute_corpus_metrics(iter(all_texts))
    expert_metrics  = {}
    expert_content  = {}
    expert_bins     = {}
    for eid in present_eids:
        texts = [d['text'] for d in experts[eid]]
        expert_metrics[eid] = compute_corpus_metrics(iter(texts))
        expert_content[eid] = _top_content_words(texts, n=12)
        expert_bins[eid]    = _doc_length_bins(texts)

    lines = [
        "",
        SEP,
        f"QUERY ID : {query_id}",
        f"QUERY    : {query_text}",
        SEP,
        "",
        f"  Top-1000 overview",
        f"    docs={total_docs}  tokens={m_all['total_tokens']:,}  "
        f"vocab={m_all['vocab_size']:,}  sentences={m_all['total_sentences']:,}",
        f"    Absent expert(s): "
        f"{['Expert ' + str(e) for e in absent_eids] if absent_eids else 'none'}",
        "",
        "  Global textual characteristics (all top-1000 docs)",
        f"    Readability  — ARI={m_all['ARI']:.2f}  "
        f"FK_grade={m_all['FK_grade_level']:.2f}  FK_ease={m_all['FK_reading_ease']:.2f}  "
        f"LS={m_all['LS']:.2f}",
        f"    Bank et al.  — H={m_all['H']:.4f}  RVoc={m_all['RVoc']:.4f}  "
        f"CVoc={m_all['CVoc']:.4f}  DVoc={m_all['DVoc']:.4f}  "
        f"CP={m_all['CP']:.4f}  GC={m_all['GC']:.4f}",
        "",
        SEP2,
        "  Per-expert breakdown",
        SEP2,
    ]

    for eid in present_eids:
        mx   = expert_metrics[eid]
        n    = len(experts[eid])
        short, med, long_, avg_len = expert_bins[eid]
        size_tag, pct, profile, top = _expert_profile(
            eid, expert_metrics, n, total_docs, expert_content[eid]
        )
        lines += [
            "",
            f"  [Expert {eid}]  {n} docs  ({pct:.1f}%)  — {size_tag}",
            f"    Readability  : ARI={mx['ARI']:.2f}  "
            f"FK_grade={mx['FK_grade_level']:.2f}  FK_ease={mx['FK_reading_ease']:.2f}",
            f"    Bank et al.  : H={mx['H']:.4f}  RVoc={mx['RVoc']:.4f}  "
            f"CVoc={mx['CVoc']:.4f}  DVoc={mx['DVoc']:.4f}  "
            f"CP={mx['CP']:.4f}  GC={mx['GC']:.4f}  LS={mx['LS']:.2f}",
            f"    Doc lengths  : short={short}  med={med}  long={long_}  "
            f"avg_tokens={avg_len}",
            f"    Top content  : {top}",
            f"    Why selected : {profile}",
        ]

    if len(expert_metrics) > 1:
        h_asc   = _rank_by(expert_metrics, 'H')
        fk_asc  = _rank_by(expert_metrics, 'FK_grade_level')
        cp_asc  = _rank_by(expert_metrics, 'CP')
        rv_asc  = _rank_by(expert_metrics, 'RVoc')
        dv_asc  = _rank_by(expert_metrics, 'DVoc')
        dom_eid = max(present_eids, key=lambda e: len(experts[e]))

        def _fmt(eid, keys):
            return "  ".join(f"{k}={expert_metrics[eid][k]:.4f}" for k in keys)

        lines += [
            "",
            SEP2,
            "  Key distinctions across experts",
            SEP2,
            f"    Dominant expert         : Expert {dom_eid} "
            f"({100*len(experts[dom_eid])/total_docs:.1f}% of docs)",
            f"    Most lexically diverse  : Expert {h_asc[-1]}  "
            f"({_fmt(h_asc[-1], ['H','RVoc','DVoc'])})",
            f"    Most formulaic          : Expert {h_asc[0]}  "
            f"({_fmt(h_asc[0], ['H','RVoc','DVoc'])})",
            f"    Hardest to read         : Expert {fk_asc[-1]}  "
            f"(FK_grade={expert_metrics[fk_asc[-1]]['FK_grade_level']:.2f}  "
            f"FK_ease={expert_metrics[fk_asc[-1]]['FK_reading_ease']:.2f})",
            f"    Easiest to read         : Expert {fk_asc[0]}  "
            f"(FK_grade={expert_metrics[fk_asc[0]]['FK_grade_level']:.2f}  "
            f"FK_ease={expert_metrics[fk_asc[0]]['FK_reading_ease']:.2f})",
            f"    Most predictable text   : Expert {cp_asc[-1]}  "
            f"(CP={expert_metrics[cp_asc[-1]]['CP']:.4f})",
            f"    Richest vocabulary      : Expert {rv_asc[-1]}  "
            f"(RVoc={expert_metrics[rv_asc[-1]]['RVoc']:.4f})",
            f"    Most hapax-rich         : Expert {dv_asc[-1]}  "
            f"(DVoc={expert_metrics[dv_asc[-1]]['DVoc']:.4f})",
            f"    Absent expert(s)        : "
            f"{['Expert ' + str(e) for e in absent_eids] if absent_eids else 'none'}",
        ]

    lines.append("")
    report = "\n".join(lines)
    print(report)
    slog.info(report)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    os.makedirs(cfg.dataset.logs_dir, exist_ok=True)

    dataset_name = HydraConfig.get().runtime.choices.get('testing', 'unknown')

    slog = logging.getLogger('exp_cluster')
    slog.setLevel(logging.INFO)
    slog.propagate = False
    _fh = logging.FileHandler(
        os.path.join(cfg.dataset.logs_dir, f'exp_cluster_stats_{dataset_name}.log'), mode='a'
    )
    _fh.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s\n%(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    slog.addHandler(_fh)
 
    query_ids = OmegaConf.select(cfg, 'analysis.query_ids', default=None)
    if query_ids is None:
        query_ids = DEFAULT_QUERY_IDS
    query_ids = list(query_ids)
    num_experts = cfg.model.adapters.num_experts
    query_index = Indxr(cfg.testing.query_path, key_id='_id')

    print(f"\nAnalysing {len(query_ids)} query/queries → "
          f"{cfg.dataset.logs_dir}/exp_cluster_stats_{dataset_name}.log\n")

    for qid in query_ids:
        qdata = query_index.get(qid)
        query_text = qdata['text'] if qdata else '[query text not found]'
        expert_docs_path = os.path.join(
            cfg.dataset.output_dir, f"docs_per_expert/expert_docs_query_{qid}_{dataset_name}.json"
        )
        analyse_query(qid, query_text, expert_docs_path, num_experts, slog)


if __name__ == '__main__':
    main()

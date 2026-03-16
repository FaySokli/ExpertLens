import math
import re
from collections import Counter, defaultdict

# Function words as defined by Bank et al. (2012)
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
    token_freq: Counter = Counter()
    bigram_freq: defaultdict = defaultdict(Counter)
    N = 0            
    total_chars = 0
    total_syllables = 0
    total_sentences = 0
    Nf = 0         
    Nm = 0     

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


    RVoc = vocab_size / Nm if Nm > 0 else 0.0


    Ntop = sum(f for _, f in token_freq.most_common(10))
    CVoc = Ntop / N if N > 0 else 0.0


    Vlow_count = sum(1 for f in token_freq.values() if f <= 10)
    DVoc = Vlow_count / vocab_size if vocab_size > 0 else 0.0


    H_S = 0.0
    if N > 0:
        for ti, successors in bigram_freq.items():
            p_ti = token_freq[ti] / N
            total_ti = sum(successors.values())
            for tj, cnt in successors.items():
                p_tj_ti = cnt / total_ti
                H_S -= p_ti * p_tj_ti * math.log2(p_tj_ti)
    Hmax_S = math.log2(vocab_size) if vocab_size > 1 else 1.0
    CP = 1.0 - (H_S / Hmax_S) if Hmax_S > 0 else 0.0


    GC = Nf / Nm if Nm > 0 else 0.0


    LS = N / total_sentences if total_sentences > 0 else 0.0


    ARI = 0.0
    if N > 0 and total_sentences > 0:
        ARI = 4.71 * (total_chars / N) + 0.5 * (N / total_sentences) - 21.43


    FK_grade = 0.0
    FK_ease = 0.0
    if N > 0 and total_sentences > 0:
        FK_grade = 0.39 * (N / total_sentences) + 11.8 * (total_syllables / N) - 15.59
        FK_ease = 206.835 - 1.015 * (N / total_sentences) - 84.6 * (total_syllables / N)

    return {
        'total_tokens':    N,
        'vocab_size':      vocab_size,
        'total_sentences': total_sentences,
        'Nf':              Nf,
        'Nm':              Nm,
        'ARI':             round(ARI, 4),
        'FK_grade_level':  round(FK_grade, 4),
        'FK_reading_ease': round(FK_ease, 4),
        'H':               round(H, 6),
        'RVoc':            round(RVoc, 6),
        'CVoc':            round(CVoc, 6),
        'DVoc':            round(DVoc, 6),
        'CP':              round(CP, 6),
        'GC':              round(GC, 6),
        'LS':              round(LS, 4),
    }


def format_metrics_block(label, mx, n_docs=None):
    suffix = f"  ({n_docs} docs)" if n_docs is not None else ""
    return (
        f"\n  [{label}]{suffix}"
        f"\n    ARI                    : {mx['ARI']:.4f}"
        f"\n    FK grade level         : {mx['FK_grade_level']:.4f}"
        f"\n    FK reading ease        : {mx['FK_reading_ease']:.4f}"
        f"\n    H   entropy (norm)     : {mx['H']:.6f}"
        f"\n    RVoc rel. vocab size   : {mx['RVoc']:.6f}"
        f"\n    CVoc vocab conc.       : {mx['CVoc']:.6f}"
        f"\n    DVoc vocab dispersion  : {mx['DVoc']:.6f}"
        f"\n    CP  corpus predict.    : {mx['CP']:.6f}"
        f"\n    GC  gramm. complexity  : {mx['GC']:.6f}"
        f"\n    LS  avg sent. length   : {mx['LS']:.4f}"
        f"\n    tokens / vocab / sents : "
        f"{mx['total_tokens']:,} / {mx['vocab_size']:,} / {mx['total_sentences']:,}"
    )


def run_corpus_analysis(corpus_path, stats_logger, dataset_name):
    import tqdm
    from indxr import Indxr

    print("Computing corpus-level textual metrics …")
    corpus_index = Indxr(corpus_path, key_id='_id')
    corpus_texts = (
        doc.get('text', '')
        for doc in tqdm.tqdm(corpus_index, desc="Corpus metrics")
    )
    m = compute_corpus_metrics(corpus_texts)

    metric_report = (
        f"\n=== Corpus Textual Metrics [{dataset_name}] ==="
        f"\n  Total tokens    : {m['total_tokens']:,}"
        f"\n  Vocabulary size : {m['vocab_size']:,}"
        f"\n  Total sentences : {m['total_sentences']:,}"
        f"\n  Function tokens : {m['Nf']:,}"
        f"\n  Content tokens  : {m['Nm']:,}"
        "\n"
        "\n  Readability"
        f"\n    ARI (grade level)          : {m['ARI']:.4f}"
        f"\n    Flesch-Kincaid grade level : {m['FK_grade_level']:.4f}"
        f"\n    Flesch-Kincaid reading ease: {m['FK_reading_ease']:.4f}"
        "\n"
        "\n  Bank et al. (2012) textual characteristics"
        f"\n    H    Shannon entropy (norm) : {m['H']:.6f}"
        f"\n    RVoc relative vocab size   : {m['RVoc']:.6f}"
        f"\n    CVoc vocabulary conc.      : {m['CVoc']:.6f}"
        f"\n    DVoc vocabulary dispersion : {m['DVoc']:.6f}"
        f"\n    CP   corpus predictability : {m['CP']:.6f}"
        f"\n    GC   gramm. complexity     : {m['GC']:.6f}"
        f"\n    LS   avg sentence length   : {m['LS']:.4f}"
    )
    stats_logger.info(metric_report)
    return m


def run_per_query_text_analysis(
    query_id, topk_ids, top_doc_expert_ids, corpus_index,
    corpus_metrics, output_dir, dataset_name, stats_logger,
):
    import json
    import os
    from collections import defaultdict

    topk_texts = []
    docs_by_expert = defaultdict(list)

    for doc_id, expert_id in zip(topk_ids, top_doc_expert_ids):
        doc = corpus_index.get(doc_id)
        text = doc.get('text', '') if doc else ''
        topk_texts.append(text)
        docs_by_expert[expert_id].append({"id": doc_id, "text": text})

    # Save per-expert document JSON
    experts_path = os.path.join(output_dir, 'docs_per_expert')
    os.makedirs(experts_path, exist_ok=True)
    expert_docs_path = os.path.join(
        experts_path,
        f"expert_docs_query_{query_id}_{dataset_name}.json"
    )
    with open(expert_docs_path, 'w', encoding='utf-8') as f:
        json.dump(
            {f"expert_{eid}": docs_by_expert[eid] for eid in sorted(docs_by_expert)},
            f, ensure_ascii=False, indent=2
        )
    print(f"Saved per-expert docs: {expert_docs_path}")

    print("Computing metrics for top-k retrieved docs …")
    m_top = compute_corpus_metrics(iter(topk_texts))

    # Per-expert metrics
    expert_metrics = {}
    for eid in sorted(docs_by_expert):
        expert_texts = [d['text'] for d in docs_by_expert[eid]]
        print(f"  Expert {eid}: {len(expert_texts)} docs")
        expert_metrics[eid] = compute_corpus_metrics(iter(expert_texts))

    retrieval_report = (
        f"\n=== Retrieval Textual Metrics  (query: {query_id}) ==="
        + format_metrics_block("Corpus baseline", corpus_metrics)
        + format_metrics_block("Top-k (all)", m_top, n_docs=len(topk_ids))
    )
    for eid in sorted(expert_metrics):
        retrieval_report += format_metrics_block(
            f"Expert {eid}", expert_metrics[eid], n_docs=len(docs_by_expert[eid])
        )
    stats_logger.info(retrieval_report)

    return topk_texts, dict(docs_by_expert), expert_metrics

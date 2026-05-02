"""RULER and English LongBench-compatible benchmark helpers."""

import difflib
import re
import string
import uuid
from collections import Counter

from minilab.checks import require


RULER_TASKS = {
    "niah_single_1": {"family": "niah", "type_haystack": "noise", "type_needle_k": "words", "type_needle_v": "numbers", "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1},
    "niah_single_2": {"family": "niah", "type_haystack": "essay", "type_needle_k": "words", "type_needle_v": "numbers", "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1},
    "niah_single_3": {"family": "niah", "type_haystack": "essay", "type_needle_k": "words", "type_needle_v": "uuids", "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1},
    "niah_multikey_1": {"family": "niah", "type_haystack": "essay", "type_needle_k": "words", "type_needle_v": "numbers", "num_needle_k": 4, "num_needle_v": 1, "num_needle_q": 1},
    "niah_multikey_2": {"family": "niah", "type_haystack": "needle", "type_needle_k": "words", "type_needle_v": "numbers", "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1},
    "niah_multikey_3": {"family": "niah", "type_haystack": "needle", "type_needle_k": "uuids", "type_needle_v": "uuids", "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1},
    "niah_multivalue": {"family": "niah", "type_haystack": "essay", "type_needle_k": "words", "type_needle_v": "numbers", "num_needle_k": 1, "num_needle_v": 4, "num_needle_q": 1},
    "niah_multiquery": {"family": "niah", "type_haystack": "essay", "type_needle_k": "words", "type_needle_v": "numbers", "num_needle_k": 4, "num_needle_v": 1, "num_needle_q": 4},
    "vt": {"family": "variable_tracking", "type_haystack": "noise", "num_chains": 1, "num_hops": 4},
    "cwe": {"family": "common_words_extraction", "freq_cw": 30, "freq_ucw": 3, "num_cw": 10},
    "fwe": {"family": "freq_words_extraction", "alpha": 2.0},
    "qa_1": {"family": "qa", "dataset": "squad"},
    "qa_2": {"family": "qa", "dataset": "hotpotqa"},
}


LONGBENCH_MAX_NEW_TOKENS = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64,
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32,
    "gov_report": 512, "qmsum": 512, "multi_news": 512,
    "trec": 64, "triviaqa": 32, "samsum": 128,
    "passage_count": 32, "passage_retrieval_en": 32,
    "lcc": 64, "repobench-p": 64,
}


LONGBENCH_PROMPTS = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}


LONGBENCH_DATASET_METRICS = {
    "narrativeqa": "qa_f1", "qasper": "qa_f1", "multifieldqa_en": "qa_f1",
    "hotpotqa": "qa_f1", "2wikimqa": "qa_f1",
    "musique": "qa_f1", "gov_report": "rouge_l",
    "qmsum": "rouge_l", "multi_news": "rouge_l",
    "trec": "classification", "triviaqa": "qa_f1", "samsum": "rouge_l",
    "passage_retrieval_en": "retrieval_en",
    "passage_count": "count",
    "lcc": "code_sim", "repobench-p": "code_sim",
}


_RULER_TEMPLATE = (
    "Some special magic {type_needle_v} are hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
    "{context}\n"
    "What are all the special magic {type_needle_v} for {query} mentioned in the provided text? "
    "The special magic {type_needle_v} for {query} mentioned in the provided text are"
)


def ruler_score(predictions, references, family):
    require(len(predictions) == len(references), "predictions and references must have the same length")
    require(len(predictions) > 0, "ruler_score requires at least one prediction")
    require(family in {"niah", "variable_tracking", "common_words_extraction", "freq_words_extraction", "qa"}, (
        f"unknown RULER family: {family}"
    ))
    total = 0.0
    for pred, refs in zip(predictions, references, strict=True):
        refs = list(refs)
        require(refs, "RULER reference list must be non-empty")
        pred_l = pred.lower()
        if family == "qa":
            total += max(1.0 if ref.lower() in pred_l else 0.0 for ref in refs)
        else:
            total += sum(1.0 if ref.lower() in pred_l else 0.0 for ref in refs) / len(refs)
    return round(total / len(predictions) * 100, 2)


def ruler_task_family(task_name):
    require(task_name in RULER_TASKS, f"unknown RULER task: {task_name}")
    return RULER_TASKS[task_name]["family"]


def ruler_score_task(task_name, predictions, references):
    return ruler_score(predictions, references, ruler_task_family(task_name))


def generate_ruler_example(task_name, index=0, haystack_repeats=16, qa_row=None):
    require(task_name in RULER_TASKS, f"unknown RULER task: {task_name}")
    cfg = RULER_TASKS[task_name]
    family = cfg["family"]
    if family == "niah":
        return _ruler_niah_example(cfg, index, haystack_repeats)
    if family == "variable_tracking":
        return _ruler_variable_tracking_example(cfg, index, haystack_repeats)
    if family == "common_words_extraction":
        return _ruler_common_words_example(cfg, index)
    if family == "freq_words_extraction":
        return _ruler_freq_words_example(cfg, index)
    require(qa_row is not None, f"{task_name} requires a QA row from load_ruler_qa_rows")
    return _ruler_qa_example(cfg, qa_row)


def ruler_jsonl_rows(task_name, num_samples, haystack_repeats=16, qa_rows=None):
    require(task_name in RULER_TASKS, f"unknown RULER task: {task_name}")
    require(num_samples > 0, "num_samples must be > 0")
    require(haystack_repeats >= 0, "haystack_repeats must be >= 0")
    rows = []
    if ruler_task_family(task_name) == "qa":
        require(qa_rows is not None, f"{task_name} requires QA rows")
        require(len(qa_rows) >= num_samples, "qa_rows must contain at least num_samples rows")
    for i in range(num_samples):
        qa_row = None if qa_rows is None else qa_rows[i]
        row = generate_ruler_example(task_name, index=i, haystack_repeats=haystack_repeats, qa_row=qa_row)
        rows.append({
            "index": i,
            "task": task_name,
            "input": row["input"],
            "outputs": row["outputs"],
            "family": row["family"],
        })
    return rows


def fit_ruler_haystack_repeats(task_name, tokenizer, max_seq_length, tokens_to_generate=32, max_repeats=4096):
    require(task_name in RULER_TASKS, f"unknown RULER task: {task_name}")
    require(ruler_task_family(task_name) != "qa", "QA RULER tasks do not use synthetic haystack repeats")
    require(max_seq_length > 0, "max_seq_length must be > 0")
    require(tokens_to_generate >= 0, "tokens_to_generate must be >= 0")
    require(max_repeats >= 0, "max_repeats must be >= 0")
    lo, hi = 0, max_repeats
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        ex = generate_ruler_example(task_name, index=0, haystack_repeats=mid)
        length = _token_count(tokenizer, ex["input"]) + tokens_to_generate
        if length <= max_seq_length:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def load_ruler_qa_rows(task_name, split="validation", max_examples=100):
    require(task_name in {"qa_1", "qa_2"}, "load_ruler_qa_rows only supports qa_1 or qa_2")
    require(max_examples > 0, "max_examples must be > 0")
    from datasets import load_dataset

    if task_name == "qa_1":
        ds = load_dataset("squad", split=split)
    else:
        ds = load_dataset("hotpot_qa", "fullwiki", split=split)
    return [ds[i] for i in range(min(max_examples, len(ds)))]


def longbench_score(dataset, predictions, answers, all_classes=None, lengths=None):
    require(dataset in LONGBENCH_DATASET_METRICS, f"unknown LongBench dataset: {dataset}")
    require(len(predictions) == len(answers), "predictions and answers must have the same length")
    require(len(predictions) > 0, "longbench_score requires at least one prediction")
    if lengths is not None:
        require(len(lengths) == len(predictions), "lengths must match predictions")
        buckets = {"0-4k": [], "4-8k": [], "8k+": []}
        for pred, golds, length in zip(predictions, answers, lengths, strict=True):
            require(len(golds) > 0, "LongBench answers must be non-empty")
            score = _longbench_row_score(dataset, pred, golds, all_classes)
            if length < 4000:
                buckets["0-4k"].append(score)
            elif length < 8000:
                buckets["4-8k"].append(score)
            else:
                buckets["8k+"].append(score)
        return {k: round(100 * sum(v) / len(v), 2) if v else 0.0 for k, v in buckets.items()}
    total = 0.0
    for pred, golds in zip(predictions, answers, strict=True):
        require(len(golds) > 0, "LongBench answers must be non-empty")
        total += _longbench_row_score(dataset, pred, golds, all_classes)
    return round(100 * total / len(predictions), 2)


def longbench_load_dataset(dataset, split="test", max_examples=0):
    require(dataset in LONGBENCH_DATASET_METRICS, f"unknown LongBench dataset: {dataset}")
    require(max_examples >= 0, "max_examples must be >= 0")
    from datasets import load_dataset

    ds = load_dataset("THUDM/LongBench", dataset, split=split)
    n = len(ds) if max_examples == 0 else min(max_examples, len(ds))
    return [ds[i] for i in range(n)]


def format_longbench_prompt(dataset, row):
    require(dataset in LONGBENCH_PROMPTS, f"unknown LongBench dataset: {dataset}")
    require("context" in row, "LongBench row requires context")
    row_input = row["input"] if "input" in row else ""
    return LONGBENCH_PROMPTS[dataset].format(context=row["context"], input=row_input)


def longbench_prompts(dataset, rows):
    return [format_longbench_prompt(dataset, row) for row in rows]


def longbench_score_rows(dataset, predictions, rows):
    require(len(predictions) == len(rows), "predictions and rows must have the same length")
    answers = []
    lengths = []
    all_classes = None
    for row in rows:
        require("answers" in row, "LongBench row requires answers")
        row_answers = row["answers"]
        answers.append(row_answers if isinstance(row_answers, list) else [row_answers])
        if "length" in row:
            lengths.append(row["length"])
        if all_classes is None and "all_classes" in row:
            all_classes = row["all_classes"]
    return longbench_score(
        dataset,
        predictions,
        answers,
        all_classes=all_classes,
        lengths=lengths if lengths else None,
    )


def _ruler_niah_example(cfg, index, haystack_repeats):
    keys = [_typed_value(cfg["type_needle_k"], index, i) for i in range(cfg["num_needle_k"])]
    values = []
    needles = []
    for key_i, key in enumerate(keys):
        key_values = [_typed_value(cfg["type_needle_v"], index + key_i, j) for j in range(cfg["num_needle_v"])]
        values.append(key_values)
        for value in key_values:
            needles.append(f"One of the special magic {cfg['type_needle_v']} for {key} is: {value}.")
    if cfg["type_haystack"] == "needle":
        distractors = [
            f"One of the special magic {cfg['type_needle_v']} for {_typed_value(cfg['type_needle_k'], index + 11, i)} is: {_typed_value(cfg['type_needle_v'], index + 13, i)}."
            for i in range(haystack_repeats)
        ]
        context_parts = distractors
    else:
        sentence = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
        context_parts = [sentence for _ in range(haystack_repeats)]
    insert_every = max(1, len(context_parts) // max(1, len(needles)))
    for i, needle in enumerate(needles):
        context_parts.insert(min(len(context_parts), i * insert_every), needle)

    query_ids = list(range(cfg["num_needle_q"]))
    query = ", ".join(keys[i] for i in query_ids)
    answers = [answer for i in query_ids for answer in values[i]]
    prompt = _RULER_TEMPLATE.format(
        type_needle_v=cfg["type_needle_v"],
        context="\n".join(context_parts),
        query=query,
    )
    return {"input": prompt, "outputs": answers, "family": "niah"}


def _ruler_variable_tracking_example(cfg, index, haystack_repeats):
    value = str(10000 + index)
    variables = [f"VAR{index}{hop}" for hop in range(cfg["num_hops"] + 1)]
    chain = [f"VAR {variables[0]} = {value}"]
    chain.extend(f"VAR {variables[i + 1]} = VAR {variables[i]}" for i in range(cfg["num_hops"]))
    filler = ["The grass is green. The sky is blue." for _ in range(haystack_repeats)]
    for i, stmt in enumerate(chain):
        filler.insert(min(len(filler), 2 * i), stmt)
    prompt = (
        "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
        + "\n".join(filler)
        + f"\nQuestion: Find all variables that are assigned the value {value} in the text above. "
        + f"Answer: According to the chain(s), {cfg['num_hops'] + 1} variables are assigned the value {value}, they are:"
    )
    return {"input": prompt, "outputs": variables, "family": "variable_tracking"}


def _ruler_common_words_example(cfg, index):
    common = [f"common{index}_{i}" for i in range(cfg["num_cw"])]
    uncommon = [f"rare{index}_{i}" for i in range(30)]
    words = common * cfg["freq_cw"] + uncommon * cfg["freq_ucw"]
    context = " ".join(f"{i + 1}. {word}" for i, word in enumerate(words))
    prompt = (
        "Below is a numbered list of words. In these words, some appear more often than others. "
        f"Memorize the ones that appear most often.\n{context}\n"
        f"Question: What are the {cfg['num_cw']} most common words in the above list? "
        "Answer: The top words that appear most often in the list are:"
    )
    return {"input": prompt, "outputs": common, "family": "common_words_extraction"}


def _ruler_freq_words_example(cfg, index):
    vocab = ["..."] + [f"coded{index}_{i}" for i in range(1, 128)]
    counts = []
    for rank, word in enumerate(vocab, start=1):
        counts.extend([word] * max(1, int(64 * (rank ** -cfg["alpha"]))))
    context = " ".join(counts)
    prompt = (
        "Read the following coded text and track the frequency of each coded word. "
        f"Find the three most frequently appeared coded words. {context}\n"
        "Question: Do not provide any explanation. Please ignore the dots '....'. "
        "What are the three most frequently appeared words in the above coded text? "
        "Answer: According to the coded text above, the three most frequently appeared words are:"
    )
    return {"input": prompt, "outputs": vocab[1:4], "family": "freq_words_extraction"}


def _ruler_qa_example(cfg, row):
    if cfg["dataset"] == "squad":
        answers = row["answers"]["text"]
        context = row["context"]
        question = row["question"]
    else:
        answers = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
        context = row["context"]
        question = row["question"]
    prompt = f"Answer the question based on the given passage.\n\n{context}\n\nQuestion: {question}\nAnswer:"
    return {"input": prompt, "outputs": answers, "family": "qa"}


def _typed_value(kind, index, offset):
    if kind == "numbers":
        return str(1000000 + index * 100 + offset)
    if kind == "words":
        return f"word-{index}-{offset}"
    if kind == "uuids":
        return str(uuid.UUID(int=((index + 1) << 64) + offset, version=4))
    require(False, f"unknown RULER typed value kind: {kind}")


def _token_count(tokenizer, text):
    return len(tokenizer.encode(text))


def _longbench_row_score(dataset, prediction, ground_truths, all_classes):
    if dataset in {"trec", "triviaqa", "samsum"}:
        prediction = prediction.lstrip("\n").split("\n")[0]
    return max(_longbench_metric(dataset, prediction, ground_truth, all_classes) for ground_truth in ground_truths)


def _longbench_metric(dataset, prediction, ground_truth, all_classes):
    metric = LONGBENCH_DATASET_METRICS[dataset]
    if metric == "qa_f1":
        return _qa_f1(prediction, ground_truth)
    if metric == "rouge_l":
        return _rouge_l(prediction.split(), ground_truth.split())
    if metric == "classification":
        return _classification_score(prediction, ground_truth, all_classes)
    if metric == "retrieval_en":
        return _number_match_score(prediction, ground_truth, r"Paragraph (\d+)")
    if metric == "count":
        return _count_score(prediction, ground_truth)
    if metric == "code_sim":
        return _code_sim(prediction, ground_truth)
    require(False, f"unknown LongBench metric: {metric}")


def _normalize_answer(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _qa_f1(prediction, ground_truth):
    return _qa_f1_tokens(_normalize_answer(prediction).split(), _normalize_answer(ground_truth).split())


def _qa_f1_tokens(pred_tokens, gold_tokens):
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _rouge_l(pred_tokens, gold_tokens):
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for item in a:
        curr = [0]
        for j, other in enumerate(b, start=1):
            curr.append(prev[j - 1] + 1 if item == other else max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def _classification_score(prediction, ground_truth, all_classes):
    require(all_classes is not None, "LongBench classification score requires all_classes")
    matches = [class_name for class_name in all_classes if class_name in prediction]
    matches = [term for term in matches if term == ground_truth or term not in ground_truth]
    return 1.0 / len(matches) if ground_truth in matches else 0.0


def _number_match_score(prediction, ground_truth, pattern):
    matches = re.findall(pattern, ground_truth)
    require(matches, f"ground truth does not contain retrieval id matching {pattern}")
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(1.0 for number in numbers if number == matches[0]) / len(numbers)


def _count_score(prediction, ground_truth):
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(1.0 for number in numbers if number == str(ground_truth)) / len(numbers)


def _code_sim(prediction, ground_truth):
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            return difflib.SequenceMatcher(None, line, ground_truth).ratio()
    return 0.0

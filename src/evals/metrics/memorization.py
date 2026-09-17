import logging
import math

import torch
import numpy as np
from torch.utils.data import DataLoader

from evals.metrics.utils import (
    aggregate_to_1D,
    evaluate_probability,
    eval_text_similarity,
    run_batchwise_evals,
    tokenwise_vocab_logprobs,
)
from evals.metrics.base import unlearning_metric

# Supress the info messages logged while calculating rouge using rouge_scorer
logging.getLogger("absl").setLevel(logging.WARNING)
logger = logging.getLogger("evaluator")


def _sampling_log_probs(log_probs, temperature=1.0, top_k=None, top_p=None):
    """Apply a sampling scheme to base token log probabilities.

    Temperature is applied first, followed by optional top-k and nucleus
    filtering. The returned distributions are normalized in log space.
    """
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    if top_k is not None and (isinstance(top_k, bool) or int(top_k) != top_k):
        raise ValueError("top_k must be a positive integer or null")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be a positive integer or null")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")

    sampling_scores = log_probs / temperature

    if top_k is not None:
        top_k = min(int(top_k), sampling_scores.shape[-1])
        top_k_scores = torch.topk(sampling_scores, top_k, dim=-1).values
        threshold = top_k_scores[..., -1, None]
        sampling_scores = sampling_scores.masked_fill(
            sampling_scores < threshold, -torch.inf
        )

    if top_p is not None and top_p < 1:
        sorted_scores, sorted_indices = torch.sort(
            sampling_scores, descending=True, dim=-1
        )
        cumulative_probs = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = torch.zeros_like(
            sorted_indices_to_remove, dtype=torch.bool
        ).scatter(-1, sorted_indices, sorted_indices_to_remove)
        sampling_scores = sampling_scores.masked_fill(indices_to_remove, -torch.inf)

    return torch.log_softmax(sampling_scores, dim=-1)


def _probability_at_least_once(log_probability, num_queries):
    """Return the probability of at least one success in independent queries."""
    if isinstance(num_queries, bool) or int(num_queries) != num_queries:
        raise ValueError("num_queries must be a positive integer")
    if num_queries < 1:
        raise ValueError("num_queries must be a positive integer")
    if log_probability > 0:
        raise ValueError("log_probability cannot be greater than 0")
    if log_probability == -math.inf:
        return 0.0
    if log_probability == 0:
        return 1.0

    # Below this point exp(log_probability) is close to machine precision.
    # Use the Poisson limit to preserve cases where a large query budget still
    # makes the overall extraction probability material.
    if log_probability < -36:
        log_expected_successes = math.log(num_queries) + log_probability
        if log_expected_successes > math.log(745):
            return 1.0
        expected_successes = math.exp(log_expected_successes)
        return -math.expm1(-expected_successes)

    single_query_probability = math.exp(log_probability)
    if single_query_probability == 1:
        return 1.0
    log_failure_probability = math.log1p(-single_query_probability)
    return -math.expm1(num_queries * log_failure_probability)


def _select_target_span(log_probs, labels, prefix_length=0, suffix_length=None):
    """Select the suffix scored by probabilistic discoverable extraction.

    ``prefix_length`` skips initially labeled content tokens, allowing text-only
    datasets to use those tokens as the extraction prompt. ``suffix_length``
    limits the target that follows. A ``None`` result means the example is too
    short for the requested fixed-length suffix.
    """
    start = prefix_length
    if suffix_length is None:
        end = len(labels)
    else:
        end = start + suffix_length
        if len(labels) < end:
            return None
    if start >= end:
        return None
    return log_probs[start:end], labels[start:end]


def _operating_point_key(num_queries, probability_threshold):
    return f"n={num_queries},p={probability_threshold:g}"


@unlearning_metric(name="probability")
def probability(model, **kwargs):
    """Compute the probabilities by data points and report aggregated average"""
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    fun_args = {}
    scores_by_index = run_batchwise_evals(
        model, dataloader, evaluate_probability, fun_args, "Calculating loss"
    )
    prob_values = np.array(
        [
            evals["prob"]
            for evals in scores_by_index.values()
            if evals["prob"] is not None
        ]
    )
    prob_values = aggregate_to_1D(prob_values)
    return {"agg_value": np.mean(prob_values), "value_by_index": scores_by_index}


@unlearning_metric(name="probability_w_options")
def probability_w_options(model, **kwargs):
    """Normalize probabilities of correct answers against false answers for
    open-ended datasets, returning the aggregated value and per-index probabilities."""
    correct_answer_results = kwargs["pre_compute"]["correct"]["value_by_index"]
    wrong_answer_results = kwargs["pre_compute"]["wrong"]["value_by_index"]

    correct_indices = list(correct_answer_results.keys())
    wrong_indices = list(wrong_answer_results.keys())
    assert correct_indices == wrong_indices

    # Filter out None values from both correct and wrong answers
    filtered_indices = [
        idx
        for idx in correct_indices
        if correct_answer_results[idx] is not None
        and wrong_answer_results[idx] is not None
    ]
    correct = np.array(
        [correct_answer_results[idx]["prob"] for idx in filtered_indices]
    )
    all_wrong = np.array(
        [wrong_answer_results[idx]["prob"] for idx in filtered_indices]
    )
    wrong = np.sum(all_wrong, axis=tuple(range(1, all_wrong.ndim)))
    probs = correct / (correct + wrong + 1e-10)

    value_by_index = dict(zip(correct_indices, [{"prob": val} for val in probs]))
    return {"agg_value": np.mean(probs), "value_by_index": value_by_index}


@unlearning_metric(name="rouge")
def rouge(model, **kwargs):
    """Calculate ROUGE metrics and return the aggregated value along with per-index scores."""
    tokenizer = kwargs["tokenizer"]
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    generation_args = kwargs["generation_args"]
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    fun_args = {"tokenizer": tokenizer, "generation_args": generation_args}
    scores_by_index = run_batchwise_evals(
        model,
        dataloader,
        eval_text_similarity,
        fun_args,
        "Calculating text similarity",
    )
    rouge_values = np.array(
        [
            evals[kwargs["rouge_type"]]
            for evals in scores_by_index.values()
            if evals[kwargs["rouge_type"]] is not None
        ]
    )
    rouge_values = aggregate_to_1D(rouge_values)
    return {
        "agg_value": np.mean(rouge_values),
        "value_by_index": scores_by_index,
    }


@unlearning_metric(name="truth_ratio")
def truth_ratio(model, **kwargs):
    """Compute the truth ratio, aggregating false/true scores, and
    return the aggregated value."""

    # Forget data: It is better if false and true are equally likely,
    # i.e., tr=false/true is closest to 1.
    def closer_to_1_better(arr):
        return np.mean(np.minimum(arr, 1 / (arr + 1e-10)))

    # Non-forget data: It is better if tr=false/true is lower, i.e.,
    # 1-tr is higher.
    def true_better(arr):
        return np.mean(np.maximum(0, 1 - arr))

    if kwargs["aggregator"] == "closer_to_1_better":
        aggregator = closer_to_1_better
    elif kwargs["aggregator"] == "true_better":
        aggregator = true_better
    else:
        raise ValueError(f"Invalid truth ratio aggregator: {kwargs['aggregator']}")

    correct_answer_results = kwargs["pre_compute"]["correct"]["value_by_index"]
    wrong_answer_results = kwargs["pre_compute"]["wrong"]["value_by_index"]

    correct_indices = list(correct_answer_results.keys())
    wrong_indices = list(wrong_answer_results.keys())
    assert correct_indices == wrong_indices

    # Filter out None values from both correct and wrong answers
    filtered_indices = [
        idx
        for idx in correct_indices
        if correct_answer_results[idx] is not None
        and wrong_answer_results[idx] is not None
    ]
    correct_avg_losses = [
        correct_answer_results[idx]["avg_loss"] for idx in filtered_indices
    ]
    wrong_avg_losses = [
        wrong_answer_results[idx]["avg_loss"] for idx in filtered_indices
    ]

    correct_avg_losses = aggregate_to_1D(np.array(correct_avg_losses))
    wrong_avg_losses = aggregate_to_1D(np.array(wrong_avg_losses))

    correct_prob = np.exp(-correct_avg_losses)
    wrong_prob = np.exp(-wrong_avg_losses)

    truth_ratios = wrong_prob / (correct_prob + 1e-10)
    value_by_index = dict(
        zip(correct_indices, [{"score": val} for val in truth_ratios])
    )
    truth_ratio_stats = np.array([evals["score"] for evals in value_by_index.values()])
    forget_tr_avg = aggregator(truth_ratio_stats)
    return {"agg_value": forget_tr_avg, "value_by_index": value_by_index}


@unlearning_metric(name="exact_memorization")
def exact_memorization(model, **kwargs):
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    def _exact_memorization(model, batch):
        log_probs_batch, labels_batch = tokenwise_vocab_logprobs(
            model, batch, grad=False, return_labels=True
        )
        em_batch = []
        for log_probs, labels in zip(log_probs_batch, labels_batch):
            valid_len = len(labels)
            if valid_len == 0:
                # Rarely, tokenization can result in a mismatch with no valid target
                # tokens for loss computation (see preprocess_chat_instance() for
                # reference). Since this condition makes no sense in terms of
                # computing EM, we just choose to set EM=None
                logger.warning(
                    "EM score for an instance is marked None, due to "
                    "tokenization issues that resulted in no valid target tokens."
                )
                em_batch.append({"score": None})
            else:
                preds = torch.argmax(log_probs, dim=-1)
                em_score = (preds == labels).sum() / valid_len
                em_batch.append({"score": em_score.item()})
        return em_batch

    fun_args = {}
    scores_by_index = run_batchwise_evals(
        model, dataloader, _exact_memorization, fun_args, "Calculating EM"
    )
    em_values = np.array(
        [
            evals["score"]
            for evals in scores_by_index.values()
            if evals["score"] is not None
        ]
    )
    em_values = aggregate_to_1D(em_values)
    return {"agg_value": np.mean(em_values), "value_by_index": scores_by_index}


@unlearning_metric(name="probabilistic_extraction")
def probabilistic_extraction(model, **kwargs):
    """Compute the (n, p)-discoverable extraction rate.

    For each target suffix, this computes its exact teacher-forced probability
    under the configured sampling scheme. It then derives the probability of
    observing that suffix at least once in ``num_queries`` independent queries.
    The aggregate is the fraction of targets whose extraction probability is at
    least ``probability_threshold``. Both operating-point arguments may be lists
    to evaluate an (n, p) grid with one model forward pass.
    """
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    num_queries_arg = kwargs["num_queries"]
    probability_threshold_arg = kwargs["probability_threshold"]
    temperature = kwargs.get("temperature", 1.0)
    top_k = kwargs.get("top_k")
    top_p = kwargs.get("top_p")
    prefix_length = kwargs.get("prefix_length", 0)
    suffix_length = kwargs.get("suffix_length")

    query_arg_is_scalar = np.isscalar(num_queries_arg)
    threshold_arg_is_scalar = np.isscalar(probability_threshold_arg)
    num_queries_values = (
        [num_queries_arg] if query_arg_is_scalar else list(num_queries_arg)
    )
    probability_thresholds = (
        [probability_threshold_arg]
        if threshold_arg_is_scalar
        else list(probability_threshold_arg)
    )
    if not num_queries_values:
        raise ValueError("num_queries must contain at least one value")
    if not probability_thresholds:
        raise ValueError("probability_threshold must contain at least one value")
    for value in num_queries_values:
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError("num_queries values must be positive integers")
    for value in probability_thresholds:
        if not 0 <= value <= 1:
            raise ValueError(
                "probability_threshold values must be in the interval [0, 1]"
            )
    if isinstance(prefix_length, bool) or int(prefix_length) != prefix_length:
        raise ValueError("prefix_length must be a non-negative integer")
    if prefix_length < 0:
        raise ValueError("prefix_length must be a non-negative integer")
    if suffix_length is not None and (
        isinstance(suffix_length, bool)
        or int(suffix_length) != suffix_length
        or suffix_length < 1
    ):
        raise ValueError("suffix_length must be a positive integer or null")

    num_queries_values = [int(value) for value in num_queries_values]
    probability_thresholds = [float(value) for value in probability_thresholds]
    prefix_length = int(prefix_length)
    suffix_length = None if suffix_length is None else int(suffix_length)
    sweep_operating_points = not (query_arg_is_scalar and threshold_arg_is_scalar)
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    def _probabilistic_extraction(model, batch):
        log_probs_batch, labels_batch = tokenwise_vocab_logprobs(
            model, batch, grad=False, return_labels=True
        )
        extraction_batch = []
        for log_probs, labels in zip(log_probs_batch, labels_batch):
            target_span = _select_target_span(
                log_probs,
                labels,
                prefix_length=prefix_length,
                suffix_length=suffix_length,
            )
            if target_span is None:
                logger.warning(
                    "Probabilistic extraction for an instance is marked None, due "
                    "to insufficient valid tokens for the requested prefix/suffix "
                    "split."
                )
                extraction_batch.append(
                    {
                        "log_probability": None,
                        "single_query_probability": None,
                        "extraction_probability": None,
                        "is_extractable": None,
                        "num_target_tokens": 0,
                    }
                )
                continue

            target_log_probs, target_labels = target_span
            decoder_log_probs = _sampling_log_probs(
                target_log_probs,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            target_log_probs = torch.gather(
                decoder_log_probs, dim=-1, index=target_labels.unsqueeze(-1)
            ).squeeze(-1)
            log_probability = target_log_probs.double().sum().item()
            single_query_probability = math.exp(log_probability)
            extraction_probabilities = {
                str(num_queries): _probability_at_least_once(
                    log_probability, num_queries
                )
                for num_queries in num_queries_values
            }
            if sweep_operating_points:
                is_extractable = {
                    _operating_point_key(num_queries, threshold): (
                        extraction_probabilities[str(num_queries)] >= threshold
                    )
                    for num_queries in num_queries_values
                    for threshold in probability_thresholds
                }
                extraction_probability = extraction_probabilities
            else:
                extraction_probability = extraction_probabilities[
                    str(num_queries_values[0])
                ]
                is_extractable = extraction_probability >= probability_thresholds[0]
            extraction_batch.append(
                {
                    "log_probability": log_probability,
                    "single_query_probability": single_query_probability,
                    "extraction_probability": extraction_probability,
                    "is_extractable": is_extractable,
                    "num_target_tokens": len(target_labels),
                }
            )
        return extraction_batch

    scores_by_index = run_batchwise_evals(
        model,
        dataloader,
        _probabilistic_extraction,
        {},
        "Calculating probabilistic extraction",
    )
    if sweep_operating_points:
        operating_point_keys = [
            _operating_point_key(num_queries, threshold)
            for num_queries in num_queries_values
            for threshold in probability_thresholds
        ]
        extractable_values = {key: [] for key in operating_point_keys}
        for evals in scores_by_index.values():
            values = evals["is_extractable"]
            values = values if isinstance(values, list) else [values]
            for value in values:
                if value is None:
                    continue
                for key in operating_point_keys:
                    extractable_values[key].append(float(value[key]))
        agg_value = {key: np.mean(values) for key, values in extractable_values.items()}
    else:
        extractable_values = []
        for evals in scores_by_index.values():
            values = np.asarray(evals["is_extractable"], dtype=object).reshape(-1)
            extractable_values.extend(
                float(value) for value in values if value is not None
            )
        agg_value = np.mean(extractable_values)

    return {
        "agg_value": agg_value,
        "value_by_index": scores_by_index,
    }


@unlearning_metric(name="extraction_strength")
def extraction_strength(model, **kwargs):
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    def _extraction_strength(model, batch):
        log_probs_batch, labels_batch = tokenwise_vocab_logprobs(
            model, batch, grad=False, return_labels=True
        )
        es_batch = []
        for log_probs, labels in zip(log_probs_batch, labels_batch):
            valid_len = len(labels)
            preds = torch.argmax(log_probs, dim=-1)
            for k in range(valid_len):
                suff_preds = preds[k:]
                suff_labels = labels[k:]
                if torch.equal(suff_preds, suff_labels):
                    break
            if valid_len == 0:
                # Rarely, tokenization can result in a mismatch with no valid target
                # tokens for loss computation (see preprocess_chat_instance() for
                # reference). Since this condition makes no sense in terms of
                # computing ES, we just choose to set ES=None
                logger.warning(
                    "ES score for an instance is marked None, due to "
                    "tokenization issues that resulted in no valid target tokens."
                )
                es_batch.append({"score": 0})
            else:
                es_score = 1 - (k / valid_len)
                es_batch.append({"score": es_score})
        return es_batch

    fun_args = {}
    scores_by_index = run_batchwise_evals(
        model, dataloader, _extraction_strength, fun_args, "Calculating ES"
    )
    es_values = np.array(
        [
            evals["score"]
            for evals in scores_by_index.values()
            if evals["score"] is not None
        ]
    )
    es_values = aggregate_to_1D(es_values)
    return {"agg_value": np.mean(es_values), "value_by_index": scores_by_index}

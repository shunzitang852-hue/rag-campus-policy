from scripts.export_final_experiment_analysis import classify_question


def test_error_taxonomy_distinguishes_ranking_and_recall_failures() -> None:
    assert classify_question({"first_relevant_rank": 1}) == "success_rank1"
    assert classify_question({"first_relevant_rank": 3}) == "ranking_not_first"
    assert classify_question({"first_relevant_rank": None, "candidate_pool_results": [{}] * 5 + [{"is_relevant": True}]}) == "relevant_below_cutoff"
    assert classify_question({"first_relevant_rank": None, "candidate_pool_results": [], "source_policy_hit": True}) == "same_policy_wrong_chunk"
    assert classify_question({"first_relevant_rank": None, "candidate_pool_results": [], "source_policy_hit": False}) == "candidate_recall_failure"


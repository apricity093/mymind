"""Train-only optional-fact amendment; frozen standards and grading stay unchanged."""
from copy import deepcopy

from benchmarks.scripts.balanced_policy_design import profile_example
from benchmarks.scripts.balanced_policy_events import read_event
from benchmarks.scripts.balanced_policy_candidates import classify, actual_subtype
from benchmarks.scripts.balanced_policy_content import expectation_support
from rl.tasks import Expectation

REVISION = 'balanced-policy-v1-train-optional-facts-r5-candidate'


def amend_standard(task, standard):
    states, evidence = read_event(task)
    sid, _ = classify(task.role, states)
    generated = profile_example(sid, actual_subtype(sid, states), states)
    result = standard.model_copy(deep=True)
    additions = []
    for expected in generated.expects:
        alt = expected['alternatives'][0]
        if expected['section'] != 'facts' or alt.get('type') not in {
                'timing_policy_covered', 'whitelist_actor', 'enablement_actor', 'token_expired'} or expected['required']:
            continue
        if any(alt in e.alternatives and e.section == 'facts' for e in result.expectations):
            continue
        expected = deepcopy(expected)
        # Actor labels define policy responsibilities, not current user state.
        if alt['type'] in {'whitelist_actor', 'enablement_actor'}:
            expected['supports'] = [['config_actors']]
        else:
            expected['supports'] = expectation_support(expected, evidence)
        result.expectations.append(Expectation.model_validate(expected))
        additions.append(expected)
    return result, additions

from scripts.prepare_stratified_dataset import assign_grouped_splits


def test_grouped_split_is_deterministic_and_keeps_groups_together():
    records = [
        {"labels": {"plastic"}},
        {"labels": {"plastic"}},
        {"labels": {"paper"}},
        {"labels": {"metal"}},
        {"labels": {"glass"}},
        {"labels": {"cardboard"}},
        {"labels": {"biodegradable"}},
    ]
    groups = [[0, 1], [2], [3], [4], [5], [6]]
    first = assign_grouped_splits(records, groups, seed=42)
    second = assign_grouped_splits(records, groups, seed=42)

    assert first == second
    assert all(set(indices).isdisjoint(set(next_indices)) for split, indices in first.items() for other, next_indices in first.items() if split != other)
    assert any({0, 1}.issubset(set(indices)) for indices in first.values())
    assert sorted(index for indices in first.values() for index in indices) == list(range(len(records)))

import pytest

from fdna.nn.evaluation import summarize_seeded_family


def test_summarize_seeded_family_requires_every_declared_seed_and_bootstraps_ops_by_seed():
    per_op_rprec = {
        600: {"g2_fixed": 0.90, "gbm": 0.86,
              "ordered_perm_avg_seed0": 0.88, "ordered_perm_avg_seed1": 0.89, "ordered_perm_avg_seed2": 0.90,
              "deepsets_seed0": 0.80, "deepsets_seed1": 0.82, "deepsets_seed2": 0.84},
        601: {"g2_fixed": 0.70, "gbm": 0.69,
              "ordered_perm_avg_seed0": 0.72, "ordered_perm_avg_seed1": 0.74, "ordered_perm_avg_seed2": 0.76,
              "deepsets_seed0": 0.60, "deepsets_seed1": 0.64, "deepsets_seed2": 0.68},
    }

    summary = summarize_seeded_family(
        "deepsets", (0, 1, 2), per_op_rprec,
        references=("g2_fixed", "gbm", "ordered_perm_avg"),
        bootstrap_kwargs={"n_boot": 200, "seed": 9},
    )

    assert summary["arms"] == ["deepsets_seed0", "deepsets_seed1", "deepsets_seed2"]
    assert summary["per_seed_mean"] == pytest.approx({"seed0": 0.70, "seed1": 0.73, "seed2": 0.76})
    assert summary["mean"] == pytest.approx(0.73)
    assert summary["std"] == pytest.approx(0.0244948974)
    assert summary["boot_vs"]["g2_fixed"]["matrix_shape"] == [2, 3]
    assert summary["boot_vs"]["g2_fixed"]["mean"] == pytest.approx(-0.07)
    assert summary["boot_vs"]["gbm"]["mean"] == pytest.approx(-0.045)
    assert summary["boot_vs"]["ordered_perm_avg"]["mean"] == pytest.approx(-0.085)

    broken = {op: dict(scores) for op, scores in per_op_rprec.items()}
    del broken[601]["deepsets_seed2"]
    with pytest.raises(KeyError, match="deepsets_seed2"):
        summarize_seeded_family("deepsets", (0, 1, 2), broken, references=("g2_fixed",))

"""Feature module generators and loader for modular feature architecture.

Each feature module is a parquet file in ``data/Level_1_Features/modules/``
with grain ``(date, session, orb_tf, breakout_ts)`` — the breakout-event level.
All features are side-independent (same for rev and cont rows).

Usage::

    from pipeline.orb_ml.features.modules.loader import load_features_from_modules

    dm = pd.read_parquet("data/Level_2_Datamart/training_datamart_orb.parquet")
    dm = load_features_from_modules(modules_dir, dm)
"""

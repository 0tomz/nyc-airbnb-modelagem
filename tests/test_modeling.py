import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from nyc_airbnb_eda.modeling import (
    build_log_target_pipeline,
    build_model_pipeline,
    display_feature_name,
    get_preprocessed_feature_names,
    original_feature_name,
)


def make_listings(n: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    data = pd.DataFrame({
        "reviews_per_month": np.where(rng.random(n) < 0.3, np.nan, rng.random(n) * 3),
        "minimum_nights": rng.integers(1, 30, n),
        "number_of_reviews": rng.integers(0, 100, n),
        "calculated_host_listings_count": rng.integers(1, 5, n),
        "latitude": 40.6 + rng.random(n) * 0.2,
        "longitude": -74.0 + rng.random(n) * 0.2,
        "availability_365": rng.integers(0, 365, n),
        "neighbourhood_group": rng.choice(["Manhattan", "Brooklyn"], n),
        "neighbourhood": rng.choice(["Chelsea", "Williamsburg"], n),
        "room_type": rng.choice(["Private room", "Entire home/apt"], n),
    })
    price = 50 + 100 * (data["room_type"] == "Entire home/apt") + rng.random(n) * 20
    return data, price


def test_log_target_pipeline_predicts_in_dollars_and_never_negative() -> None:
    X, y = make_listings()
    model = build_log_target_pipeline(Ridge(alpha=1.0)).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (len(X),)
    assert (pred >= 0).all()
    # Predictions are back on the dollar scale, not on log1p(price).
    assert abs(pred.mean() - y.mean()) < 0.5 * y.mean()


def test_model_pipeline_fits_preprocessor_inside_pipeline() -> None:
    X, y = make_listings()
    pipe = build_model_pipeline(DummyRegressor(strategy="median")).fit(X, y)
    assert set(pipe.named_steps) == {"preprocessor", "model"}
    assert np.allclose(pipe.predict(X), y.median())


def test_feature_names_match_transformed_columns() -> None:
    X, y = make_listings()
    pipe = build_model_pipeline(Ridge()).fit(X, y)
    pre = pipe.named_steps["preprocessor"]
    names = get_preprocessed_feature_names(pre)
    assert len(names) == pre.transform(X).shape[1]


def test_feature_name_helpers() -> None:
    assert original_feature_name("categorical__neighbourhood_group_Manhattan") == "neighbourhood_group"
    assert original_feature_name("categorical__neighbourhood_Tribeca") == "neighbourhood"
    assert original_feature_name("numeric__longitude") == "longitude"
    assert display_feature_name("categorical__neighbourhood_Tribeca") == "neighbourhood = Tribeca"
    assert display_feature_name("categorical__room_type_Private room") == "room_type = Private room"
    assert display_feature_name("skewed__minimum_nights") == "minimum_nights"

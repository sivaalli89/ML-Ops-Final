import numpy as np

from name_that_llm.features import StyleFeatures, make_skeleton


def test_skeleton_matches_the_documented_example():
    common = {"the", "it", "was"}
    assert make_skeleton("The cat sat; it was warm.", common) == "the W W ; it was W ."


def test_skeleton_keeps_punctuation_and_hides_rare_words():
    skeleton = make_skeleton("Zyxwv, qwerty!", {"the"})
    assert skeleton == "W , W !"


def test_style_features_have_the_same_columns_for_new_text(toy_data):
    features = StyleFeatures(n_common_words=10).fit(toy_data["text"])
    train_x = features.transform(toy_data["text"])
    new_x = features.transform(["a brand new passage ; with some words ."])
    assert train_x.shape[1] == new_x.shape[1]
    assert train_x.dtype == np.float32
    assert np.isfinite(new_x).all()

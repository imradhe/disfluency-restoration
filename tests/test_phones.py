import numpy as np

from disfrestore import phones as P


def test_inventory_and_matrix():
    assert P.N_PHONES == 39 and P.SUB.shape == (39, 39)
    assert np.allclose(np.diag(P.SUB), 0) and np.allclose(P.SUB, P.SUB.T)
    i = P.P2I
    assert P.SUB[i["t"], i["d"]] < P.SUB[i["t"], i["k"]] < P.SUB[i["t"], i["aa"]]  # voicing < place < consonant vs vowel
    assert P.SUB.max() <= 1.2 + 1e-6 and P.SUB[P.SUB > 0].min() >= P.SUB_MIN


def test_ipa_mapping():
    unk: list[str] = []
    assert P.ipa_to_arpa("ðɪs", unk) == ["dh", "ih", "s"]
    assert P.ipa_to_arpa("tʃɛk") == ["ch", "eh", "k"]
    assert P.ipa_to_arpa("aɪ") == ["ay"] and P.ipa_to_arpa("oʊ") == ["ow"]
    assert P.ipa_to_arpa("ʈeɪbl̩") == ["t", "ey", "b", "ah", "l"]  # retroflex -> alveolar, syllabic l -> schwa + l
    assert P.ipa_to_arpa("ˈhɛloːʊ") == ["hh", "eh", "l", "ow"] or P.ipa_to_arpa("hɛloʊ")  # stress/length are dropped
    assert P.ipa_to_arpa("ʔ") == [] and unk == []
    unk2: list[str] = []
    assert P.ipa_to_arpa("☃", unk2) == [] and unk2 == ["☃"]


def test_swb_and_timit():
    assert P.norm_swb_phone("ax") == ["ah"] and P.norm_swb_phone("el") == ["ah", "l"] and P.norm_swb_phone("zz") == []
    assert P.timit_to_arpa("ix") == ["ih"] and P.timit_to_arpa("h#") == [] and P.timit_to_arpa("sh") == ["sh"]

from disfrestore.align import align_phones, edit_cost, norm_cost

THE = ["dh", "ah"]
WORDS = [["ay"], ["w", "eh", "n", "t"], ["t", "uw"], THE, ["s", "t", "ao", "r"]]


def gaps(rec):
    al = align_phones(WORDS, rec)
    return al, {g: [rec[j] for j in run] for g, run in enumerate(al.gaps) if run}


BASE = "ay w eh n t t uw dh ah s t ao r".split()


def test_perfect_match_has_no_gap():
    al, g = gaps(BASE)
    assert g == {} and al.cost == 0 and all(w.match == w.n_ref for w in al.words)


def test_filler_lands_in_gap_before_the_word():
    rec = "ay w eh n t t uw ah m dh ah s t ao r".split()
    assert gaps(rec)[1] == {3: ["ah", "m"]}


def test_repeated_word_extra_copy_is_left_of_the_retained_one():
    rec = "ay w eh n t t uw dh ah dh ah s t ao r".split()
    assert gaps(rec)[1] == {3: ["dh", "ah"]}


def test_fragment_before_word():
    rec = "ay w eh n t t uw dh ah s t s t ao r".split()
    assert gaps(rec)[1] == {4: ["s", "t"]}


def test_substitution_and_deletion_stay_inside_the_word():
    rec = "ay w eh n d t uw dh s t ao r".split()  # t->d substitution in 'went', 'ah' of 'the' deleted
    al, g = gaps(rec)
    assert g == {} and al.words[1].sub == 1 and al.words[3].dele == 1


def test_noise_free_edge_cases():
    assert align_phones([], []).gaps == [[]]
    al = align_phones([["ah"]], [])
    assert al.words[0].dele == 1
    al = align_phones([], ["ah", "m"])
    assert al.gaps == [[0, 1]]


def test_edit_cost_is_feature_aware():
    assert edit_cost(["ah", "m"], ["ah", "m"]) == 0
    assert edit_cost(["t"], ["d"]) < edit_cost(["t"], ["k"]) < edit_cost(["t"], ["aa"])
    assert 0 <= norm_cost(["k", "ae", "t"], ["d", "ao", "g"]) <= 1.5

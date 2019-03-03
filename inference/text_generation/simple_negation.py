import os
import numpy as np
from vocab import male_names, female_names, cities_and_states, countries
from vocab_pt import male_names_pt, female_names_pt, cities_pt, countries_pt
from util import get_new_item, get_n_different_items
from util import vi, not_vi, vi_pt, not_vi_pt
from util import create_csv_contradiction, create_csv_entailment
from util import create_csv_NLI


def entailment_instance_1(person_list,
                          place_list,
                          n,
                          vi_function,
                          not_vi_function):
    """
    $P:= pm V(x_1, y_1) , dots, pm V(x_i, y_i), dots, pm V(x_n, y_n)$
    $H:= pm V(x_i, y_i)$
    """
    Subjects = get_n_different_items(person_list, n)
    people_O = [get_new_item(Subjects, person_list) for _ in range(n)]
    places = get_n_different_items(place_list, n)
    Objects = get_n_different_items(people_O + places, n)
    fs = np.random.choice([vi_function, not_vi_function], n)
    sentence1 = [f(x, y) for f, x, y in zip(fs, Subjects, Objects)]
    id_ = np.random.choice(len(Subjects))
    sentence2 = sentence1[id_]
    sentence1 = ", ".join(sentence1)
    label = "entailment"
    people_O = list(set(Objects).intersection(people_O))
    places = list(set(Objects).intersection(places))
    people = ", ".join(Subjects + people_O)
    Subjects = ", ".join(Subjects)
    Objects = ", ".join(Objects)
    places = ", ".join(places)

    return sentence1, sentence2, label, Subjects, Objects, id_, people, places


def neutral_instance_1(person_list,
                       place_list,
                       n,
                       vi_function,
                       not_vi_function):
    """
    $P:= pm V(x_1, y_1) , dots, pm V(x_i, y_i), dots, pm V(x_n, y_n)$
    $H:= pm V(y_i, x_i)$
    """
    Subjects = get_n_different_items(person_list, n)
    people_O = [get_new_item(Subjects, person_list) for _ in range(n)]
    places = get_n_different_items(place_list, n)
    Objects = get_n_different_items(people_O + places, n)
    inter = len(set(Objects).intersection(people_O))
    if inter == 0:
        Objects[0] = people_O[0]
    np.random.shuffle(Objects)
    id_ = np.random.choice(len(Subjects))
    while Objects[id_] not in people_O:
        id_ = np.random.choice(len(Subjects))
    fs = np.random.choice([vi_function, not_vi_function], n)
    sentence1 = ", ".join([f(x, y) for f, x, y in zip(fs, Subjects, Objects)])
    f2 = np.random.choice([vi_function, not_vi_function])
    sentence2 = f2(Objects[id_], Subjects[id_])
    label = "neutral"
    people_O = list(set(Objects).intersection(people_O))
    people = ", ".join(Subjects + people_O)
    Subjects.append(Objects[id_])
    Objects.append(Subjects[id_])
    places = list(set(Objects).intersection(places))
    Subjects = ", ".join(Subjects)
    Objects = ", ".join(Objects)
    places = ", ".join(places)

    return sentence1, sentence2, label, Subjects, Objects, id_, people, places


def neutral_instance_2(person_list,
                       place_list,
                       n,
                       vi_function,
                       not_vi_function):
    """
    $P:= pm V(x_1, y_1) , dots, pm V(x_n, y_n)$
    $H:= pm V(x^{*}, y^{*})$
    where $x^{*}  not in x_1, dots, x_n $ or $y^{*}  not in y_1, dots, y_n$.   # noqa
    """
    Subjects = get_n_different_items(person_list, n)
    people_O = [get_new_item(Subjects, person_list) for _ in range(n)]
    places = get_n_different_items(place_list, n)
    Objects = get_n_different_items(people_O + places, n)
    fs = np.random.choice([vi_function, not_vi_function], n)
    sentence1 = [f(x, y) for f, x, y in zip(fs, Subjects, Objects)]
    sentence1 = ", ".join(sentence1)
    id_ = -1
    fs2 = np.random.choice([vi_function, not_vi_function])
    Subject2 = get_new_item(Subjects + people_O, person_list)
    Object2 = [get_new_item(Subjects + people_O + [Subject2], person_list)]
    place2 = get_new_item(places, place_list)
    Object2 += [place2]
    Object2 = np.random.choice(Object2)

    sentence2_1 = fs2(np.random.choice(Subjects), Object2)
    sentence2_2 = fs2(Subject2, np.random.choice(Objects))
    sentence2_3 = fs2(Subject2, Object2)

    people_O = list(set(Objects).intersection(people_O))

    dice = np.random.choice([1, 2, 3])
    if dice == 1:
        sentence2 = sentence2_1
        people = ", ".join(Subjects + people_O)
        places = list(set(Objects + [Object2]).intersection(places + [place2]))
        Subjects = ", ".join(Subjects)
        Objects = ", ".join(Objects + [Object2])

    elif dice == 2:
        sentence2 = sentence2_2
        people = ", ".join(Subjects + people_O + [Subject2])
        places = list(set(Objects).intersection(places))
        Subjects = ", ".join(Subjects + [Subject2])
        Objects = ", ".join(Objects)

    else:
        sentence2 = sentence2_3
        people = ", ".join(Subjects + people_O + [Subject2])
        places = list(set(Objects + [Object2]).intersection(places + [place2]))
        Subjects = ", ".join(Subjects + [Subject2])
        Objects = ", ".join(Objects + [Object2])

    places = ", ".join(places)

    label = "neutral"

    return sentence1, sentence2, label, Subjects, Objects, id_, people, places


def contradiction_instance_1(person_list,
                             place_list,
                             n,
                             vi_function,
                             not_vi_function):
    """
    $P:= pm V(x_1, y_1) , dots, V(x_i, y_i), dots, pm V(x_n, y_n)$
    $H:= not V(y_i, x_i)$
    """
    Subjects = get_n_different_items(person_list, n)
    people_O = [get_new_item(Subjects, person_list) for _ in range(n)]
    places = get_n_different_items(place_list, n)
    Objects = get_n_different_items(people_O + places, n)

    fs = []
    while vi_function not in fs:
        fs = np.random.choice([vi_function, not_vi_function], n)

    id_ = 0
    while fs[id_] != vi_function:
        id_ = np.random.choice(len(fs))

    sentence1 = ", ".join([f(x, y) for f, x, y in zip(fs, Subjects, Objects)])
    sentence2 = not_vi_function(Subjects[id_], Objects[id_])
    label = "contradiction"
    people_O = list(set(Objects).intersection(people_O))
    places = list(set(Objects).intersection(places))
    people = ", ".join(Subjects + people_O)
    Subjects = ", ".join(Subjects)
    Objects = ", ".join(Objects)
    places = ", ".join(places)

    return sentence1, sentence2, label, Subjects, Objects, id_, people, places


def i2eng(f):
    return lambda x, y, z: f(x, y, z, vi_function=vi, not_vi_function=not_vi)  # noqa


def i2pt(f):
    return lambda x, y, z: f(x, y, z, vi_function=vi_pt, not_vi_function=not_vi_pt)  # noqa


entailment_instances = [entailment_instance_1]
neutral_instances = [neutral_instance_1, neutral_instance_2]
contradiction_instances = [contradiction_instance_1]


entailment_instances_eng = list(map(i2eng, entailment_instances))
neutral_instances_eng = list(map(i2eng, neutral_instances))
contradiction_instances_eng = list(map(i2eng, contradiction_instances))

entailment_instances_pt = list(map(i2pt, entailment_instances))
neutral_instances_pt = list(map(i2pt, neutral_instances))
contradiction_instances_pt = list(map(i2pt, contradiction_instances))

if __name__ == '__main__':

    # call this script in the main folder, i.e., type
    # python inference/text_generation/simple_negation.py

    cwd = os.getcwd()
    base_path_NLI = os.path.join(cwd, "data", "NLI")
    base_path_RTE = os.path.join(cwd, "data", "RTE")
    base_path_CD = os.path.join(cwd, "data", "CD")

    # english

    # CD
    create_csv_contradiction(out_path=os.path.join(base_path_CD,
                                                   "simple_negation_train.csv"),  # noqa
                             size=10000,
                             positive_instances_list=contradiction_instances_eng,  # noqa
                             negative_instances_list=entailment_instances_eng + neutral_instances_eng,  # noqa
                             person_list=male_names,
                             place_list=countries,
                             n=12,
                             min_n=2)

    create_csv_contradiction(out_path=os.path.join(base_path_CD,
                                                   "simple_negation_test.csv"),  # noqa
                             size=1000,
                             positive_instances_list=contradiction_instances_eng,  # noqa
                             negative_instances_list=entailment_instances_eng + neutral_instances_eng,  # noqa
                             person_list=female_names,
                             place_list=cities_and_states,
                             n=12,
                             min_n=2)

    # RTE
    create_csv_entailment(out_path=os.path.join(base_path_RTE,
                                                "simple_negation_train.csv"),  # noqa
                          size=10000,
                          positive_instances_list=entailment_instances_eng,  # noqa
                          negative_instances_list=contradiction_instances_eng + neutral_instances_eng,  # noqa
                          person_list=male_names,
                          place_list=countries,
                          n=12,
                          min_n=2)

    create_csv_entailment(out_path=os.path.join(base_path_RTE,
                                                "simple_negation_test.csv"),  # noqa
                          size=1000,
                          positive_instances_list=entailment_instances_eng,  # noqa
                          negative_instances_list=contradiction_instances_eng + neutral_instances_eng,  # noqa
                          person_list=female_names,
                          place_list=cities_and_states,
                          n=12,
                          min_n=2)
    # NLI
    create_csv_NLI(out_path=os.path.join(base_path_NLI,
                                         "simple_negation_train.csv"),  # noqa,
                   size=10008,
                   entailment_instances_list=entailment_instances_eng,
                   neutral_instances_list=neutral_instances_eng,
                   contradiction_instances_list=contradiction_instances_eng,
                   person_list=male_names,
                   place_list=countries,
                   n=12,
                   min_n=2)

    create_csv_NLI(out_path=os.path.join(base_path_NLI,
                                         "simple_negation_test.csv"),  # noqa,
                   size=1000,
                   entailment_instances_list=entailment_instances_eng,
                   neutral_instances_list=neutral_instances_eng,
                   contradiction_instances_list=contradiction_instances_eng,
                   person_list=female_names,
                   place_list=cities_and_states,
                   n=12,
                   min_n=2)

    # portuguese

    # CD
    create_csv_contradiction(out_path=os.path.join(base_path_CD,
                                                   "simple_negation_pt_train.csv"),  # noqa
                             size=10000,
                             positive_instances_list=contradiction_instances_pt,  # noqa
                             negative_instances_list=entailment_instances_pt + neutral_instances_pt,  # noqa
                             person_list=male_names_pt,
                             place_list=countries_pt,
                             n=12,
                             min_n=2)

    create_csv_contradiction(out_path=os.path.join(base_path_CD,
                                                   "simple_negation_pt_test.csv"),  # noqa
                             size=1000,
                             positive_instances_list=contradiction_instances_pt,  # noqa
                             negative_instances_list=entailment_instances_pt + neutral_instances_pt,  # noqa
                             person_list=female_names_pt,
                             place_list=cities_pt,
                             n=12,
                             min_n=2)

    # RTE
    create_csv_entailment(out_path=os.path.join(base_path_RTE,
                                                "simple_negation_pt_train.csv"),  # noqa
                          size=10000,
                          positive_instances_list=entailment_instances_pt,  # noqa
                          negative_instances_list=contradiction_instances_pt + neutral_instances_pt,  # noqa
                          person_list=male_names_pt,
                          place_list=countries_pt,
                          n=12,
                          min_n=2)

    create_csv_entailment(out_path=os.path.join(base_path_RTE,
                                                "simple_negation_pt_test.csv"),  # noqa
                          size=1000,
                          positive_instances_list=entailment_instances_pt,  # noqa
                          negative_instances_list=contradiction_instances_pt + neutral_instances_pt,  # noqa
                          person_list=female_names_pt,
                          place_list=cities_pt,
                          n=12,
                          min_n=2)
    # NLI
    create_csv_NLI(out_path=os.path.join(base_path_NLI,
                                         "simple_negation_pt_train.csv"),  # noqa,
                   size=10008,
                   entailment_instances_list=entailment_instances_pt,
                   neutral_instances_list=neutral_instances_pt,
                   contradiction_instances_list=contradiction_instances_pt,
                   person_list=male_names_pt,
                   place_list=countries_pt,
                   n=12,
                   min_n=2)

    create_csv_NLI(out_path=os.path.join(base_path_NLI,
                                         "simple_negation_pt_test.csv"),  # noqa,
                   size=1000,
                   entailment_instances_list=entailment_instances_pt,
                   neutral_instances_list=neutral_instances_pt,
                   contradiction_instances_list=contradiction_instances_pt,
                   person_list=female_names,
                   place_list=cities_pt,
                   n=12,
                   min_n=2)

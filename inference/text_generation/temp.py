import numpy as np
from vocab import male_names, female_names, cities_and_states, countries
from vocab_pt import male_names_pt, female_names_pt, cities_pt, countries_pt
from util import get_new_item, get_n_different_items
from util import vi, not_vi, vi_pt, not_vi_pt, create_csv


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
    Subjects.append(Objects[id_])
    Objects.append(Subjects[id_])
    Subjects = ", ".join(Subjects)
    Objects = ", ".join(Objects)

    return sentence1, sentence2, label, Subjects, Objects, id_


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
    Object2 += [get_new_item(places, place_list)]
    Object2 = np.random.choice(Object2)

    sentence2_1 = fs2(np.random.choice(Subjects), Object2)
    sentence2_2 = fs2(Subject2, np.random.choice(Objects))
    sentence2_3 = fs2(Subject2, Object2)
    dice = np.random.choice([1, 2, 3])
    if dice == 1:
        sentence2 = sentence2_1
        Subjects = ", ".join(Subjects)
        Objects = ", ".join(Objects + [Object2])

    elif dice == 2:
        sentence2 = sentence2_2
        Subjects = ", ".join(Subjects + [Subject2])
        Objects = ", ".join(Objects)

    else:
        sentence2 = sentence2_3
        Subjects = ", ".join(Subjects + [Subject2])
        Objects = ", ".join(Objects + [Object2])

    label = "neutral"

    return sentence1, sentence2, label, Subjects, Objects, id_


# def contradiction_instance_1(person_list,
#                        place_list,
#                        n,
#                        vi_function,
#                        not_vi_function):
#     """
#     $P:= pm V(x_1, y_1) , dots, pm V(x_i, y_i), dots, pm V(x_n, y_n)\}$
#     $H:= pm V(y_i, x_i)$
#     """
#     Subjects = get_n_different_items(person_list, n)
#     people_O = [get_new_item(Subjects, person_list) for _ in range(n)]
#     places = get_n_different_items(place_list, n)
#     Objects = get_n_different_items(people_O + places, n)
#     inter = len(set(Objects).intersection(people_O))
#     if inter == 0:
#         Objects[0] = people_O[0]
#     np.random.shuffle(Objects)
#     id_ = np.random.choice(len(Subjects))
#     while Objects[id_] not in people_O:
#         id_ = np.random.choice(len(Subjects))
#     fs = np.random.choice([vi_function, not_vi_function], n)
#     sentence1 = ", ".join([f(x, y) for f, x, y in zip(fs, Subjects, Objects)])
#     f2 = np.random.choice([vi_function, not_vi_function])
#     sentence2 = f2(Objects[id_], Subjects[id_])
#     label = "contradiction"
#     Subjects.append(Objects[id_])
#     Objects.append(Subjects[id_])
#     Subjects = ", ".join(Subjects)
#     Objects = ", ".join(Objects)

#     return sentence1, sentence2, label, Subjects, Objects, id_


if __name__ == '__main__':

    a = entailment_instance_1(person_list=male_names,
                              place_list=countries,
                              n=2,
                              vi_function=vi,
                              not_vi_function=not_vi)
    print(a)

    b = entailment_instance_1(person_list=male_names_pt,
                           place_list=countries_pt,
                           n=2,
                           vi_function=vi_pt,
                           not_vi_function=not_vi_pt)
    print(b)

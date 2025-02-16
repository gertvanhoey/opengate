import awkward as ak
import opengate as gate
from tqdm import tqdm
from ..exception import fatal
import numpy as np
from collections import Counter
import sys
from collections import deque
import pandas as pd
from itertools import chain


def coincidences_sorter(
    singles_tree,
    time_window,
    policy,
    min_transaxial_distance,
    transaxial_plane,
    max_axial_distance,
    chunk_size=10000,
    return_type="dict",
):
    # Check validity of singles_tree
    required_branches = {
        "EventID",
        "GlobalTime",
        "PreStepUniqueVolumeID",
        "TotalEnergyDeposit",
        "PostPosition_X",
        "PostPosition_Y",
        "PostPosition_Z",
    }
    missing_branches = required_branches - set(singles_tree.keys())
    if missing_branches:
        if len(missing_branches) == 1:
            raise ValueError(
                f"Required branch {missing_branches.pop()} is missing in singles tree"
            )
        else:
            raise ValueError(
                f"Required branches {missing_branches} are missing in singles tree"
            )

    # Check validity of policy
    policy_functions = {
        "removeMultiples": remove_multiples,
        "takeAllGoods": take_all_goods,
        "takeWinnerOfGoods": take_winner_of_goods,
        "takeIfOnlyOneGood": take_if_only_one_good,
        "takeWinnerIfIsGood": take_winner_if_is_good,
        "takeWinnerIfAllAreGoods": take_winner_if_all_are_goods,
    }
    if policy not in policy_functions:
        raise ValueError(
            f"Unknown policy '{policy}', must be one of {policy_functions.keys()}"
        )

    # Check validity of return_type
    known_return_types = ["dict", "pd"]
    if return_type not in known_return_types:
        raise ValueError(
            f"Unknown return type '{return_type}', must be one of {known_return_types}"
        )

    queue = deque()
    coincidences = []
    for chunk in singles_tree.iterate(step_size=chunk_size):
        queue.append(ak.to_dataframe(chunk))
        if len(queue) > 1:
            coincidences.append(
                process_chunk(
                    queue, time_window, min_transaxial_distance, max_axial_distance
                )
            )
            queue.popleft()
    coincidences.append(
        process_chunk(queue, time_window, min_transaxial_distance, max_axial_distance)
    )
    all_coincidences = pd.concat(coincidences, axis=0)

    filtered_coincidences = policy_functions[policy](
        all_coincidences,
        min_transaxial_distance,
        transaxial_plane,
        max_axial_distance,
    )

    if return_type == "dict":
        return filtered_coincidences.to_dict()
    elif return_type == "pd":
        return filtered_coincidences


def process_chunk(queue, time_window, min_transaxial_distance, max_axial_distance):
    chunk = queue[0]
    next_chunk = queue[1] if len(queue) > 1 else None
    t1 = chunk["GlobalTime"]
    t1_min = np.min(t1)
    t1_max = np.max(t1)
    if next_chunk is not None:
        t2 = next_chunk["GlobalTime"]
        t2_min = np.min(t2)
        t2_max = np.max(t2)
        assert (
            t2_min - time_window > t1_min
            and t2_min - time_window < t1_max
            and t2_max > t1_max
        )
    coinc = run_coincidence_detection_in_chunk(
        chunk, time_window, min_transaxial_distance, max_axial_distance
    )
    if next_chunk is not None:
        coinc = coinc.loc[coinc["GlobalTime1"] < t2_min - time_window].reset_index(
            drop=True
        )
        singles_to_transfer = chunk.loc[
            chunk["GlobalTime"] >= t2_min - time_window
        ].reset_index(drop=True)
        queue[1] = pd.concat([singles_to_transfer, next_chunk], axis=0)
    return coinc


def run_coincidence_detection_in_chunk(
    chunk, time_window, min_transaxial_distance, max_axial_distance
):
    print(f"There are {len(chunk)} singles")
    chunk["VolumeIDHash"] = pd.util.hash_pandas_object(
        chunk["PreStepUniqueVolumeID"], index=False
    )
    time_np = chunk["GlobalTime"].to_numpy()
    time_np_sorted_indices = np.argsort(time_np)
    time_np = time_np[time_np_sorted_indices]
    indices1 = np.zeros((0,), dtype=np.int32)
    indices2 = np.zeros((0,), dtype=np.int32)
    offset = 1
    while True:
        delta_time = time_np[offset:] - time_np[:-offset]
        indices = np.nonzero(delta_time <= time_window)[0]
        if len(indices) == 0:
            break
        indices1 = np.concatenate((indices1, indices))
        indices2 = np.concatenate((indices2, indices + offset))
        offset += 1
    indices12 = np.vstack((indices1, indices2)).T
    indices12 = indices12[np.lexsort((indices12[:, 1], indices12[:, 0]))]
    rename_dict1 = {name: name + "1" for name in chunk.columns.tolist()}
    rename_dict2 = {name: name + "2" for name in chunk.columns.tolist()}
    coinc1 = (
        chunk.iloc[time_np_sorted_indices[indices12[:, 0]]]
        .rename(columns=rename_dict1)
        .reset_index(drop=True)
    )
    coinc2 = (
        chunk.iloc[time_np_sorted_indices[indices12[:, 1]]]
        .rename(columns=rename_dict2)
        .reset_index(drop=True)
    )
    interleaved_columns = list(
        chain(*zip(coinc1.columns, coinc2.columns))
    )  # Interleave column names
    coinc = pd.concat([coinc1, coinc2], axis=1)[interleaved_columns]
    coinc = coinc.loc[coinc["VolumeIDHash1"] != coinc["VolumeIDHash2"]].reset_index(
        drop=True
    )
    return coinc


def remove_multiples(
    coincidences, min_transaxial_distance, transaxial_plane, max_axial_distance
):
    filtered_coincidences = filter_multi(coincidences)
    return filtered_coincidences


def take_all_goods(
    coincidences, min_transaxial_distance, transaxial_plane, max_axial_distance
):
    filtered_coincidences = filter_goods(
        coincidences, min_transaxial_distance, transaxial_plane, max_axial_distance
    )
    return filtered_coincidences


def take_winner_of_goods(
    coincidences, min_transaxial_distance, transaxial_plane, max_axial_distance
):
    filtered_coincidences = filter_goods(
        coincidences, min_transaxial_distance, transaxial_plane, max_axial_distance
    )
    filtered_coincidences = filter_max_energy(filtered_coincidences)
    return filtered_coincidences


def take_if_only_one_good(
    coincidences, min_transaxial_distance, transaxial_plane, max_axial_distance
):
    filtered_coincidences = filter_goods(
        coincidences, min_transaxial_distance, transaxial_plane, max_axial_distance
    )
    filtered_coincidences = filter_multi(filtered_coincidences)
    return filtered_coincidences


def take_winner_if_is_good(
    coincidences, min_transaxial_distance, transaxial_plane, max_axial_distance
):
    filtered_coincidences = filter_max_energy(coincidences)
    filtered_coincidences = filter_goods(
        filtered_coincidences,
        min_transaxial_distance,
        transaxial_plane,
        max_axial_distance,
    )
    return filtered_coincidences


def take_winner_if_all_are_goods(
    coincidences, min_transaxial_distance, transaxial_plane, max_axial_distance
):
    filtered_coincidences = filter_goods(
        coincidences, min_transaxial_distance, transaxial_plane, max_axial_distance
    )

    counts = coincidences["index1"].value_counts()
    filtered_counts = filtered_coincidences["index2"].value_counts()

    # Find values where df1 has more occurrences than df2
    to_remove = counts[counts > filtered_counts].index

    # Remove groups from df1 where A is in to_remove
    filtered_coincidences = coincidences[~coincidences["index1"].isin(to_remove)]

    filtered_coincidences = filter_max_energy(filtered_coincidences)

    return filtered_coincidences


def filter_multi(coincidences):
    # Remove coincidences where the first single appears more than once
    filtered_coincidences = coincidences[~coincidences["index1"].duplicated(keep=False)]
    return filtered_coincidences


def filter_goods(
    coincidences, min_transaxial_distance, transaxial_plane, max_axial_distance
):
    td = transaxial_distance(coincidences, transaxial_plane)
    # >= is better
    filtered_coincidences = coincidences.loc[td > min_transaxial_distance].reset_index(
        drop=True
    )
    ad = axial_distance(filtered_coincidences, transaxial_plane)
    # <= is better
    filtered_coincidences = filtered_coincidences.loc[
        ad < max_axial_distance
    ].reset_index(drop=True)
    return filtered_coincidences


def filter_max_energy(coincidences):
    filtered_coincidences = coincidences.copy(deep=True)
    filtered_coincidences["TotalEnergyInCoincidence"] = (
        filtered_coincidences["TotalEnergyDeposit1"]
        + filtered_coincidences["TotalEnergyDeposit2"]
    )
    filtered_coincidences = filtered_coincidences.loc[
        filtered_coincidences.groupby("EventID1")["TotalEnergyInCoincidence"].idxmax()
    ]
    filtered_coincidences = filtered_coincidences.drop(
        columns=["TotalEnergyInCoincidence"]
    )
    return filtered_coincidences


def transaxial_distance(coincidences, transaxial_plane):
    if transaxial_plane not in ("xy", "yz", "xz"):
        raise ValueError(
            f"Invalid transaxial_plane: '{transaxial_plane}'. Expected one of 'xy', 'yz' or 'xz'."
        )
    a1, a2 = [
        coincidences[f"PostPosition_{transaxial_plane[0].upper()}{n}"].to_numpy()
        for n in (1, 2)
    ]
    b1, b2 = [
        coincidences[f"PostPosition_{transaxial_plane[1].upper()}{n}"].to_numpy()
        for n in (1, 2)
    ]
    return np.sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2)


def axial_distance(coincidences, transaxial_plane):
    if transaxial_plane not in ("xy", "yz", "xz"):
        raise ValueError(
            f"Invalid transaxial_plane: '{transaxial_plane}'. Expected one of 'xy', 'yz' or 'xz'."
        )
    axial_coordinate = (set("xyz") - set(transaxial_plane)).pop().upper()
    a1, a2 = [
        coincidences[f"PostPosition_{axial_coordinate}{n}"].to_numpy() for n in (1, 2)
    ]
    return np.abs(a1 - a2)

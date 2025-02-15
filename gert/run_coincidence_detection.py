import opengate as gate
import opengate.contrib.pet.siemensbiograph as pet_biograph
from opengate.actors.coincidences import coincidences_sorter
import pandas as pd
import uproot
import numpy as np
import time
import os.path
from itertools import chain
from collections import deque
import awkward as ak

ns = gate.g4_units.ns
mm = gate.g4_units.mm
deg = gate.g4_units.deg
Bq = gate.g4_units.Bq
kBq = 1000 * Bq
MBq = 1000 * kBq
sec = gate.g4_units.second


def run_coincidence_detection(
    root_file_path, time_window, min_transaxial_distance, max_axial_distance
):
    start = time.time()
    with uproot.open(root_file_path) as root_file:
        singles_tree = root_file["singles"]
        coincidences = coincidences_sorter(
            singles_tree,
            time_window,
            policy="takeAllGoods",
            min_transaxial_distance=min_transaxial_distance,
            transaxial_plane="xy",
            max_axial_distance=max_axial_distance,
            chunk_size="1000MB",
        )
    coincidences_pd = pd.DataFrame.from_dict(coincidences)
    # print(coincidences.keys())
    time_spent = time.time() - start
    return coincidences_pd, time_spent


def transaxial_distance(coincidences):
    x1 = coincidences["PostPosition_X1"].to_numpy()
    x2 = coincidences["PostPosition_X2"].to_numpy()
    y1 = coincidences["PostPosition_Y1"].to_numpy()
    y2 = coincidences["PostPosition_Y2"].to_numpy()
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def axial_distance(coincidences):
    z1 = coincidences["PostPosition_Z1"].to_numpy()
    z2 = coincidences["PostPosition_Z2"].to_numpy()
    return abs(z1 - z2)


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
    td = transaxial_distance(coinc)
    # >= is better
    coinc = coinc.loc[transaxial_distance(coinc) > min_transaxial_distance].reset_index(
        drop=True
    )
    # <= is better
    coinc = coinc.loc[axial_distance(coinc) < max_axial_distance].reset_index(drop=True)
    return coinc


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


def run_coincidence_detection3(
    root_file_path, chunk_size, time_window, min_transaxial_distance, max_axial_distance
):
    start = time.time()
    with uproot.open(root_file_path) as root_file:
        singles_tree = root_file["singles"]
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
            process_chunk(
                queue, time_window, min_transaxial_distance, max_axial_distance
            )
        )
    time_spent = time.time() - start
    return pd.concat(coincidences, axis=0), time_spent


def run_coincidence_detection2(
    root_file_path, time_window, min_transaxial_distance, max_axial_distance
):
    start = time.time()
    with uproot.open(root_file_path) as root_file:
        singles_tree = root_file["singles"]
        singles_pd = singles_tree.arrays(library="pd")
        print(f"There are {len(singles_pd)} singles")
        # singles_pd.insert(0, 'index', range(len(singles_pd)))
        singles_pd["VolumeIDHash"] = pd.util.hash_pandas_object(
            singles_pd["PreStepUniqueVolumeID"], index=False
        )
        time_np = singles_pd["GlobalTime"].to_numpy()
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
        rename_dict1 = {name: name + "1" for name in singles_pd.columns.tolist()}
        rename_dict2 = {name: name + "2" for name in singles_pd.columns.tolist()}
        coinc1 = (
            singles_pd.iloc[time_np_sorted_indices[indices12[:, 0]]]
            .rename(columns=rename_dict1)
            .reset_index(drop=True)
        )
        coinc2 = (
            singles_pd.iloc[time_np_sorted_indices[indices12[:, 1]]]
            .rename(columns=rename_dict2)
            .reset_index(drop=True)
        )
        # coinc = pd.concat([coinc1, coinc2], axis=1)
        interleaved_columns = list(
            chain(*zip(coinc1.columns, coinc2.columns))
        )  # Interleave column names
        coinc = pd.concat([coinc1, coinc2], axis=1)[interleaved_columns]
        coinc = coinc.loc[coinc["VolumeIDHash1"] != coinc["VolumeIDHash2"]].reset_index(
            drop=True
        )
        # >= is better
        td = transaxial_distance(coinc)
        # for row in range(len(coinc.index)):
        #     print(f"({coinc["index1"][row]}, {coinc["index2"][row]}) td {td[row]}")
        coinc = coinc.loc[
            transaxial_distance(coinc) > min_transaxial_distance
        ].reset_index(drop=True)
        # <= is better
        coinc = coinc.loc[axial_distance(coinc) < max_axial_distance].reset_index(
            drop=True
        )
        time_spent = time.time() - start
        return coinc, time_spent


def copy_singles(input_file_path, output_file_path, sort=True):
    with uproot.open(input_file_path) as input_root_file:
        singles_pd = input_root_file["singles"].arrays(library="pd")
        if sort:
            singles_pd.sort_values("GlobalTime", inplace=True, ignore_index=False)
    with uproot.recreate(output_file_path) as output_root_file:
        # print(f"columns {singles_pd.columns.tolist()}")
        output_root_file["singles"] = singles_pd  # with index
        # output_root_file["singles"] = singles_pd.to_dict(orient="list") # without index


def root_file_name(prefix, activity, duration, num_threads):
    return f"{prefix}_{activity / MBq:.0f}MBq_{duration / sec:.06f}sec_{num_threads}threads.root"


def simulate(activity, duration, num_threads):
    sim = gate.Simulation()
    sim.number_of_threads = NUM_THREADS
    sim.random_engine = "MixMaxRng"
    sim.random_seed = 1234

    pet = pet_biograph.add_pet(sim, "scanner")

    source = sim.add_source("GenericSource", "source")
    source.particle = "back_to_back"
    source.position.type = "sphere"
    source.position.radius = 1 * mm
    source.direction.type = "iso"
    source.activity = ACTIVITY / NUM_THREADS

    stats = sim.add_actor("SimulationStatisticsActor", "stats")
    singles = pet_biograph.add_digitizer(
        sim,
        pet.name,
        root_file_name("siemens", activity, duration, num_threads),
        hits_name="hits",
        singles_name="singles",
    )

    sim.run_timing_intervals = [[0, DURATION]]
    sim.run()


if __name__ == "__main__":

    ACTIVITY = 5000 * kBq
    DURATION = 0.1 * sec  # 0.01 * sec
    NUM_THREADS = 6

    singles_file_name = root_file_name("siemens", ACTIVITY, DURATION, NUM_THREADS)
    sorted_singles_file_name = f"{singles_file_name[:-5]}_sorted.root"
    if not os.path.isfile(singles_file_name):
        simulate(ACTIVITY, DURATION, NUM_THREADS)
        copy_singles(singles_file_name, sorted_singles_file_name, sort=True)
        copy_singles(singles_file_name, singles_file_name, sort=False)

    time_window = 1 * ns
    min_transaxial_distance = 156 * mm
    max_axial_distance = 168 * mm

    coincidences1s, time_spent1s = run_coincidence_detection(
        sorted_singles_file_name,
        time_window,
        min_transaxial_distance,
        max_axial_distance,
    )
    print(
        f"run_coincidence_detection with sorted input file resulted in {len(coincidences1s.index)} coincidences"
    )

    # coincidences1u, time_spent1u = run_coincidence_detection(
    #     singles_file_name,
    #     time_window,
    #     min_transaxial_distance,
    #     max_axial_distance,
    # )

    # print(f"columns {coincidences1s.columns.tolist()}")
    # print(coincidences.to_string())
    # # coincidences2, time_spent2 = run_coincidence_detection2(
    # #     singles_file_name, time_window, min_transaxial_distance, max_axial_distance
    # # )

    # coincidences2, time_spent2 = run_coincidence_detection2(
    #     singles_file_name,
    #     time_window,
    #     min_transaxial_distance,
    #     max_axial_distance,
    # )
    # print(coincidences2.to_string())

    print(
        f"Coincidences1s:  {len(coincidences1s.index)} duration {time_spent1s:.03f} seconds"
    )
    # print(
    #     f"Coincidences1u:  {len(coincidences1u.index)} duration {time_spent1u:.03f} seconds"
    # )
    # print(
    #     f"Coincidences2: {len(coincidences2.index)} duration {time_spent2:.03f} seconds"
    # )
    # print(f"Speedup {time_spent1s / time_spent2:.03f}")

    # # # tup1 = [(i1, i2) for (i1, i2) in zip(coincidences["index1"], coincidences["index2"])]
    # tup1s = list(zip(coincidences1s["index1"], coincidences1s["index2"]))
    # tup2 = list(zip(coincidences2["index1"], coincidences2["index2"]))

    # missing = 0
    # for t in tup2:
    #     if t not in tup1s:
    #         missing += 1
    #         print(f"{t} is missing in the current implementation")
    # print(f"{missing} coincidences missing in the current implementation (1s)")
    # missing = 0
    # for t in tup1s:
    #     if t not in tup2:
    #         missing += 1
    #         print(f"{t} is missing in the new implementation")
    # print(f"{missing} coincidences missing in the new implementation (2)")

    chunk_size = 4000 * 20
    coincidences3, time_spent3 = run_coincidence_detection3(
        singles_file_name,
        chunk_size,
        time_window,
        min_transaxial_distance,
        max_axial_distance,
    )
    print(
        f"Coincidences3: {len(coincidences3.index)} duration {time_spent3:.03f} seconds"
    )
    print(f"Speedup {time_spent1s / time_spent3:.03f}")

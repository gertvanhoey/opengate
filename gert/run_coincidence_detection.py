import opengate as gate
import opengate.contrib.pet.siemensbiograph as pet_biograph
from opengate.actors.coincidences import coincidences_sorter
import pandas as pd
import uproot
import numpy as np
import time
import os.path

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
    time_spent = time.time() - start
    return coincidences_pd, time_spent


def transaxial_distance_squared(coincidences):
    x1 = coincidences["PostPosition_X1"].to_numpy()
    x2 = coincidences["PostPosition_X2"].to_numpy()
    y1 = coincidences["PostPosition_Y1"].to_numpy()
    y2 = coincidences["PostPosition_Y2"].to_numpy()
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def axial_distance_squared(coincidences):
    z1 = coincidences["PostPosition_Z1"].to_numpy()
    z2 = coincidences["PostPosition_Z2"].to_numpy()
    return (z1 - z2) ** 2


def run_coincidence_detection2(
    root_file_path, time_window, min_transaxial_distance, max_axial_distance
):
    start = time.time()
    with uproot.open(root_file_path) as root_file:
        singles_pd = root_file["singles"].arrays(library="pd")
        singles_pd["VolumeIDHash"] = pd.util.hash_pandas_object(
            singles_pd["PreStepUniqueVolumeID"], index=False
        )
        time_np = np.sort(singles_pd["GlobalTime"].to_numpy())
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
            singles_pd.iloc[indices12[:, 0]]
            .rename(columns=rename_dict1)
            .reset_index(drop=True)
        )
        coinc2 = (
            singles_pd.iloc[indices12[:, 1]]
            .rename(columns=rename_dict2)
            .reset_index(drop=True)
        )
        coinc = pd.concat([coinc1, coinc2], axis=1)
        coinc = coinc.loc[coinc["VolumeIDHash1"] != coinc["VolumeIDHash2"]].reset_index(
            drop=True
        )
        coinc = coinc.loc[
            transaxial_distance_squared(coinc) >= min_transaxial_distance**2
        ].reset_index(drop=True)
        coinc = coinc.loc[
            axial_distance_squared(coinc) <= max_axial_distance**2
        ].reset_index(drop=True)
        time_spent = time.time() - start
        return coinc, time_spent


def sort_singles(input_file_path, output_file_path):
    with uproot.open(input_file_path) as input_root_file:
        print(input_root_file.keys())
        singles_tree = input_root_file["singles"]
        singles_pd = singles_tree.arrays(library="pd")[
            [
                "EventID",
                "PreStepUniqueVolumeID",
                "PostPosition_X",
                "PostPosition_Y",
                "PostPosition_Z",
                "GlobalTime",
                "TotalEnergyDeposit",
            ]
        ]
    singles_pd.sort_values("GlobalTime", inplace=True)
    with uproot.recreate(output_file_path) as output_root_file:
        output_root_file["singles"] = singles_pd


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
    DURATION = 0.0001 * sec  # 0.01 * sec
    NUM_THREADS = 1

    singles_file_name = root_file_name("siemens", ACTIVITY, DURATION, NUM_THREADS)
    sorted_singles_file_name = f"{singles_file_name[:-5]}_sorted.root"
    if not os.path.isfile(singles_file_name):
        simulate(ACTIVITY, DURATION, NUM_THREADS)
        sort_singles(singles_file_name, sorted_singles_file_name)

    time_window = 1 * ns
    min_transaxial_distance = 156 * mm
    max_axial_distance = 168 * mm
    coincidences, time_spent = run_coincidence_detection(
        sorted_singles_file_name,
        time_window,
        min_transaxial_distance,
        max_axial_distance,
    )
    print(coincidences)
    coincidences2, time_spent2 = run_coincidence_detection2(
        singles_file_name, time_window, min_transaxial_distance, max_axial_distance
    )
    print(coincidences2)

    print(
        f"Coincidences:  {len(coincidences.index)} duration {time_spent:.03f} seconds"
    )
    print(
        f"Coincidences2: {len(coincidences2.index)} duration {time_spent2:.03f} seconds"
    )
    print(f"Speedup {time_spent / time_spent2:.03f}")

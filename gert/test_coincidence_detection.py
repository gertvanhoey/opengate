import opengate as gate
import opengate.contrib.pet.siemensbiograph as pet_biograph
import pandas as pd
import uproot
import os.path
import time
from opengate.actors.coincidences import coincidences_sorter
from opengate.actors.coincidences2 import coincidences_sorter as coincidences_sorter2

ns = gate.g4_units.ns
mm = gate.g4_units.mm
deg = gate.g4_units.deg
Bq = gate.g4_units.Bq
kBq = 1000 * Bq
MBq = 1000 * kBq
sec = gate.g4_units.second


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


def copy_singles(input_file_path, output_file_path, sort=True):
    with uproot.open(input_file_path) as input_root_file:
        singles_pd = input_root_file["singles"].arrays(library="pd")
        if sort:
            singles_pd["SingleIndex"] = range(len(singles_pd.index))
            singles_pd.sort_values("GlobalTime", inplace=True, ignore_index=False)
    with uproot.recreate(output_file_path) as output_root_file:
        # output_root_file["singles"] = singles_pd  # with index
        # output_root_file["singles"] = uproot.newtree(
        #     {col: singles_pd[col].dtype for col in singles_pd.columns}
        # )
        data_dict = {col: singles_pd[col].to_numpy() for col in singles_pd.columns}
        output_root_file["singles"] = data_dict


def compare_coincidences(coincidences1, coincidences2):
    # tup1 = set(zip(coincidences1["SingleIndex1"], coincidences1["SingleIndex2"]))
    # tup2 = set(zip(coincidences2["SingleIndex1"], coincidences2["SingleIndex2"]))
    tup1 = set(
        zip(
            coincidences1["EventID1"],
            coincidences1["EventID2"],
            coincidences1["PreStepUniqueVolumeID1"],
            coincidences1["PreStepUniqueVolumeID2"],
        )
    )
    tup2 = set(
        zip(
            coincidences2["EventID1"],
            coincidences2["EventID2"],
            coincidences2["PreStepUniqueVolumeID1"],
            coincidences2["PreStepUniqueVolumeID2"],
        )
    )
    missing1 = tup2 - tup1
    print(f"{len(missing1)} coincidences missing in the current implementation (1)")
    if len(missing1) > 0:
        print(f"missing in current implementation: {missing1}")
    missing2 = tup1 - tup2
    print(f"{len(missing2)} coincidences missing in the new implementation (2)")
    if len(missing2) > 0:
        print(f"missing in new implementation: {missing2}")


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

    policies = [
        "removeMultiples",
        "takeAllGoods",
        "takeWinnerOfGoods",
        "takeIfOnlyOneGood",
        "takeWinnerIfIsGood",
        "takeWinnerIfAllAreGoods",
    ]

    for policy in policies:
        print(f"Policy '{policy}'")

        if True:
            # Current version of coincidence detection
            start = time.time()
            with uproot.open(sorted_singles_file_name) as root_file:
                singles_tree = root_file["singles"]
                coincidences = coincidences_sorter(
                    singles_tree,
                    time_window,
                    policy=policy,
                    min_transaxial_distance=min_transaxial_distance,
                    transaxial_plane="xy",
                    max_axial_distance=max_axial_distance,
                    chunk_size="1000MB",
                )
            coincidences1s = pd.DataFrame.from_dict(coincidences)
            time_spent1s = time.time() - start
            coincidences1s = coincidences1s.drop(
                columns=["SingleIndex1", "SingleIndex2"]
            )
            print(
                f"Current algorithm with sorted input file resulted in {len(coincidences1s.index)} coincidences ({time_spent1s:.03f} seconds)"
            )
            # print(coincidences1s)

        # New version of coincidence detection
        start = time.time()
        with uproot.open(singles_file_name) as root_file:
            singles_tree = root_file["singles"]
            coincidences2 = coincidences_sorter2(
                singles_tree,
                time_window,
                policy=policy,
                min_transaxial_distance=min_transaxial_distance,
                transaxial_plane="xy",
                max_axial_distance=max_axial_distance,
                chunk_size=4000 * 20,
                return_type="pd",
            )
        # coincidences2 = pd.DataFrame.from_dict(coincidences)
        time_spent2 = time.time() - start

        print(
            f"New algorithm with unsorted input file resulted in {len(coincidences2.index)} coincidences ({time_spent2:.03f} seconds)"
        )
        # print(coincidences2)

        if True:
            print(f"Speedup {time_spent1s / time_spent2:.03f}")

            compare_coincidences(coincidences1s, coincidences2)

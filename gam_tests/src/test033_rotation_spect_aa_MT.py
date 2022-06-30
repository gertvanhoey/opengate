#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gam_gate as gam
import contrib.spect_ge_nm670 as gam_spect
import contrib.phantom_nema_iec_body as gam_iec
from scipy.spatial.transform import Rotation
import numpy as np

paths = gam.get_default_test_paths(__file__, '')

# create the simulation
sim = gam.Simulation()

# main options
ui = sim.user_info
ui.g4_verbose = False
ui.check_volumes_overlap = False
ui.random_seed = 123456

# units
m = gam.g4_units('m')
cm = gam.g4_units('cm')
keV = gam.g4_units('keV')
mm = gam.g4_units('mm')
Bq = gam.g4_units('Bq')
sec = gam.g4_units('second')
deg = gam.g4_units('deg')
kBq = 1000 * Bq
MBq = 1000 * kBq

''' ================================================== '''
# main parameters
ui.visu = False
ui.g4_verbose = False
ui.visu_verbose = False
ui.number_of_threads = 2
"""
WARNING : it works but it is much slower than mono-thread simulation.
Indeed, between each run the master synchronize all workers and this requires that
all workers finish their jobs (hence longer one). 
Once all threads are synchronized, the geometry is moved. 
Then only, workers can start their next run.

It is unknown if this MT version can be useful (yet), maybe for very large 
simulation ?  
"""
colli_flag = not ui.visu
ac = 10 * MBq
ac = 1 * MBq
distance = 15 * cm
psd = 6.11 * cm
p = [0, 0, -(distance + psd)]
''' ================================================== '''

# world size
world = sim.world
world.size = [1.5 * m, 1.5 * m, 1.5 * m]
world.material = 'G4_AIR'

# spect head (debug mode = very small collimator)
spect1 = gam_spect.add_ge_nm67_spect_head(sim, 'spect1', collimator=colli_flag, debug=False)
spect1.translation, spect1.rotation = gam.get_transform_orbiting(p, 'x', 180)

# spect head (debug mode = very small collimator)
spect2 = gam_spect.add_ge_nm67_spect_head(sim, 'spect2', collimator=colli_flag, debug=False)
spect2.translation, spect2.rotation = gam.get_transform_orbiting(p, 'x', 0)

# physic list
sim.set_cut('world', 'all', 10 * mm)
# sim.set_cut('spect1_crystal', 'all', 1 * mm)
# sim.set_cut('spect2_crystal', 'all', 1 * mm)

# source #1
sources = []
source = sim.add_source('Generic', 'source1')
source.particle = 'gamma'
source.energy.type = 'mono'
source.energy.mono = 140.5 * keV
source.position.type = 'sphere'
source.position.radius = 2 * mm
source.position.translation = [0, 0, 20 * mm]
source.direction.type = 'iso'
source.direction.acceptance_angle.volumes = ['spect2', 'spect1']
source.direction.acceptance_angle.intersection_flag = True
source.direction.acceptance_angle.normal_flag = True
source.direction.acceptance_angle.normal_vector = [0, 0, -1]
source.direction.acceptance_angle.normal_tolerance = 10 * deg
source.activity = ac / ui.number_of_threads
sources.append(source)

# source #1
source2 = sim.add_source('Generic', 'source2')
gam.copy_user_info(source, source2)
source2.position.radius = 1 * mm
source2.position.translation = [20 * mm, 0, -20 * mm]
sources.append(source2)

# add stat actor
stat = sim.add_actor('SimulationStatisticsActor', 'Stats')
stat.output = paths.output / 'test033_stats.txt'

# add default digitizer (it is easy to change parameters if needed)
gam_spect.add_ge_nm670_spect_simplified_digitizer(sim, 'spect1_crystal', paths.output / 'test033_proj_1.mhd')
gam_spect.add_ge_nm670_spect_simplified_digitizer(sim, 'spect2_crystal', paths.output / 'test033_proj_2.mhd')

# motion of the spect, create also the run time interval
heads = [spect1, spect2]

# create a list of run (total = 1 second)
n = 10
sim.run_timing_intervals = gam.range_timing(0, 1 * sec, n)

for head in heads:
    motion = sim.add_actor('MotionVolumeActor', f'Move_{head.name}')
    motion.mother = head.name
    motion.translations, motion.rotations = \
        gam.volume_orbiting_transform('x', 0, 180, n, head.translation, head.rotation)
    motion.priority = 5

# go
sim.initialize()
sim.start()

stats = sim.get_actor('Stats')
print(stats)

s = 0
for source in sources:
    s += gam.get_source_skipped_particles(sim, source.name)
print(f'Skipped particles {s}')

########################
gam.warning(f'Check skipped')
ref_skipped = 1970136
tol = 0.05
d = abs(ref_skipped - s) / ref_skipped
is_ok = d < tol
gam.print_test(is_ok, f'Skipped particles ref={ref_skipped}, get {s} -> {d * 100}% vs tol={tol * 100}%')

########################
gam.warning(f'Check stats')
stats_ref = gam.read_stat_file(paths.output_ref / 'test033_stats.txt')
stats_ref.counts.run_count *= ui.number_of_threads
is_ok = gam.assert_stats(stats, stats_ref, 0.05) and is_ok

# compare edep map
gam.warning(f'Check images')
is_ok = gam.assert_images(paths.output / 'test033_proj_1.mhd',
                          paths.output_ref / 'test033_proj_1.mhd',
                          stats, tolerance=70, axis='x') and is_ok
is_ok = gam.assert_images(paths.output / 'test033_proj_2.mhd',
                          paths.output_ref / 'test033_proj_2.mhd',
                          stats, tolerance=80, axis='x') and is_ok

gam.test_ok(is_ok)

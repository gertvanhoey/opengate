import numpy as np
import opengate as gate
import opengate_core as g4
from box import Box
from scipy.spatial.transform import Rotation


class GenericSource(gate.SourceBase):
    """
    GenericSource close to the G4 GPS (General Particle Source).

    Particle: type, activity, weight, half_life, TAC

    Position: point, box, sphere, etc ..., sigma, confine, translation+rotation

    Direction: iso, focus, sigma, acceptance angle

    Energy: mono, sigma, spectrum etc

    """

    type_name = "GenericSource"

    @staticmethod
    def set_default_user_info(user_info):
        gate.SourceBase.set_default_user_info(user_info)
        # initial user info
        user_info.particle = "gamma"
        user_info.ion = Box()
        user_info.n = 0
        user_info.activity = 0
        user_info.weight = -1
        user_info.weight_sigma = -1
        user_info.half_life = -1  # negative value is no half_life
        user_info.tac_times = None
        user_info.tac_activities = None
        user_info.tac_from_decay_parameters = None
        # ion
        user_info.ion = Box()
        user_info.ion.Z = 0  # Z: Atomic Number
        user_info.ion.A = 0  # A: Atomic Mass (nn + np +nlambda)
        user_info.ion.E = 0  # E: Excitation energy (i.e. for metastable)
        # position
        user_info.position = Box()
        user_info.position.type = "point"
        user_info.position.radius = 0
        user_info.position.sigma_x = 0
        user_info.position.sigma_y = 0
        user_info.position.size = [0, 0, 0]
        user_info.position.translation = [0, 0, 0]
        user_info.position.rotation = Rotation.identity().as_matrix()
        user_info.position.confine = None
        # angle (direction)
        user_info.direction = Box()
        user_info.direction.type = "iso"
        user_info.direction.momentum = [0, 0, 1]
        user_info.direction.focus_point = [0, 0, 0]
        user_info.direction.sigma = [0, 0]
        user_info.direction.acceptance_angle = Box()
        user_info.direction.acceptance_angle.skip_policy = "SkipEvents"  # or ZeroEnergy
        user_info.direction.acceptance_angle.volumes = []
        user_info.direction.acceptance_angle.intersection_flag = False
        user_info.direction.acceptance_angle.normal_flag = False
        user_info.direction.acceptance_angle.normal_vector = [0, 0, 1]
        deg = gate.g4_units("deg")
        user_info.direction.acceptance_angle.normal_tolerance = 3 * deg
        # energy
        user_info.energy = Box()
        user_info.energy.type = "mono"
        user_info.energy.mono = 0
        user_info.energy.sigma_gauss = 0
        user_info.energy.is_cdf = False
        user_info.energy.min_energy = None
        user_info.energy.max_energy = None
        user_info.energy.histogram_weight = None
        user_info.energy.histogram_energy = None
        user_info.energy.spectrum_weight = None
        user_info.energy.spectrum_energy = None
        user_info.energy.ion_gamma_mother = None
        user_info.energy.ion_gamma_daughter = None

    def __del__(self):
        pass

    def create_g4_source(self):
        return g4.GateGenericSource()

    def __init__(self, user_info):
        super().__init__(user_info)
        if not self.user_info.particle.startswith("ion"):
            return
        words = self.user_info.particle.split(" ")
        if len(words) > 1:
            self.user_info.ion.Z = words[1]
        if len(words) > 2:
            self.user_info.ion.A = words[2]
        if len(words) > 3:
            self.user_info.ion.E = words[3]

        # will be set by g4 source
        self.fTotalZeroEvents = 0
        self.fTotalSkippedEvents = 0

    def initialize(self, run_timing_intervals):
        ui = self.user_info
        # Check user_info type
        # if not isinstance(self.user_info, Box):
        #    gate.fatal(f'Generic Source: user_info must be a Box, but is: {self.user_info}')
        if not isinstance(self.user_info, gate.UserInfo):
            gate.fatal(
                f"Generic Source: user_info must be a UserInfo, but is: {self.user_info}"
            )
        if not isinstance(ui.position, Box):
            gate.fatal(
                f"Generic Source: user_info.position must be a Box, but is: {ui.position}"
            )
        if not isinstance(ui.direction, Box):
            gate.fatal(
                f"Generic Source: user_info.direction must be a Box, but is: {ui.direction}"
            )
        if not isinstance(ui.energy, Box):
            gate.fatal(
                f"Generic Source: user_info.energy must be a Box, but is: {ui.energy}"
            )

        # check energy type
        l = [
            "mono",
            "gauss",
            "F18_analytic",
            "O15_analytic",
            "C11_analytic",
            "histogram",
            "spectrum_lines",
            "range",
        ]
        l.extend(gate.all_beta_plus_radionuclides)
        if not ui.energy.type in l:
            gate.fatal(
                f"Cannot find the energy type {ui.energy.type} for the source {ui.name}.\n"
                f"Available types are {l}"
            )

        # special case for beta plus energy spectra
        # FIXME put this elsewhere
        if ui.particle == "e+":
            if ui.energy.type in gate.all_beta_plus_radionuclides:
                data = gate.read_beta_plus_spectra(ui.energy.type)
                ene = data[:, 0] / 1000  # convert from KeV to MeV
                proba = data[:, 1]
                cdf, total = gate.compute_cdf_and_total_yield(proba, ene)
                # total = total * 1000  # (because was in MeV)
                # ui.activity *= total
                ui.energy.is_cdf = True
                self.g4_source.SetEnergyCDF(ene)
                self.g4_source.SetProbabilityCDF(cdf)

        # Compute a TAC from ion decay ?
        if ui.tac_from_decay_parameters is not None:
            self.initialize_start_end_time(run_timing_intervals)
            p = Box(ui.tac_from_decay_parameters)
            ui.tac_times, ui.tac_activities = gate.get_tac_from_decay(
                p.ion_name, p.daughter, ui.activity, ui.start_time, ui.end_time, p.bins
            )

        # Set up a TAC if needed
        self.update_tac_activity()

        # check
        self.check_ui_activity(ui)
        self.check_confine(ui)

        # initialize (must be the last step here because set user_info)
        gate.SourceBase.initialize(self, run_timing_intervals)

    def check_ui_activity(self, ui):
        if ui.n > 0 and ui.activity > 0:
            gate.fatal(f"Cannot use both n and activity, choose one: {self.user_info}")
        if ui.n == 0 and ui.activity == 0:
            gate.fatal(f"Choose either n or activity : {self.user_info}")
        if ui.activity > 0:
            ui.n = 0
        if ui.n > 0:
            ui.activity = 0

    def check_confine(self, ui):
        if ui.position.confine:
            if ui.position.type == "point":
                gate.warning(
                    f"In source {ui.name}, "
                    f"confine is used, while position.type is point ... really ?"
                )

    def prepare_output(self):
        gate.SourceBase.prepare_output(self)
        # store the output from G4 object
        self.fTotalZeroEvents = self.g4_source.fTotalZeroEvents
        self.fTotalSkippedEvents = self.g4_source.fTotalSkippedEvents

    def update_tac_activity(self):
        ui = self.user_info
        if ui.tac_times is None and ui.tac_activities is None:
            return
        n = len(ui.tac_times)
        if n != len(ui.tac_activities):
            gate.fatal(
                f"option tac_activities must have the same size than tac_times in source '{ui.name}'"
            )

        # scale the activity if energy_spectrum is given (because total may not be 100%)
        if ui.energy.spectrum_weight is not None:
            total = sum(ui.energy.spectrum_weight)
            ui.tac_activities = np.array(ui.tac_activities) * total

        # it is important to set the starting time for this source as the tac
        # may start later than the simulation timing
        i = 0
        while i < len(ui.tac_activities) and ui.tac_activities[i] <= 0:
            i += 1
        if i >= len(ui.tac_activities):
            gate.warning(f"Source '{ui.name}' TAC with zero activity.")
            sec = gate.g4_units("s")
            ui.start_time = ui.end_time + 1 * sec
        else:
            ui.start_time = ui.tac_times[i]
            ui.activity = ui.tac_activities[i]
            self.g4_source.SetTAC(ui.tac_times, ui.tac_activities)


def get_source_skipped_events(output, source_name):
    ui = output.simulation.user_info
    n = 0
    if ui.number_of_threads > 1 or ui.force_multithread_mode:
        for i in range(1, ui.number_of_threads + 1):
            s = output.get_source_MT(source_name, i)
            n += s.fTotalSkippedEvents
    else:
        n = output.get_source(source_name).fTotalSkippedEvents
    return n


def get_source_zero_events(output, source_name):
    ui = output.simulation.user_info
    n = 0
    if ui.number_of_threads > 1 or ui.force_multithread_mode:
        for i in range(1, ui.number_of_threads + 1):
            s = output.get_source_MT(source_name, i)
            n += s.fTotalZeroEvents
    else:
        n = output.get_source(source_name).fTotalZeroEvents
    return n

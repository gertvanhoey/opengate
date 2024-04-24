import numpy as np
from pathlib import Path

from ..base import GateObject
from ..image import (
    write_itk_image,
    update_image_py_to_cpp,
    create_3d_image,
    get_py_image_from_cpp_image,
    sum_itk_images,
    divide_itk_images,
)
from ..utility import ensure_filename_is_str, insert_suffix_before_extension
from ..exception import warning, fatal

from .dataitems import available_data_container_classes


def _setter_hook_belongs_to(self, belongs_to):
    if belongs_to is None:
        fatal("The belongs_to attribute of an ActorOutput cannot be None.")
    try:
        belongs_to_name = belongs_to.name
    except AttributeError:
        belongs_to_name = belongs_to
    return belongs_to_name


def _setter_hook_path(self, path):
    return Path(path)


class ActorOutputBase(GateObject):
    user_info_defaults = {
        "belongs_to": (
            None,
            {
                "doc": "Name of the actor to which this output belongs.",
                "setter_hook": _setter_hook_belongs_to,
                "required": True,
            },
        ),
        "output_filename": (
            None,
            {
                "doc": "Filename for the data represented by this actor output. "
                "Relative paths and filenames are taken "
                "relative to the global simulation output folder "
                "set via the Simulation.output_path option. ",
            },
        ),
        "write_to_disk": (
            True,
            {
                "doc": "Should the data be written to disk?",
            },
        ),
        "extra_suffix": (
            "",
            {
                "doc": "Extra suffix to be appended to file name. ",
            },
        ),
        "keep_data_in_memory": (
            True,
            {
                "doc": "Should the data be kept in memory after the end of the simulation? "
                "Otherwise, it is only stored on disk and needs to be re-loaded manually. "
                "Careful: Large data structures like a phase space need a lot of memory.",
            },
        ),
        "keep_data_per_run": (
            False,
            {
                "doc": "In case the simulation has multiple runs, should separate results per run be kept?"
            },
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data_per_run = {}  # holds the data per run in memory
        self.merged_data = None  # holds the data merged from multiple runs in memory

    def __len__(self):
        return len(self.data_per_run)

    def __getitem__(self, which):
        return self.get_data(which, None)

    @property
    def data(self):
        if len(self.data_per_run) > 1:
            warning(
                f"You are using the convenience property 'data' to access the data in this actor output. "
                f"This returns you the data from the first run, but the actor output stores "
                f"data from {len(self.data_per_run)} runs. "
                f"To access them, use 'data_per_run[RUN_INDEX]' instead or 'merged_data'. "
            )
        return self.data_per_run[0]

    @property
    def belongs_to_actor(self):
        return self.simulation.actor_manager.get_actor(self.belongs_to)

    def initialize(self):
        if self.output_filename is None:
            self.output_filename = f"output_{self.name}_from_actor_{self.belongs_to_actor.name}.{self.default_suffix}"

    def write_data_if_requested(self, *args, **kwargs):
        if self.write_to_disk is True:
            self.write_data(*args, **kwargs)

    def get_output_path(self, which):
        full_data_path = insert_suffix_before_extension(
            self.simulation.get_output_path(self.output_filename), self.extra_suffix
        )
        if which == "merged":
            return insert_suffix_before_extension(full_data_path, "merged")
        else:
            try:
                run_index = int(which)
            except ValueError:
                fatal(
                    f"Invalid argument 'which' in get_output_path() method "
                    f"of {type(self).__name__} called {self.name}"
                    f"Valid arguments are a run index (int) or the term 'merged'. "
                )
            return insert_suffix_before_extension(full_data_path, f"run{run_index:04f}")

    def close(self):
        if self.keep_data_in_memory is False:
            self.data_per_run = {}
            self.merged_data = None
        super().close()

    def get_data(self, *args, **kwargs):
        raise NotImplementedError("This is the base class. ")

    def store_data(self, *args, **kwargs):
        raise NotImplementedError("This is the base class. ")

    def write_data(self, *args, **kwargs):
        raise NotImplementedError("This is the base class. ")

    def load_data(self, which):
        raise NotImplementedError(
            f"Your are calling this method from the base class {type(self).__name__}, "
            f"but it should be implemented in the specific derived class"
        )


class AutoMergeActorOutput(ActorOutputBase):
    user_info_defaults = {
        "merge_method": (
            "sum",
            {
                "doc": "How should images from runs be merged?",
                "allowed_values": ("sum",),
            },
        ),
        "auto_merge": (
            True,
            {
                "doc": "In case the simulation has multiple runs, should results from separate runs be merged?"
            },
        ),
        # "data_container_class": (
        #     None,
        #     {"doc": "FIXME"},
        # ),
    }

    def __init__(self, data_container_class, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if data_container_class in available_data_container_classes.values():
            self.data_container_class = data_container_class
        else:
            try:
                self.data_container_class = available_data_container_classes[
                    data_container_class
                ]
            except KeyError:
                fatal(
                    f"Unknown data item class {data_container_class}. "
                    f"Available classes are: {list(available_data_container_classes.keys())}"
                )

    # def __contains__(self, item):
    #     return item in self.data_per_run

    def merge_data(self, list_of_data):
        if self.merge_method == "sum":
            merged_data = list_of_data[0]
            for d in list_of_data[1:]:
                merged_data += d
            return merged_data

    def merge_data_from_runs(self):
        self.merged_data = self.merge_data(list(self.data_per_run.values()))

    def merge_into_merged_data(self, data):
        self.merged_data = self.merge_data([self.merged_data, data])

    def end_of_run(self, run_index):
        if self.keep_data_per_run is False:
            if self.auto_merge is True:
                self.merge_into_merged_data(self.data_per_run[run_index])
            self.data_per_run[run_index] = None

    def end_of_simulation(self):
        if self.auto_merge is True:
            self.merge_data_from_runs()
        if self.keep_data_per_run is False:
            for k in self.data_per_run:
                self.data_per_run[k] = None

    def get_data_container(self, which):
        if which == "merged":
            return self.merged_data
        else:
            try:
                run_index = int(which)  # might be a run_index
                if (
                    run_index in self.data_per_run
                    and self.data_per_run[run_index] is not None
                ):
                    return self.data_per_run[run_index]
                else:
                    fatal(f"No data stored for run index {run_index}")
            except ValueError:
                fatal(
                    f"Invalid argument 'which' in get_data() method of ActorOutput {self.name}. "
                    f"Allowed values are: 'merged' or a valid run_index. "
                )

    def get_data(self, which, item=None):
        return self.get_data_container(which).get_data(item)

    def store_data(self, which, *data):
        if isinstance(data, self.data_container_class):
            data_item = data
        else:
            data_item = self.data_container_class(data=data)
        if which == "merged":
            self.merged_data = data_item
        else:
            try:
                run_index = int(which)  # might be a run_index
                if run_index not in self.data_per_run:
                    self.data_per_run[run_index] = data_item
                else:
                    fatal(
                        f"A data item is already set for run index {run_index}. "
                        f"You can only merge additional data into it. Overwriting is not allowed. "
                    )
            except ValueError:
                fatal(
                    f"Invalid argument 'which' in store_data() method of ActorOutput {self.name}. "
                    f"Allowed values are: 'merged' or a valid run_index. "
                )

    def load_data(self, which):
        raise NotImplementedError(
            f"Your are calling this method from the base class {type(self).__name__}, "
            f"but it should be implemented in the specific derived class"
        )

    def collect_data(self, which, return_identifier=False):
        if which == "merged":
            data = [self.merged_data]
            identifiers = ["merged"]
        elif which == "all_runs":
            data = list(self.data_per_run.values())
            identifiers = list(self.data_per_run.keys())
        elif which == "all":
            data = list(self.data_per_run.values())
            data.append(self.merged_data)
            identifiers = list(self.data_per_run.keys())
            identifiers.append("merged")
        else:
            try:
                ri = int(which)
            except ValueError:
                fatal(f"Invalid argument which in method collect_images(): {which}")
            data = [self.data_per_run[ri]]
            identifiers = [ri]
        if return_identifier is True:
            return data, identifiers
        else:
            return data

    def write_data(self, which):
        if which == "all_runs":
            for k in self.data_per_run.keys():
                self.write_data(k)
        elif which == "merged":
            if self.merged_data is not None:
                self.merged_data.write(self.get_output_path(which))
        elif which == "all":
            self.write_data("all_runs")
            self.write_data("merged")
        else:
            try:
                data = self.data_per_run[which]
            except KeyError:
                fatal(
                    f"Invalid argument 'which' in method write_data(): {which}. "
                    f"Allowed values are 'all', 'all_runs', 'merged', or a valid run_index"
                )
            if data is not None:
                data.write(self.get_output_path(which))

    def get_output_path_for_item(self, which, item):
        output_path = self.get_output_path(which)
        if which == "merged":
            data = self.merged_data
        else:
            try:
                data = self.data_per_run[which]
            except KeyError:
                fatal(
                    f"Invalid argument 'which' in method get_output_path_for_item(): {which}. "
                    f"Allowed values are 'merged' or a valid run_index. "
                )
        return data.get_output_path_to_item(output_path, item)


class ActorOutputImage(AutoMergeActorOutput):
    user_info_defaults = {
        "size": (
            None,
            {
                "doc": "Size of the image in voxels.",
            },
        ),
        "spacing": (
            None,
            {
                "doc": "Spacing of the image.",
            },
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_suffix = "mhd"

    def set_image_properties(self, which, **kwargs):
        for image_data in self.collect_data(which):
            if image_data is not None:
                image_data.set_image_properties(**kwargs)

    def get_image_properties(self, which, item=0):
        if which == "merged":
            if self.merged_data is not None:
                return self.merged_data.get_image_properties()[item]
        else:
            try:
                run_index = int(which)
                try:
                    image_data_container = self.data_per_run[run_index]
                except KeyError:
                    fatal(f"No data found for run index {run_index}.")
                if image_data_container is not None:
                    return image_data_container.get_image_properties()[item]

            except ValueError:
                fatal(
                    f"Illegal argument 'which'. Provide a valid run index or the term 'merged'."
                )

    def create_empty_image(self, run_index, *args, **kwargs):
        self.data_per_run[run_index].create_empty_image(*args, **kwargs)


# concrete classes usable in Actors:
class ActorOutputSingleImage(ActorOutputImage):

    def __init__(self, *args, **kwargs):
        super().__init__("SingleItkImageDataItem", *args, **kwargs)


class ActorOutputQuotientImage(ActorOutputImage):

    def __init__(self, *args, **kwargs):
        super().__init__("QuotientItkImageDataItem", *args, **kwargs)


class ActorOutputRoot(ActorOutputBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_suffix = "root"

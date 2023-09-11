import copy
from .geometry.helpers_geometry import volume_builders, volume_type_names
from .source.helpers_source import source_builders, source_type_names
from .actor.helpers_actor import actor_builders, actor_type_names
from .actor.helpers_filter import filter_builders, filter_type_names
from .helpers import fatal


element_builders = {
    "Volume": volume_builders,
    "Source": source_builders,
    "Actor": actor_builders,
    "Filter": filter_builders,
}


def get_element_class(element_type, type_name):
    """
    Return the class of the given type_name (in the element_type list)
    """
    elements = None
    if element_type == "Volume":
        elements = volume_type_names
    if element_type == "Source":
        elements = source_type_names
    if element_type == "Actor":
        elements = actor_type_names
    if element_type == "Filter":
        elements = filter_type_names
    if not elements:
        fatal(
            f"Error, element_type={element_type} is   unknown. Use Volume, Source or Actor."
        )
    for e in elements:
        # check the class has type_name
        if not hasattr(e, "type_name"):
            fatal(
                f'Error, the class {e.__name__} *must* have a static attribute called "type_name"'
            )
        # is the type the one we are looking ?
        if e.type_name == type_name:
            return e
    s = [x.type_name for x in elements]
    fatal(
        f'Error {element_type}: the type "{type_name}" is unknown. Known types are {s}'
    )


def get_builder(element_type, type_name):
    """
    Return a function that build an element of the class type_name
    Check everything first.
    """
    # get type of element builder
    if element_type not in element_builders:
        fatal(
            f"The element type: {element_type} is unknown.\n"
            f"Known element types are {element_builders.keys()}"
        )
    builders = element_builders[element_type]
    # get builder
    if type_name not in builders:
        fatal(
            f"The element type: {type_name} is unknown.\n"
            f"Known type names are {builders.keys()}"
        )
    builder = builders[type_name]
    return builder


def new_element(user_info, simulation=None):
    """
    Create a new element (Volume, Source, Actor, Filter), according to the type name
    - use the element_builders to find the class to build
    - create a new element, with the name as parameter to the constructor
    - initialize the default list of keys in the user_info
    - set a pointer to the Simulation object
    """
    builder = get_builder(user_info.element_type, user_info.type_name)
    # build (create the object)
    e = builder(user_info)
    # set the simulation pointer
    e.set_simulation(simulation)
    return e


def copy_user_info(v1, v2):
    """
    Copy all attributes from v1 to v2, except the name.
    v1 is assumed to be a UserInfo object with several attribute members.
    v2 must have the (at least) the same set of attributes.
    Values are (deep) copied.
    """
    for k in v1.__dict__:
        if k == "name":
            continue
        if k == "_name":
            continue
        setattr(v2, k, copy.deepcopy(v1.__dict__[k]))

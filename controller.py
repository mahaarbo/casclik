"""Controllers

This module contains the controller classes that can be constructed
from the skill specification. Some would call this the solver, as it
solves the contraint problem formulated by the skill specification but
we've called it the controller as it chooses the derivatives of
robot_var and virtual_var to 'converge' to the desired values. And
what is a controller if not exactly that? The user is still tasked
with making sure the control values are applied to the system.


"""
import casadi as cs
from skill_specification import SkillSpecification
from constraints import EqualityConstraint, SetConstraint


class BaseController(object):
    def __init__(self, skill_spec):
        pass

    def __repr__(self):
        return self.controller_type+"<"+self.skill_spec.label+">"


class ReactiveQPController(BaseController):
    """Reactive QP controller.

    The reactive QP controller can handle skills with input. It
    defines a QP problem from the skill specification. As it is a
    reactive controller resolved to the speeds of robot_var, it can
    handle inputs such as force sensors with a dampening effect.

    Args:
        skill_spec (SkillSpecification): skill specification
        robot_var_weights (list,numpy.ndarray): weights in QP
        virtual_var_weights (list, numpy.ndarray): weights in QP
        slack_var_weights (list, numpy.ndarray): weights in QP
        options (dict): options dictionary, see self.options_info
    """
    controller_type = "ReactiveQPController"
    options_info = """TODO
    solver_opts (dict): solver options, see casadi."""

    def __init__(self, skill_spec,
                 robot_var_weights=None,
                 virtual_var_weights=None,
                 slack_var_weights=None,
                 options=None):
        self.skill_spec = skill_spec
        if robot_var_weights is None:
            robot_var_weights = [1.]*skill_spec.n_robot_var
        self.robot_var_weights = robot_var_weights
        if virtual_var_weights is None:
            virtual_var_weights = [1.]*skill_spec.n_virtual_var
        self.virtual_var_weights = virtual_var_weights
        if slack_var_weights is None:
            slack_var_weights = [1.]*skill_spec.n_slack_var
        self.slack_var_weights = slack_var_weights
        if options is None:
            options = {}
        self.options = options
        

class ModelPredictiveController(BaseController):
    """Model Predictive controller.

    The model predictive controller can only handle skills where there
    is no input. This is because the input variables are considered to
    be unknown before they occur. It is based on the reactive QP
    controller, but considers multiple steps ahead instead of just
    one.

    Args:
        skill_spec (SkillSpecification): skill specification
        options (dict): options dictionary, see self.options_info
    """
    controller_type = "ModelPredictiveController"
    options_info = """TODO
    solver_opt (dict): solver options, see casadi."""

    def __init__(self, skill_spec, horizon_length, options):
        if skill_spec.input_var is not None:
            raise TypeError("ModelPredictiveController can only handle skills"
                            + " without input_var, "+skill_spec.label+" has "
                            + str(skill_spec.input_var) + " as input_var.")
        self.skill_spec = skill_spec


class PseudoInverseController(BaseController):
    """Pseudo Inverse controller.

    The pseudo inverse controller is based on the Set-Based task
    controller of Signe Moe, and reminiscent of the stack-of-tasks
    controller. It uses the moore-penrose pseudo-inverse to calculate
    robot_var speeds, and the in_tangent_cone to evaluate handle
    set-based constraints. As it is a reactive controller resolved to
    the speeds of robot_var, it can handle inputs such as force
    sensors with a dampening effect.

    Args:
        skill_spec (SkillSpecification): skill specification
        options (dict): options dictionary, see self.options_info
    """
    controller_type = "PseudoInverseController"
    options_description = """TODO"""

    def __init__(self, skill_spec, options):
        self.skill_spec = skill_spec

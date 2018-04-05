"""Controllers

This module contains the controller classes that can be constructed
from the skill specification. Some would call this the solver, as it
solves the contraint problem formulated by the skill specification but
we've called it the controller as it chooses the derivatives of
robot_var and virtual_var to 'converge' to the desired values. And
what is a controller if not exactly that? The user is still tasked
with making sure the control values are applied to the system.

TODO:
    * Make so we can't delete the weights in ReactiveQPController
    * Make it possible to choose H opt.prob. in ReactiveQPController
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
        self.skill_spec = skill_spec  # Core of everything
        # Weights for Quadratic cost
        self.robot_var_weights = robot_var_weights
        self.virtual_var_weights = virtual_var_weights
        self.slack_var_weights = slack_var_weights

        # Create the expr. for the optimization variable
        if self.skill_spec.n_slack_var == 0:
            self.opt_var = cs.vertcat(self.skill_spec.robot_vel_var,
                                      self.skill_spec.virtual_vel_var)
        else:
            self.opt_var = cs.vertcat(self.skill_spec.robot_vel_var,
                                      self.skill_spec.virtual_vel_var,
                                      self.skill_spec.slack_var)
        # Options are dicts
        if options is None:
            options = {}
        self.options = options

    @property
    def robot_var_weights(self):
        """Get or set the robot_var_weights. This is an iterable (e.g. list,
        np.ndarray) with the same length as the robot_var.
        """
        return self._robot_var_weights

    @robot_var_weights.setter
    def robot_var_weights(self, weights):
        if weights is None:
            weights = [1.]*self.skill_spec.n_robot_var
        elif len(weights) != self.skill_spec.n_robot_var:
            raise ValueError("robot_var_weights and robot_var dimensions"
                             + " do not match")
        self._robot_var_weights = weights

    @property
    def virtual_var_weights(self):
        """Get or set the virutal_var_weights. This is an iterable thing with
        len (e.g. list, np.ndarray) with the same length as the
        virtual_var."""
        return self._virtual_var_weights

    @virtual_var_weights.setter
    def virtual_var_weights(self, weights):
        if weights is None:
            weights = [1.]*self.skill_spec.n_virtual_var
        elif len(weights) != self.skill_spec.n_virtual_var:
            raise ValueError("virtual_var_weights and virtual_var dimensions"
                             + " do not match")
        self._virtual_var_weights = weights

    @property
    def slack_var_weights(self):
        """Get or set the slack_var_weights. This is an iterable thing with
        len (e.g. list, np.ndarray) with the same length as n_slack_var in the
        skill specification."""
        return self._slack_var_weights

    @slack_var_weights.setter
    def slack_var_weights(self, weights):
        if weights is None:
            weights = [10.]*self.skill_spec.n_slack_var
        elif len(weights) != self.skill_spec.n_slack_var:
            raise ValueError("slack_var_weights and slack_var dimensions"
                             + " do not match.")
        self._slack_var_weights = weights

    def get_cost_expr(self):
        cost = 0
        rob_w = self.robot_var_weights
        virt_w = self.virtual_var_weights
        slack_w = self.slack_var_weights
        rob_v = self.skill_spec.robot_vel_var
        virt_v = self.skill_spec.virtual_vel_var
        slack_v = self.skill_spec.slack_var
        cost += cs.mtimes(cs.mtimes(rob_v.T, cs.diag(rob_w)), rob_v)
        if virt_v is not None:
            cost += cs.mtimes(cs.mtimes(virt_v.T, cs.diag(virt_w)), virt_v)
        if slack_v is not None:
            cost += cs.mtimes(cs.mtimes(slack_v.T, cs.diag(slack_w)), slack_v)
        return cost

    def get_constraint_expr(self):
        """Returns a casadi expression describing all the constraints, and
        expressions for their upper and lower bounds.

        Return:
            tuple: (A*x, Blb,Bub) for Blb<=A*x<Bub, where A*x, Blb and Bub are
            casadi expressions.
        """
        cnstr_expr_list = []
        lb_cnstr_expr_list = []  # lower bound
        ub_cnstr_expr_list = []  # upper bound
        slack_v_ind = 0
        time_var = self.skill_spec.time_var
        robot_var = self.skill_spec.robot_var
        robot_vel_var = self.skill_spec.robot_vel_var
        virtual_var = self.skill_spec.virtual_var
        virtual_vel_var = self.skill_spec.virtual_vel_var
        slack_var = self.skill_spec.slack_var

        for cnstr in self.skill_spec.constraints:
            expr_len = cnstr.expression.size()[0]
            # What's A*opt_var?
            cnstr_expr = cnstr.jtimes(robot_var, robot_vel_var)
            if virtual_var is not None:
                cnstr_expr += cnstr.jtimes(virtual_var, virtual_vel_var)
            
            # Everyone wants a feedforward
            lb_cnstr_expr = -cnstr.jacobian(time_var)
            ub_cnstr_expr = -cnstr.jacobian(time_var)
            # Setup bounds
            if isinstance(cnstr, EqualityConstraint):
                lb_cnstr_expr += -cs.mtimes(cnstr.gain, cnstr.expression)
                ub_cnstr_expr += -cs.mtimes(cnstr.gain, cnstr.expression)
            if isinstance(cnstr, SetConstraint):
                lb_cnstr_expr += -cs.mtimes(cnstr.gain, cnstr.set_min - cnstr.expression)
                ub_cnstr_expr += -cs.mtimes(cnstr.gain, cnstr.set_max - cnstr.expression)
            # Soft constraints have slack
            if cnstr.constraint_type == "soft":
                lb_cnstr_expr += slack_var[slack_v_ind:slack_v_ind+expr_len]
                ub_cnstr_expr += slack_var[slack_v_ind:slack_v_ind+expr_len]
            # Add to lists
            cnstr_expr_list += [cnstr_expr]
            lb_cnstr_expr_list += [lb_cnstr_expr]
            ub_cnstr_expr_list += [ub_cnstr_expr]
        cnstr_expr_full = cs.vertcat(*cnstr_expr_list)
        lb_cnstr_expr_full = cs.vertcat(*lb_cnstr_expr_list)
        ub_cnstr_expr_full = cs.vertcat(*ub_cnstr_expr_list)
        return cnstr_expr_full, lb_cnstr_expr_full, ub_cnstr_expr_full

class ReactiveNLPController(BaseController):
    # You can choose cost function
    # You can choose whether constraints are EULER or RK4
    # You can have robot_vel and virtual_vel constraints
    pass


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

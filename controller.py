"""Controllers

This module contains the controller classes that can be constructed
from the skill specification. Some would call this the solver, as it
solves the contraint problem formulated by the skill specification but
I've called it the controller as it chooses robot_vel_var and
virtual_vel_var to 'converge' to the desired values. And what is a
controller if not that? The user is still tasked with making sure the
control values are applied to the system in an orderly fashion of
course.

TODO:
    * Make so we can't delete the weights in ReactiveQPController
    * Add logging in controllers when making prob/solv/func
    * Add compiling of problem_functions for faster evaluation of big ones
    * Make virtual var internal to controllers
    * Add sanity check on ReactiveNLPController:cost_expression.setter
    * Add VelocityEqualityConstraint and VelocitySetConstraint support
    * Add inital loop-closure solving. (Getting initial virtual_var/slack).
    * Add warmstart option to ReactiveNLPController
    * Fix default options of ReactiveQPController
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
    defines a quadratic problem with constraints from the skill
    specification. As it is a reactive controller resolved to the
    speeds of robot_var, it can handle inputs such as force sensors
    with a dampening effect.

    Given that v = [robot_vel_var, virtual_vel_var], this is
        min_v v^T H v
        s.t.: constraints
    where H is the cost expression. The cost expression is a diagonal
    matrix where the entries on the diagonal (the weights) must be
    either floats or casadi expressions defined by robot_var,
    virtual_var, or input_var. We're following the eTaSL approach, and
    have a "weight_shifter", or regularization constant in the cost. It
    is to shift the weight between v and the slack variables.

    Args:
        skill_spec (SkillSpecification): skill specification
        robot_var_weights (list): weights in QP, can be floats, or MX syms
        virtual_var_weights (list): weights in QP, can be floats, or MX syms
        slack_var_weights (list): weights in QP, can be floats or MX syms
        options (dict): options dictionary, see self.options_info

    """
    controller_type = "ReactiveQPController"
    options_info = """TODO
    solver_opts (dict): solver options, see casadi.
    function_opts (dict): problem function options. See below."""
    weight_shifter = 0.001  # See eTaSL paper, corresponds to mu

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
        ns = self.skill_spec.n_slack_var
        if weights is None:
            weights = [1.]*ns
        elif isinstance(weights, cs.MX) and weights.size()[0] != ns:
            raise ValueError("slack_var_weights and slack_var dimensions"
                             + " do not match.")
        elif len(weights) != self.skill_spec.n_slack_var:
            raise ValueError("slack_var_weights and slack_var dimensions"
                             + " do not match.")
        self._slack_var_weights = weights

    def get_cost_expr(self):
        """Returns a casadi expression describing the cost.

        Return:
             H for min_opt_var opt_var^T*H*opt_var"""
        nrob = self.skill_spec.n_robot_var
        nvirt = self.skill_spec.n_virtual_var
        nslack = self.skill_spec.n_slack_var
        n_opt_var = nrob + nvirt + nslack
        H = cs.MX.zeros((n_opt_var, n_opt_var))
        H[:nrob, :nrob] = self.weight_shifter*cs.diag(self.robot_var_weights)
        if nvirt > 0:
            H[nrob:nrob+nvirt, nrob:nrob+nvirt] = self.weight_shifter*cs.diag(self.virtual_var_weights)
        if nslack > 0:
            H[-nslack:, -nslack:] = self.weight_shifter*cs.MX.eye(nslack) + cs.diag(self.slack_var_weights)
        return H

    def get_constraints_expr(self):
        """Returns a casadi expression describing all the constraints, and
        expressions for their upper and lower bounds.

        Return:
            tuple: (A, Blb,Bub) for Blb<=A*x<Bub, where A*x, Blb and Bub are
            casadi expressions.
        """
        cnstr_expr_list = []
        lb_cnstr_expr_list = []  # lower bound
        ub_cnstr_expr_list = []  # upper bound
        time_var = self.skill_spec.time_var
        robot_var = self.skill_spec.robot_var
        virtual_var = self.skill_spec.virtual_var
        n_slack = self.skill_spec.n_slack_var
        slack_ind = 0
        for cnstr in self.skill_spec.constraints:
            expr_size = cnstr.expression.size()
            # What's A*opt_var?
            cnstr_expr = cnstr.jacobian(robot_var)
            if virtual_var is not None:
                cnstr_expr = cs.horzcat(cnstr_expr,
                                        cnstr.jacobian(virtual_var))
            # Everyone wants a feedforward
            lb_cnstr_expr = -cnstr.jacobian(time_var)
            ub_cnstr_expr = -cnstr.jacobian(time_var)
            # Setup bounds
            if isinstance(cnstr, EqualityConstraint):
                lb_cnstr_expr += -cs.mtimes(cnstr.gain, cnstr.expression)
                ub_cnstr_expr += -cs.mtimes(cnstr.gain, cnstr.expression)
            if isinstance(cnstr, SetConstraint):
                ub_cnstr_expr += cs.mtimes(cnstr.gain,
                                           cnstr.set_max - cnstr.expression)
                lb_cnstr_expr += cs.mtimes(cnstr.gain,
                                           cnstr.set_min - cnstr.expression)
            # Soft constraints have slack
            if n_slack > 0:
                slack_mat = cs.DM.zeros((expr_size[0], n_slack))
                if cnstr.constraint_type == "soft":
                    slack_mat[:, slack_ind:slack_ind + expr_size[0]] = -cs.DM.eye(expr_size[0])
                    slack_ind += expr_size[0]
                cnstr_expr = cs.horzcat(cnstr_expr, slack_mat)
            # Add to lists
            cnstr_expr_list += [cnstr_expr]
            lb_cnstr_expr_list += [lb_cnstr_expr]
            ub_cnstr_expr_list += [ub_cnstr_expr]
        cnstr_expr_full = cs.vertcat(*cnstr_expr_list)
        lb_cnstr_expr_full = cs.vertcat(*lb_cnstr_expr_list)
        ub_cnstr_expr_full = cs.vertcat(*ub_cnstr_expr_list)
        return cnstr_expr_full, lb_cnstr_expr_full, ub_cnstr_expr_full

    def setup_solver(self):
        """Initialize the QP solver.

        This uses the casadi low-level interface for QP problems. It
        uses the sparsity of the H, A, B_lb and B_ub matrices.
        """
        H_expr = self.get_cost_expr()
        A_expr, Blb_expr, Bub_expr = self.get_constraints_expr()
        if "solver_opts" not in self.options:
            self.options["solver_opts"] = {}
        self.solver = cs.conic("solver",
                               "qpoases",
                               {"h": H_expr.sparsity(),
                                "a": A_expr.sparsity()},
                               self.options["solver_opts"])

    def setup_problem_functions(self):
        """Initializes the problem functions.

        With opt_var = v, optimization problem is of the form:
           min_v   v^T*H*v
           s.t.: B_lb <= A*v <= B_ub
        In this function we define the functions that form A, B_lb and B_ub."""
        H_expr = self.get_cost_expr()
        A_expr, Blb_expr, Bub_expr = self.get_constraints_expr()
        time_var = self.skill_spec.time_var
        robot_var = self.skill_spec.robot_var
        list_vars = [time_var, robot_var]
        list_names = ["time_var", "robot_var"]
        virtual_var = self.skill_spec.virtual_var
        if virtual_var is not None:
            list_vars += [virtual_var]
            list_names += ["virtual_var"]
        input_var = self.skill_spec.input_var
        if input_var is not None:
            list_vars += [input_var]
            list_names += ["input_var"]
        H_func = cs.Function("H_func", list_vars, [H_expr], list_names, ["H"])
        A_func = cs.Function("A_func", list_vars, [A_expr], list_names, ["A"])
        Blb_func = cs.Function("Blb_expr", list_vars, [Blb_expr],
                               list_names, ["Blb"])
        Bub_func = cs.Function("Bub_expr", list_vars, [Bub_expr],
                               list_names, ["Bub"])
        self.H_func = H_func
        self.A_func = A_func
        self.Blb_func = Blb_func
        self.Bub_func = Bub_func

    def solve(self, time_var,
              robot_var,
              virtual_var=None,
              input_var=None):
        currvals = [time_var, robot_var]
        if virtual_var is not None:
            currvals += [virtual_var]
        if input_var is not None:
            currvals += [input_var]
        H = self.H_func(*currvals)
        A = self.A_func(*currvals)
        Blb = self.Blb_func(*currvals)
        Bub = self.Bub_func(*currvals)
        self.res = self.solver(h=H, a=A, lba=Blb, uba=Bub)
        nrob = self.skill_spec.n_robot_var
        nvirt = self.skill_spec.n_virtual_var
        if nvirt > 0:
            return self.res["x"][:nrob], self.res["x"][nrob:nrob+nvirt]
        else:
            return self.res["x"][:nrob]


class ReactiveNLPController(BaseController):
    """Reactive NLP Controller.

    The reactive NLP controller can handle skills with input. It
    defines a nonlinear problem with constraints from the skill
    specification. The robot_var, virtual_var, and input_var in the
    expressions are handled as parameters to the solver. The
    constraints are constructed from the skill_specification similarly
    as it is in the ReactiveQPController. The main usage for
    controller is when you have a nonlinear cost function you want to
    employ.

    Args:
        skill_spec (SkillSpecification): skill specification
        cost_expr (cs.MX): expression of the cost function
        slack_var_weights (list): weights in QP, defaults to 1.
        options (dict): options dictionary, see self.options_info
    """
    controller_type = "ReactiveNLPController"
    options_info = """TODO
    solver_name (str): type of solver, default ipopt.
    solver_opts (dict): solver options, see casadi.
    function_opts (dict): problem function options. See below."""
    weight_shifter = 0.001

    def __init__(self, skill_spec,
                 cost_expr,
                 slack_var_weights=None,
                 options=None):
        self.skill_spec = skill_spec
        self.slack_var_weights = slack_var_weights
        self.cost_expression = cost_expr
        # Fix default options
        if options is None:
            options = {}
        if "solver_name" not in options:
            options["solver_name"] = "ipopt"
        if "solver_opts" not in options:
            options["solver_opts"] = {}
        solver_opts = options["solver_opts"]
        if "print_time" not in solver_opts:
            solver_opts["print_time"] = False
        if options["solver_name"] == "ipopt" and "ipopt" not in solver_opts:
            solver_opts["ipopt"] = {}
        if "print_level" not in solver_opts["ipopt"]:
            solver_opts["ipopt"] = {"print_level": 0}
            
        self.options = options

    @property
    def cost_expression(self):
        """Get or set the cost expression."""
        return self._cost_expression

    @cost_expression.setter
    def cost_expression(self, expr):
        self._cost_expression = expr

    @property
    def slack_var_weights(self):
        """Get or set the slack_var_weights. Can be list np.ndarray, cs.MX, but needs to have the same length"""
        return self._slack_var_weights

    @slack_var_weights.setter
    def slack_var_weights(self, weights):
        ns = self.skill_spec.n_slack_var
        if weights is None:
            weights = [1.]*ns
        elif isinstance(weights, cs.MX) and weights.size()[0] != ns:
            raise ValueError("slack_var_weights and slack_var dimensions"
                             + " do not match.")
        elif len(weights) != self.skill_spec.n_slack_var:
            raise ValueError("slack_var_weights and slack_var dimensions"
                             + " do not match.")
        self._slack_var_weights = weights

    @property
    def skill_spec(self):
        """Get or set the skill_spec. Automatically sets self._opt_var."""
        return self._skill_spec

    @skill_spec.setter
    def skill_spec(self, spec):
        list_opt_var = [spec.robot_vel_var]
        n_opt_var = spec.n_robot_var
        if spec.virtual_var is not None:
            list_opt_var += [spec.virtual_vel_var]
            n_opt_var += spec.n_virtual_var
        if spec.slack_var is not None:
            list_opt_var += [spec.slack_var]
            n_opt_var += spec.n_slack_var
        self._opt_var = cs.vertcat(*list_opt_var)
        self._n_opt_var = n_opt_var
        self._skill_spec = spec

    def get_regularised_cost_expr(self):
        slack_var = self.skill_spec.slack_var
        if slack_var is not None:
            nslack = self.skill_spec.n_slack_var
            slack_H = cs.diag(self.slack_var_weights)
            slack_H += self.weight_shifter*cs.MX.eye(nslack)
            slack_cost = cs.mtimes(cs.mtimes(slack_var.T, slack_H), slack_var)
        else:
            slack_cost = 0.0
        return self.weight_shifter*self.cost_expression + slack_cost

    def get_constraints_expr(self):
        cnstr_expr_list = []
        lb_cnstr_expr_list = []  # lower bound
        ub_cnstr_expr_list = []  # upper bound
        time_var = self.skill_spec.time_var
        robot_var = self.skill_spec.robot_var
        robot_vel_var = self.skill_spec.robot_vel_var
        virtual_var = self.skill_spec.virtual_var
        virtual_vel_var = self.skill_spec.virtual_vel_var
        slack_var = self.skill_spec.slack_var
        n_slack = self.skill_spec.n_slack_var
        slack_ind = 0
        for cnstr in self.skill_spec.constraints:
            expr_size = cnstr.expression.size()
            # What's the partials?
            cnstr_expr = cnstr.jtimes(robot_var,
                                      robot_vel_var)
            if virtual_var is not None:
                cnstr_expr += cnstr.jtimes(virtual_var,
                                           virtual_vel_var)
            # Everyone wants a feedforward in numerics
            lb_cnstr_expr = -cnstr.jacobian(time_var)
            ub_cnstr_expr = -cnstr.jacobian(time_var)
            # Setup bounds based on type
            if isinstance(cnstr, EqualityConstraint):
                lb_cnstr_expr += -cs.mtimes(cnstr.gain, cnstr.expression)
                ub_cnstr_expr += -cs.mtimes(cnstr.gain, cnstr.expression)
            if isinstance(cnstr, SetConstraint):
                ub_cnstr_expr += cs.mtimes(cnstr.gain,
                                           cnstr.set_max - cnstr.expression)
                lb_cnstr_expr += cs.mtimes(cnstr.gain,
                                           cnstr.set_min - cnstr.expression)
            # Soft constraints have slack
            if n_slack > 0:
                if cnstr.constraint_type == "soft":
                    cnstr_expr += -slack_var[slack_ind:slack_ind+expr_size[0]]
                    slack_ind += expr_size[0]
            # Add to lists
            cnstr_expr_list += [cnstr_expr]
            lb_cnstr_expr_list += [lb_cnstr_expr]
            ub_cnstr_expr_list += [ub_cnstr_expr]
        cnstr_expr_full = cs.vertcat(*cnstr_expr_list)
        lb_cnstr_expr_full = cs.vertcat(*lb_cnstr_expr_list)
        ub_cnstr_expr_full = cs.vertcat(*ub_cnstr_expr_list)
        return cnstr_expr_full, lb_cnstr_expr_full, ub_cnstr_expr_full

    def setup_solver(self):
        full_cost_expr = self.get_regularised_cost_expr()
        cnstr_expr, lb_cnstr_expr, ub_cnstr_expr = self.get_constraints_expr()
        # Define externals
        time_var = self.skill_spec.time_var
        robot_var = self.skill_spec.robot_var
        list_par = [time_var, robot_var]
        list_names = ["time_var", "robot_var"]
        virtual_var = self.skill_spec.virtual_var
        if virtual_var is not None:
            list_par += [virtual_var]
            list_names += ["virtual_var"]
        input_var = self.skill_spec.input_var
        if input_var is not None:
            list_par += [input_var]
            list_names += ["input_var"]
        nlp_dict = {"x": self._opt_var,
                    "p": cs.vertcat(*list_par),
                    "f": full_cost_expr,
                    "g": cnstr_expr}
        self.solver = cs.nlpsol("solver",
                                self.options["solver_name"],
                                nlp_dict,
                                self.options["solver_opts"])

    def setup_problem_functions(self):
        full_cost_expr = self.get_regularised_cost_expr()
        cnstr_expr, lb_cnstr_expr, ub_cnstr_expr = self.get_constraints_expr()
        # Define external inputs
        time_var = self.skill_spec.time_var
        robot_var = self.skill_spec.robot_var
        list_vars = [time_var, robot_var]
        list_names = ["time_var", "robot_var"]
        virtual_var = self.skill_spec.virtual_var
        if virtual_var is not None:
            list_vars += [virtual_var]
            list_names += ["virtual_var"]
        input_var = self.skill_spec.input_var
        if input_var is not None:
            list_vars += [input_var]
            list_names += ["input_var"]
        # Cost and cnstr have opt_var in them
        cost_func = cs.Function("cost", list_vars+[self._opt_var],
                                [full_cost_expr], list_names+["opt_var"],
                                ["cost"])
        cnstr_func = cs.Function("cnstr", list_vars+[self._opt_var],
                                 [cnstr_expr], list_names+["opt_var"],
                                 ["cnstr"])
        # lb and ub are for numerics
        lb_cnstr_func = cs.Function("lb_cnstr", list_vars, [lb_cnstr_expr],
                                    list_names, ["lb_cnstr"])
        ub_cnstr_func = cs.Function("ub_cnstr", list_vars, [ub_cnstr_expr],
                                    list_names, ["ub_cnstr"])
        self.cost_func = cost_func
        self.cnstr_func = cnstr_func
        self.lb_cnstr_func = lb_cnstr_func
        self.ub_cnstr_func = ub_cnstr_func

    def solve(self, time_var, robot_var,
              virtual_var=None,
              input_var=None,
              opt_var0=None):
        currvals = [time_var, robot_var]
        if virtual_var is not None:
            currvals += [virtual_var]
        if input_var is not None:
            currvals += [input_var]
        lb_num = self.lb_cnstr_func(*currvals)
        ub_num = self.ub_cnstr_func(*currvals)
        if opt_var0 is None:
            opt_var0 = [0.]*self._n_opt_var
        self.res = self.solver(x0=opt_var0, ubg=ub_num,
                               lbg=lb_num, p=cs.vertcat(*currvals))
        nrob = self.skill_spec.n_robot_var
        nvirt = self.skill_spec.n_virtual_var
        if nvirt > 0:
            return self.res["x"][:nrob], self.res["x"][nrob:nrob+nvirt]
        else:
            return self.res["x"][:nrob]


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
        if options is None:
            options = {}
        self.options = options

    @property
    def skill_spec(self):
        """Get or set the skill specification. This also sets the number of
         modes."""
        return self._skill_spec

    @skill_spec.setter
    def skill_spec(self, spec):
        cnstr_count = spec.count_constraints()
        self.n_modes = 2**cnstr_count["set"]
        self._skill_spec = spec

    def get_in_tangent_cone_function(self, cnstr):
        """Returns a casadi function for the SetConstraint instance."""
        if not isinstance(cnstr, SetConstraint):
            raise TypeError("in_tangent_cone is only available for"
                            + " SetConstraint")
            return None
        time_var = self.skill_spec.time_var
        robot_var = self.skill_spec.time_var
        list_vars = [time_var, robot_var]
        list_names = ["time_var", "robot_var"]
        robot_vel_var = self.skill_spec.robot_vel_var
        opt_var = [robot_vel_var]
        virtual_var = self.skill_spec.virtual_var
        virtual_vel_var = self.skill_spec.virtual_vel_var
        input_var = self.skill_spec.input_var
        expr = cnstr.expression
        set_min = cnstr.set_min
        set_max = cnstr.set_max
        dexpr = cs.jacobian(expr, time_var)
        dexpr += cs.jtimes(expr, robot_var, robot_vel_var)
        if virtual_var is not None:
            list_vars += [virtual_var]
            list_names += ["virtual_var"]
            opt_var += [virtual_vel_var]
            dexpr += cs.jtimes(expr, virtual_var, virtual_vel_var)
        if input_var is not None:
            list_vars += [input_var]
            list_vars += ["input_var"]
        in_tc = cs.if_else(
            set_min < expr,
            cs.if_else(
                expr < set_max,
                True,
                cs.if_else(
                    dexpr < 0,
                    True,
                    False,
                    True),
                False,
                True),
            cs.if_else(
                dexpr > 0,
                True,
                False,
                True
            ),
            True
        )
        return cs.Function("in_tc_"+cnstr.label,
                           list_vars+opt_var,
                           [in_tc],
                           list_vars+["opt_var"],
                           ["in_tc_"+cnstr.label])

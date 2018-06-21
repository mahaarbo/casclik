"""Reactive NLP controller, see class doc.
"""

import casadi as cs
from constraints import EqualityConstraint, SetConstraint
from constraints import VelocityEqualityConstraint, VelocitySetConstraint
from base_controller import BaseController


class ReactiveNLPController(BaseController):
    """Reactive NLP Controller.

    The reactive NLP controller can handle skills with input. It
    defines a nonlinear problem with constraints from the skill
    specification. The robot_var, virtual_var, and input_var in the
    expressions are handled as parameters to the solver. The
    constraints are constructed from the skill_specification similarly
    as it is in the ReactiveQPController. The main usage for this
    controller is when you have a nonlinear cost function you want to
    employ. As with the ReactiveQPController, you can overload your
    own functions for the cost if you have it in casadi compatible
    external format.

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
        """Get or set the slack_var_weights. Can be list np.ndarray, cs.MX,
        but needs to have the same length"""
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

    @property
    def options(self):
        """Get or set the options. See ReactiveNLPController.options_info.  Or
        read @options.setter in source code. Basically we stop solvers
        from printing, and allow SQP methods to warmstart.

        """
        return self._options

    @options.setter
    def options(self, opt):
        if opt is None:
            opt = {}
        if "solver_name" not in opt:
            opt["solver_name"] = "ipopt"
        if "solver_opts" not in opt:
            opt["solver_opts"] = {}

        solver_opts = opt["solver_opts"]
        if "print_time" not in solver_opts:
            solver_opts["print_time"] = False

        if opt["solver_name"] == "blocksqp":
            if "print_header" not in solver_opts:
                solver_opts["print_header"] = False
            if "print_iteration" not in solver_opts:
                solver_opts["print_iteration"] = False
            if "warmstart" not in solver_opts:
                solver_opts["warmstart"] = False
            if "qpsol" not in solver_opts:
                solver_opts["qpsol"] = "qpoases"
            if "qpsol_options" not in solver_opts:
                solver_opts["qpsol_options"] = {"printLevel": "none"}

        if opt["solver_name"] == "ipopt":
            if "ipopt" not in solver_opts:
                solver_opts["ipopt"] = {}
                # ipopt options are a bit weird
                # see documentation on nlpsol
            if "print_level" not in solver_opts["ipopt"]:
                solver_opts["ipopt"]["print_level"] = 0
            if "jit" not in solver_opts:
                solver_opts["jit"] = True
            if "jit_options" not in solver_opts:
                solver_opts["jit_options"] = {"flags": "-Ofast"}

        if opt["solver_name"] == "scpgen":
            if "print_header" not in solver_opts:
                solver_opts["print_header"] = False

        if opt["solver_name"] == "sqpmethod":
            if "print_header" not in solver_opts:
                solver_opts["print_header"] = False
            if "print_iteration" not in solver_opts:
                solver_opts["print_iteration"] = False
            if "qpsol" not in solver_opts:
                solver_opts["qpsol"] = "qpoases"
            if "qpsol_options" not in solver_opts:
                solver_opts["qpsol_options"] = {"printLevel": "none"}
        self._options = opt

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
            elif isinstance(cnstr, SetConstraint):
                ub_cnstr_expr += cs.mtimes(cnstr.gain,
                                           cnstr.set_max - cnstr.expression)
                lb_cnstr_expr += cs.mtimes(cnstr.gain,
                                           cnstr.set_min - cnstr.expression)
            elif isinstance(cnstr, VelocityEqualityConstraint):
                ub_cnstr_expr += cnstr.target
                lb_cnstr_expr += cnstr.target
            elif isinstance(cnstr, VelocitySetConstraint):
                ub_cnstr_expr += cnstr.set_max
                lb_cnstr_expr += cnstr.set_min
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

    def solve_initial_problem(self, robot_var0):
        """Solves the initial problem, finding slack and virtual variables."""
        raise NotImplementedError("To be done")

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

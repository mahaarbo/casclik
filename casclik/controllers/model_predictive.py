"""Model predictive controller, see class doc.
"""

import casadi as cs
from casclik.constraints import EqualityConstraint, SetConstraint
from casclik.constraints import VelocityEqualityConstraint, VelocitySetConstraint
from casclik.controllers.base_controller import BaseController

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
    solver_opt (dict): solver options, see casadi.
    cost_integration_method (str): "euler" or "rk4", default="euler".
    constraint_integration_method (str): "euler" or "rk4", default="euler".
    """
    weight_shifter = 0.001

    def __init__(self, skill_spec,
                 cost_expr,
                 horizon_length,
                 timestep=0.01,
                 slack_var_weights=None,
                 options=None):
        self.skill_spec = skill_spec
        self.cost_expression = cost_expr
        self.horizon_length = horizon_length
        self.timestep = timestep
        self.slack_var_weights = slack_var_weights
        self.options = options

    @property
    def cost_expression(self):
        """Get or set the cost expression."""
        return self._cost_expression

    @cost_expression.setter
    def cost_expression(self, expr):
        self._cost_expression = expr

    @property
    def horizon_length(self):
        """Get or set the horizon length. Checks if it's an int."""
        return self._horizon_length

    @horizon_length.setter
    def horizon_length(self, hl):
        if not isinstance(hl, int):
            raise TypeError("Horizon length must be int, currently it is"
                            + " " + str(type(hl) + "."))
        self._horizon_length = hl

    @property
    def timestep(self):
        """Get or set the timestep length. Checks if it's a float."""
        return self._timestep

    @timestep.setter
    def timestep(self, dt):
        if not isinstance(dt, float):
            raise TypeError("Timestep must be float, currently it is"
                            + " " + str(type(dt)) + ".")
        self._timestep = dt

    @property
    def slack_var_weights(self):
        """Get or set the slack_var_weights. Can be list, np.dnarray, cs.MX."""
        return self._slack_var_weights

    @slack_var_weights.setter
    def slack_var_weights(self, weights):
        ns = self.skill_spec.n_slack_var
        if weights is None:
            weights = [1.]*ns
        elif isinstance(weights, cs.MX) and weights.size()[0] != ns:
            raise ValueError("slack_var_weights and slack_var dimensions"
                             + " do not match")
        elif isinstance(weights, cs.DM) and weights.size()[0] != ns:
            raise ValueError("slack_var_weights and slack_var dimensions"
                             + " do not match")
        elif len(weights) != ns:
            raise ValueError("slack_var_weights and slack_var dimensions"
                             + " do not match")
        self._slack_var_weigths = weights

    @property
    def skill_spec(self):
        """Get or set the skill_spec. Automatically sets self._opt_var, and
        sanity checks for input"""
        return self._skill_spec

    @skill_spec.setter
    def skill_spec(self, spec):
        if spec.input_var is not None:
            raise TypeError("ModelPredictiveController can only handle skills"
                            + " without input_var, " + spec.label + " has "
                            + str(spec.input_var) + " as input_var.")
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
        """Get or set the options. See ModelPredictiveController.options_info.
        Or read@options.setter in source code. Basically we stop solvers from
        printing and allow SQP methods to warmstart.
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
                solver_opts["jit_options"] = {#"compiler": "gcc",
                                              "flags": "-Ofast"}

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

    def get_regularised_cost_integrand_expr(self):
        slack_var = self.skill_spec.slack_var
        if slack_var is not None:
            nslack = self.skill_spec.n_slack_var
            slack_H = cs.diag(self.slack_var_weights)
            slack_H += self.weight_shifter*cs.MX.eye(nslack)
            slack_cost = cs.mtimes(cs.mtimes(slack_var.T, slack_H), slack_var)
        else:
            return self.cost_expression
        return self.weight_shifter*self.cost_expression + slack_cost

    def get_cost_function(self):
        cost_integrand_expr = self.get_regularised_cost_integrand_expr()

    def get_constraints_expr(self):
        cnstr_expr_list = []
        lb_cnstr_expr_list = []
        ub_cnstr_expr_list = []
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
            cnstr_expr = cnstr.jtimes(robot_var,
                                      robot_vel_var)
            if virtual_var is not None:
                cnstr_expr += cnstr.jtimes(virtual_var,
                                           virtual_vel_var)
            lb_cnstr_expr = -cnstr.jacobian(time_var)
            ub_cnstr_expr = -cnstr.jacobian(time_var)
            if isinstance(cnstr, EqualityConstraint):
                ub_cnstr_expr += -cs.mtimes(cnstr.gain, cnstr.expression)
                lb_cnstr_expr += -cs.mtimes(cnstr.gain, cnstr.expression)
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

            if n_slack > 0:
                if cnstr.constraint_type == "soft":
                    cnstr_expr += -slack_var[slack_ind:slack_ind+expr_size[0]]
                    slack_ind += expr_size[0]
            cnstr_expr_list += [cnstr_expr]
            ub_cnstr_expr_list += [ub_cnstr_expr]
            lb_cnstr_expr_list += [lb_cnstr_expr]
        cnstr_expr_full = cs.vertcat(*cnstr_expr_list)
        ub_cnstr_expr_full = cs.vertcat(*ub_cnstr_expr_list)
        lb_cnstr_expr_full = cs.vertcat(*lb_cnstr_expr_list)
        return cnstr_expr_full, lb_cnstr_expr_full, ub_cnstr_expr_full

    def setup_problem_functions(self):
        full_cost_expr = self.get_regularised_cost_expr()
        cnstr_expr, lb_cnstr_expr, ub_cnstr_expr = self.get_constraints_expr()
        time_var = self.skill_spec.time_var
        robot_var = self.skill_spec.robot_var
        list_vars = [time_var, robot_var]
        list_names = ["time_var", "robot_var"]
        cntrl_vars = [self.skill_spec.robot_vel_var]
        cntrl_names = ["robot_vel_var"]
        virtual_var = self.skill_spec.virtual_var
        if virtual_var is not None:
            list_vars += [virtual_var]
            list_names += ["virtual_var"]
            cntrl_vars += [self.skill_spec.virtual_vel_var]
            cntrl_names += ["virtual_vel_var"]
        list_vars += cntrl_var
        list_names += cntrl_names
        function_options = self.options["function_opts"]
        cost_func = cs.Function("cost", list_vars, [full_cost_expr],
                                list_names, ["cost"],
                                function_options)
        cnstr_func = cs.Function("cnstr", list_vars, [cnstr_expr],
                                 list_names, ["cnstr"],
                                 function_options)
        lb_cnstr_func = cs.Function("lb_cnstr", list_vars, [lb_cnstr_expr],
                                    list_names, ["lb_cnstr"],
                                    function_options)
        ub_cnstr_func = cs.Function("ub_cnstr", list_vars, [ub_cnstr_expr],
                                    list_names, ["ub_cnstr"],
                                    function_options)
        self.cost_func = cost_func
        self.cnstr_func = cnstr_func
        self.lb_cnstr_func = lb_cnstr_func
        self.ub_cnstr_func = ub_cnstr_func
        return (cost_func, cnstr_func, lb_cnstr_func, ub_cnstr_func)
    
    def setup_solver(self):
        all_funcs = self.setup_problem_functions()
        cost_func = all_funcs[0]
        cnstr_func = all_funcs[1]
        lb_cnstr_func = all_funcs[2]
        ub_cnstr_func = all_funcs[3]
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
        
    def setup_problem_functions(self):
        pass

    def solve(self, time_var, robot_var,
              virtual_var=None,
              opt_var0=None):
        currvals = [time_var, robot_var]
        if virtual_var is not None:
            currvals += [virtual_var]
        lb_num = self.lb_cnstr_func(*currvals)
        ub_num = self.ub_cnstr_func(*currvals)
        if opt_var0 is None:
            opt_var0 = [0.]*self.n_opt_var*self.horizon_length
        else:
            raise TypeError("TODO")
        self.res = self.solver(x0=opt_var0, ubg=ub_num,
                               lbg=lb_num, p=cs.vertcat(*currvals))
        nrob = self.skill_spec.n_robot_var
        nvirt = self.skill_spec.n_virtual_var
        if nvirt > 0:
            return self.res["x"][:nrob], self.res["x"][nrob:nrob+nvirt]
        else:
            return self.res["x"][:nrob]

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
    be unknown before they occur. It is based on the reactive NLP
    controller, but considers multiple steps ahead instead of just
    one.

    Args:
        skill_spec (SkillSpecification): skill specification
        cost_expr (cs.MX): expression of the singlestep cost
        horizon_length (int): horizon_length, default=5
        timestep (float): integration length, default=0.01
        slack_var_weights (list): weights in singlestep nlp, defaults to 1
        options (dict): options dictionary, see self.options_info
        """
    controller_type = "ModelPredictiveController"
    options_info = """TODO
    solver_opt (dict): solver options, see casadi.
    cost_integration_method (str): rectangle, trapezoidal, or simpson. default=trapezoidal.
    """
    weight_shifter = 0.001

    def __init__(self, skill_spec,
                 cost_expr=None,
                 horizon_length=5,
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
        if expr is None:
            robot_vel_var = self.skill_spec.robot_vel_var
            expr = cs.dot(robot_vel_var, robot_vel_var)
        has_r = cs.jacobian(expr, self.skill_spec.robot_var).nnz() > 0
        has_rvel = cs.jacobian(expr, self.skill_spec.robot_vel_var).nnz() > 0
        if self.skill_spec._has_virtual:
            has_v = cs.jacobian(expr, self.skill_spec.virtual_var).nnz() > 0
            has_vvel = cs.jacobian(expr, self.skill_spec.virtual_vel_var).nnz() > 0
        if not (has_r or has_rvel or has_v or has_vvel):
            raise ValueError("Cost must be an expression containing "
                             + "robot_var, robot_var_vel, virtual_var,"
                             + " or virtual_vel_var.")
        if expr.size()[0] != 1:
            raise ValueError("Cost must be a scalar. You have size(cost_expr)="
                             + str(expr.size()))
        if expr.size()[1] != 1:
            raise ValueError("Cost must be a scalar. You have size(cost_expr)="
                             + str(expr.size()))
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
        if weights is None:
            weights = cs.vertcat([1.]*self.skill_spec.n_slack_var)
        elif isinstance(weights, cs.GenericCommonMatrix):
            if weights.size2 != 1:
                raise ValueError("slack_var_weights must be a vector.")
            elif weights.size1 != self.skill_spec.n_slack_var:
                raise ValueError("slack_var_weights and slack_var dimensions"
                                 + " do not match.")
        elif isinstance(weights, (list, cs.np.ndarray)):
            if len(weights) != self.skill_spec.n_slack_var:
                raise ValueError("slack_var_weights and slack_var dimensions"
                                 + " do not match")
            weights = cs.vertcat(weights)
        self._slack_var_weights = weights

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
        if "cost_integration_method" not in opt:
            opt["cost_integration_method"] = "rectangle"
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

        if "function_opts" not in opt:
            opt["function_opts"] = {}
        function_opts = opt["function_opts"]
        if "jit" not in function_opts:
            function_opts["jit"] = True
        if "compiler" not in function_opts:
            function_opts["compiler"] = "shell"
        if "print_time" not in function_opts:
            function_opts["print_time"] = False
        if "jit_options" not in function_opts:
            function_opts["jit_options"] = {"flags": "-O2"}
        self._options = opt

    def get_cost_integrand_function(self):
        """Returns a casadi function for the discretized integrand of
        the cost expression integrated one timestep. For the rectangle
        method, this just amounts to timing by the timestep.

        As with the other controllers, the cost is affected by the
        weight shifter, giving a regularised cost with the slack
        variables.
        """
        # Setup new symbols needed
        dt = self.timestep
        # Setup skill_spec symbols
        time_var = self.skill_spec.time_var
        robot_var = self.skill_spec.robot_var
        list_vars = [time_var, robot_var]
        list_names = ["time_var", "robot_var"]
        robot_vel_var = self.skill_spec.robot_vel_var
        cntrl_vars = [robot_vel_var]
        cntrl_names = ["robot_vel_var"]
        virtual_var = self.skill_spec.virtual_var
        if virtual_var is not None:
            list_vars += [virtual_var]
            list_names += ["virtual_var"]
            virtual_vel_var = self.skill_spec.virtual_vel_var
            cntrl_vars += [virtual_vel_var]
            cntrl_names += ["virtual_vel_var"]
        slack_var = self.skill_spec.slack_var
        if slack_var is not None:
            list_vars += [slack_var]
            list_names += ["slack_var"]
        # Full symbol list same way as in other controllers
        list_vars += cntrl_vars
        list_names += cntrl_names

        # Expression for the cost with regularisation:
        if slack_var is not None:
            slack_H = cs.diag(self.weight_shifter + self.slack_var_weights)
            slack_cost = cs.mtimes(cs.mtimes(slack_var.T, slack_H), slack_var)
            regularised_cost = self.weight_shifter*self.cost_expression
            regularised_cost += slack_cost
        else:
            regularised_cost = self.cost_expression
        # Choose integration method
        if self.options["cost_integration_method"].lower() == "rectangle":
            cost_integrand = regularised_cost*dt
        elif self.options["cost_integration_method"].lower() == "trapezoidal":
            # Trapezoidal rule
            raise NotImplementedError("Trapezoidal rule integration not"
                                      + " implemented.")
        elif self.options["cost_integration_method"].lower() == "simpson":
            # Simpson rule
            raise NotImplementedError("Simpson rule integration not"
                                      + " implemented.")
        else:
            raise NotImplementedError(self.options["cost_integration_method"]
                                      + " is not a known integration method.")
        return cs.Function("fc_k", list_vars, [cost_integrand],
                           list_names, ["cost_integrand"])

    def get_reactive_cnstr_functions(self):
        """The constraints associated with a skill are split into
        two. The reactive part and the predictive part. The reactive
        part has large parts being numeric, the predictive part is
        pretty much only symbolic. Because of this we do them separately
        and generate the expressions and functions in one go.
        """
        cnstr_expr_list = []
        lb_cnstr_expr_list = []
        ub_cnstr_expr_list = []
        # Setup skill_spec symbols
        time_var = self.skill_spec.time_var
        robot_var = self.skill_spec.robot_var
        list_vars = [time_var, robot_var]
        list_pars = [time_var, robot_var]
        list_names = ["time_var", "robot_var"]
        list_par_names = ["measured_time_var0", "measured_robot_var0"]
        robot_vel_var = self.skill_spec.robot_vel_var
        cntrl_vars = [robot_vel_var]
        cntrl_names = ["robot_vel_var"]
        virtual_var = self.skill_spec.virtual_var
        if virtual_var is not None:
            list_vars += [virtual_var]
            list_pars += [virtual_var]
            list_names += ["virtual_var"]
            list_par_names += ["measured_virtual_var0"]
            virtual_vel_var = self.skill_spec.virtual_vel_var
            cntrl_vars += [virtual_vel_var]
            cntrl_names += ["virtual_vel_var"]
        slack_var = self.skill_spec.slack_var
        if slack_var is not None:
            list_vars += [slack_var]
            list_names += ["slack_var"]
        n_slack = self.skill_spec.n_slack_var
        slack_ind = 0
        # Full symbol list same way as in other controllers
        list_vars += cntrl_vars
        list_names += cntrl_names
        # Loop over the constraints
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
                lb_cnstr_expr += cs.mtimes(cnstr.gain,
                                           cnstr.set_min - cnstr.expression)
                ub_cnstr_expr += cs.mtimes(cnstr.gain,
                                           cnstr.set_max - cnstr.expression)
            elif isinstance(cnstr, VelocityEqualityConstraint):
                lb_cnstr_expr += cnstr.target
                ub_cnstr_expr += cnstr.target
            elif isinstance(cnstr, VelocitySetConstraint):
                lb_cnstr_expr += cnstr.set_min
                ub_cnstr_expr += cnstr.set_max
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
        cnstr_func = cs.Function("fcnstr", list_vars, [cnstr_expr_full],
                                 list_names, ["cnstr"])
        lb_cnstr_func = cs.Function("fcnstr_lb", list_pars,
                                    [lb_cnstr_expr_full],
                                    list_par_names, ["cnstr_lb"])
        ub_cnstr_func = cs.Function("fcnstr_ub", list_pars,
                                    [ub_cnstr_expr_full],
                                    list_par_names, ["cnstr_ub"])
        return cnstr_func, lb_cnstr_func, ub_cnstr_func

    def get_predictive_cnstr_functions(self):
        """The constraints associated with a skill are split into
        two. The reactive part and the predictive part. The reactive
        part has large parts being numeric, the predictive part is
        pretty much only symbolic. Because of this we do them separately
        and generate the expressions and functions in one go.
        """
        cnstr_expr_list = []
        lb_cnstr_expr_list = []
        ub_cnstr_expr_list = []
        # Setup skill_spec symbols
        time_var = self.skill_spec.time_var
        robot_var = self.skill_spec.robot_var
        list_vars = [time_var, robot_var]
        list_names = ["time_var", "robot_var"]
        robot_vel_var = self.skill_spec.robot_vel_var
        cntrl_vars = [robot_vel_var]
        cntrl_names = ["robot_vel_var"]
        virtual_var = self.skill_spec.virtual_var
        if virtual_var is not None:
            list_vars += [virtual_var]
            list_names += ["virtual_var"]
            virtual_vel_var = self.skill_spec.virtual_vel_var
            cntrl_vars += [virtual_vel_var]
            cntrl_names += ["virtual_vel_var"]
        slack_var = self.skill_spec.slack_var
        if slack_var is not None:
            list_vars += [slack_var]
            list_names += ["slack_var"]
        n_slack = self.skill_spec.n_slack_var
        slack_ind = 0
        # Full symbol list same way as in other controllers
        list_vars += cntrl_vars
        list_names += cntrl_names
        # Loop over the constraints
        for cnstr in self.skill_spec.constraints:
            expr_size = cnstr.expression.size()
            # Here we construct the whole time derivative
            cnstr_expr = cnstr.jacobian(time_var)
            cnstr_expr += cnstr.jtimes(robot_var,
                                       robot_vel_var)
            if virtual_var is not None:
                cnstr_expr += cnstr.jtimes(virtual_var,
                                           virtual_vel_var)
            cnstr_expr2 = None  # We will need this in the sets
            # Then we evaluate the type of constraint
            if isinstance(cnstr, EqualityConstraint):
                cnstr_expr += cs.mtimes(cnstr.gain, cnstr.expression)
                lb_cnstr_expr = [0.0]*expr_size[0]
                ub_cnstr_expr = [0.0]*expr_size[0]
            elif isinstance(cnstr, SetConstraint):
                # This we have to split into two. This is because we're not
                # guaranteed that the user has chosen a numeric gain. And if we
                # are to support differing lower and upper gains in the future,
                # we have to split it into two.
                cnstr_expr2 = cnstr_expr
                # cnstr_expr is lower, cnstr_expr2 is upper

                # Avoid infs in evaluated constraint expressions
                if isinstance(cnstr.set_min, (list, cs.np.ndarray)):
                    set_min = cnstr.set_min
                    for idx, item in enumerate(cnstr.set_min):
                        set_min[idx] = max(cnstr.set_min[idx], -1e20)
                elif isinstance(cnstr.set_min, (int, float)):
                    set_min = max(cnstr.set_min, -1e20)
                if isinstance(cnstr.set_max, (list, cs.np.ndarray)):
                    set_max = cnstr.set_max
                    for idx, item in enumerate(cnstr.set_max):
                        set_max[idx] = min(cnstr.set_max[idx], 1e20)
                elif isinstance(cnstr.set_max, (int, float)):
                    set_max = min(cnstr.set_max, 1e20)

                cnstr_expr += - cs.mtimes(cnstr.gain,
                                          set_min - cnstr.expression)
                cnstr_expr2 += - cs.mtimes(cnstr.gain,
                                           set_max - cnstr.expression)
                lb_cnstr_expr = [0.0]*expr_size[0]
                ub_cnstr_expr = [cs.inf]*expr_size[0]
                lb_cnstr_expr2 = [-cs.inf]*expr_size[0]
                ub_cnstr_expr2 = [0.0]*expr_size[0]
            elif isinstance(cnstr, VelocityEqualityConstraint):
                lb_cnstr_expr = cnstr.target
                ub_cnstr_expr = cnstr.target
            elif isinstance(cnstr, VelocitySetConstraint):
                lb_cnstr_expr = cnstr.set_min
                ub_cnstr_expr = cnstr.set_max
            # Soft constraints have slack
            if n_slack > 0:
                if cnstr.constraint_type == "soft":
                    cnstr_expr += -slack_var[slack_ind:slack_ind+expr_size[0]]
                    slack_ind += expr_size[0]
            # Add to lists
            cnstr_expr_list += [cnstr_expr]
            lb_cnstr_expr_list += [lb_cnstr_expr]
            ub_cnstr_expr_list += [ub_cnstr_expr]
            if cnstr_expr2 is not None:
                cnstr_expr_list += [cnstr_expr2]
                lb_cnstr_expr_list += [lb_cnstr_expr2]
                ub_cnstr_expr_list += [ub_cnstr_expr2]
        # Went through all the constraints, finish up
        cnstr_expr_full = cs.vertcat(*cnstr_expr_list)
        lb_cnstr_expr_full = cs.vertcat(*lb_cnstr_expr_list)
        ub_cnstr_expr_full = cs.vertcat(*ub_cnstr_expr_list)
        cnstr_func = cs.Function("fcnstr_pred", list_vars, [cnstr_expr_full],
                                 list_names, ["cnstr"])
        lb_cnstr_func = cs.Function("fcnstr_lb_pred", [],
                                    [lb_cnstr_expr_full],
                                    [], ["cnstr_lb"])
        ub_cnstr_func = cs.Function("fcnstr_ub_pred", [],
                                    [ub_cnstr_expr_full],
                                    [], ["cnstr_ub"])
        return cnstr_func, lb_cnstr_func, ub_cnstr_func

    def setup_problem_functions(self):
        """Sets up the relevant casadi functions and expressions for the
        MPC NLP problem.
        """
        # Setup relevant functions for each single timestep
        fcost_integrand = self.get_cost_integrand_function()
        all_cnstr_funcs_reactive = self.get_reactive_cnstr_functions()
        fcnstr_reactive = all_cnstr_funcs_reactive[0]
        flb_cnstr_reactive = all_cnstr_funcs_reactive[1]
        fub_cnstr_reactive = all_cnstr_funcs_reactive[2]
        all_cnstr_funcs_predictive = self.get_predictive_cnstr_functions()
        fcnstr_predictive = all_cnstr_funcs_predictive[0]
        flb_cnstr_predictive = all_cnstr_funcs_predictive[1]
        fub_cnstr_predictive = all_cnstr_funcs_predictive[2]
        # The reactive constraint functions are for choosing the first cntrl
        # input, the predictive ones are the ones for the subsequent steps.

        # Important sizes
        nrob = self.skill_spec.n_robot_var
        nvirt = self.skill_spec.n_virtual_var
        nslack = self.skill_spec.n_slack_var
        dt = self.timestep

        # Where the MPC problem formulation is stored:
        mpc_opt_vars = []
        mpc_cnstrs = []
        mpc_cnstr_lb = []
        mpc_cnstr_ub = []
        mpc_cost = 0.0

        # Initial step
        time_var0 = cs.MX.sym("time_var0")
        robot_var_k = cs.MX.sym("robot_var0", nrob)
        list_vars_k = [time_var0, robot_var_k]
        mpc_opt_vars += [time_var0, robot_var_k]
        if nvirt > 0:
            virtual_var_k = cs.MX.sym("virtual_var0", nvirt)
            list_vars_k += [virtual_var_k]
            mpc_opt_vars += [virtual_var_k]
        # Lift initial conditions
        measured_time_var0 = cs.MX.sym("mtime_var0")
        mpc_cnstrs += [time_var0]
        mpc_cnstr_lb += [measured_time_var0]
        mpc_cnstr_ub += [measured_time_var0]
        measured_robot_var0 = cs.MX.sym("mrobot_var0", nrob)
        mpc_cnstrs += [robot_var_k]
        mpc_cnstr_lb += [measured_robot_var0]
        mpc_cnstr_ub += [measured_robot_var0]
        list_pars = [measured_time_var0, measured_robot_var0]
        list_par_names = ["time_var0", "robot_var0"]
        if nvirt > 0:
            # The first time this is run we will use the initial solver The
            # subsequent times it will use the previous estimated one?  Or
            # should we solve the initial solver once before every control
            # step?
            measured_virtual_var0 = cs.MX.sym("mvirtual_var0", nvirt)
            mpc_cnstrs += [virtual_var_k]
            mpc_cnstr_lb += [measured_virtual_var0]
            mpc_cnstr_ub += [measured_virtual_var0]
            list_pars += [measured_virtual_var0]
            list_par_names += ["virtual_var0"]
        # Loop over the horizon
        for k in range(self.horizon_length):
            # Control input this step
            cntrl_vars_k = []
            if nslack > 0:
                slack_var_k = cs.MX.sym("slack_var"+str(k), nslack)
                cntrl_vars_k += [slack_var_k]
            robot_vel_var_k = cs.MX.sym("robot_vel_var"+str(k), nrob)
            cntrl_vars_k += [robot_vel_var_k]
            if nvirt > 0:
                virtual_vel_var_k = cs.MX.sym("virtual_vel_var"+str(k), nvirt)
                cntrl_vars_k += [virtual_vel_var_k]
            mpc_opt_vars += cntrl_vars_k
            # Cost for step
            mpc_cost += fcost_integrand(*(list_vars_k+cntrl_vars_k))
            # Task constraints
            if k == 0:
                mpc_cnstrs += [fcnstr_reactive(*(list_vars_k+cntrl_vars_k))]
                mpc_cnstr_lb += [flb_cnstr_reactive(*(list_pars))]
                mpc_cnstr_ub += [fub_cnstr_reactive(*(list_pars))]
            else:
                mpc_cnstrs += [fcnstr_predictive(*(list_vars_k+cntrl_vars_k))]
                mpc_cnstr_lb += [flb_cnstr_predictive()["cnstr_lb"]]
                mpc_cnstr_ub += [fub_cnstr_predictive()["cnstr_ub"]]
            # Prediction step
            robot_var_p = robot_var_k + robot_vel_var_k*dt
            if nvirt > 0:
                virtual_var_p = virtual_var_k + virtual_vel_var_k*dt
            # Symbols for states in next step
            robot_var_k = cs.MX.sym("robot_var"+str(k+1), nrob)
            list_vars_k = [time_var0+dt*(k+1), robot_var_k]
            mpc_opt_vars += [robot_var_k]
            if nvirt > 0:
                virtual_var_k = cs.MX.sym("virtual_var"+str(k+1), nvirt)
                list_vars_k += [virtual_var_k]
                mpc_opt_vars += [virtual_var_k]
            # Shooting gap constraint
            mpc_cnstrs += [robot_var_p - robot_var_k]
            mpc_cnstr_lb += [0.]*nrob
            mpc_cnstr_ub += [0.]*nrob
            if nvirt > 0:
                mpc_cnstrs += [virtual_var_p - virtual_var_k]
                mpc_cnstr_lb += [0.]*nvirt
                mpc_cnstr_ub += [0.]*nvirt
        # And thus we have peered across the horizon and seen it all
        mpc_cnstrs_expr = cs.vertcat(*mpc_cnstrs)
        mpc_cnstr_lb_expr = cs.vertcat(*mpc_cnstr_lb)
        mpc_cnstr_ub_expr = cs.vertcat(*mpc_cnstr_ub)
        mpc_opt_vars_expr = cs.vertcat(*mpc_opt_vars)
        # Let's make functions for lb and ub
        mpc_cnstr_lb_func = cs.Function("lb_cnstr", list_pars,
                                        [mpc_cnstr_lb_expr],
                                        list_par_names, ["lb_cnstr"],
                                        self.options["function_opts"])
        mpc_cnstr_ub_func = cs.Function("ub_cnstr", list_pars,
                                        [mpc_cnstr_ub_expr],
                                        list_par_names, ["ub_cnstr"],
                                        self.options["function_opts"])
        self.mpc_problem = {
            "nlp": {
                "x": mpc_opt_vars_expr,
                "f": mpc_cost,
                "g": mpc_cnstrs_expr
            },
            "num": {
                "lb": mpc_cnstr_lb_func,
                "ub": mpc_cnstr_ub_func
            }
        }

    def setup_solver(self):
        # Setup relevant functions and expressions
        self.setup_problem_functions()
        self.solver = cs.nlpsol("solver",
                                self.options["solver_name"],
                                self.mpc_problem["nlp"],
                                self.options["solver_opts"])

    def get_horizons(self):
        """Returns a tuple of the desired inputs and the predicted states.
        Can only be called after solve
        """
        # The resulting decision variables of the NLP:
        nlp_opt = self.res["x"]
        # Some important sizes
        nrob = self.skill_spec.n_robot_var
        nvirt = self.skill_spec.n_virtual_var
        nslack = self.skill_spec.n_slack_var
        res_rob = []
        res_rob_vel = []
        res_virt = []
        res_virt_vel = []
        res_slack = []
        # The index starts after t0
        idx = 1
        res_rob += [nlp_opt[idx:idx+nrob]]
        idx += nrob
        if nvirt > 0:
            res_virt += [nlp_opt[idx:idx+nvirt]]
            idx += nvirt
        for i in range(self.horizon_length):
            if nslack > 0:
                res_slack += [nlp_opt[idx:idx+nslack]]
                idx += nslack
            res_rob_vel += [nlp_opt[idx:idx+nrob]]
            idx += nrob
            if nvirt > 0:
                res_virt_vel += [nlp_opt[idx:idx+nvirt]]
                idx += nvirt
            res_rob += [nlp_opt[idx:idx+nrob]]
            idx += nrob
            if nvirt > 0:
                res_virt += [nlp_opt[idx:idx+nvirt]]
                idx += nvirt
        res = [res_rob,
               res_rob_vel,
               res_virt,
               res_virt_vel,
               res_slack]
        return res

    def solve(self, time_var, robot_var,
              virtual_var=None,
              opt_var0=None):
        currvals = [time_var, robot_var]
        if virtual_var is not None:
            currvals += [virtual_var]
        lb_num = self.mpc_problem["num"]["lb"](*currvals)
        ub_num = self.mpc_problem["num"]["ub"](*currvals)
        if opt_var0 is None:
            opt_var0 = [0.]*self.mpc_problem["nlp"]["x"].shape[0]
        else:
            raise NotImplementedError("Warm start of opt_var to be done.")
        self.res = self.solver(x0=opt_var0, ubg=ub_num,
                               lbg=lb_num)
        # Sizes:
        nrob = self.skill_spec.n_robot_var
        if self.skill_spec._has_virtual:
            # Handles user error when user adds virtual var
            # but it's not actually in the expressions
            nvirt = self.skill_spec.n_virtual_var
        else:
            nvirt = 0
        nslack = self.skill_spec.n_slack_var
        # Index after the lifted initial conditions:
        des_ind = 1 + nrob + nvirt + nslack
        # Get results:
        res_robot_vel = self.res["x"][des_ind:des_ind+nrob]
        des_ind += nrob
        if nvirt > 0 and self.skill_spec._has_virtual:
            res_virtual_vel = self.res["x"][des_ind:des_ind+nvirt]
            des_ind += nvirt
        else:
            res_virtual_vel = None
        if nslack > 0:
            res_slack = self.res["x"][des_ind:des_ind+nslack]
        return res_robot_vel, res_virtual_vel, res_slack

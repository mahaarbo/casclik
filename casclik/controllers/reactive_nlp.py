"""Reactive NLP controller, see class doc.
"""

import casadi as cs
from casclik.constraints import EqualityConstraint, SetConstraint
from casclik.constraints import VelocityEqualityConstraint, VelocitySetConstraint
from casclik.controllers.base_controller import BaseController


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
        has_r = cs.jacobian(expr, self.skill_spec.robot_var).nnz() > 0
        has_rvel = cs.jacobian(expr, self.skill_spec.robot_vel_var).nnz() > 0
        if self.skill_spec._has_virtual:
            has_v = cs.jacobian(expr, self.skill_spec.virtual_var).nnz() > 0
            has_vvel = cs.jacobian(expr, self.skill_spec.virtual_vel_var).nnz() > 0
        if not (has_r or has_rvel or has_v or has_vvel):
            raise ValueError("Cost must be an expression containing "
                             + "robot_var, robot_var_vel, virtual_var,"
                             + " or virtual_vel_var.")
        self._cost_expression = expr

    @property
    def slack_var_weights(self):
        """Get or set the slack_var_weights. Can be list np.ndarray, cs.MX,
        but needs to have the same length"""
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
            function_opts["jit_options"] = {"compiler": "gcc",
                                            "flags": "-O2"}
        self._options = opt

    def get_regularised_cost_expr(self):
        slack_var = self.skill_spec.slack_var
        if slack_var is not None:
            slack_H = cs.diag(self.weight_shifter + self.slack_var_weights)
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
        if virtual_var is not None and self.skill_spec._has_virtual:
            list_par += [virtual_var]
            list_names += ["virtual_var"]
        input_var = self.skill_spec.input_var
        if input_var is not None and self.skill_spec._has_input:
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
        if virtual_var is not None and self.skill_spec._has_virtual:
            list_vars += [virtual_var]
            list_names += ["virtual_var"]
        input_var = self.skill_spec.input_var
        if input_var is not None and self.skill_spec.has_input:
            list_vars += [input_var]
            list_names += ["input_var"]
        # Cost and cnstr have opt_var in them
        cost_func = cs.Function("cost", list_vars+[self._opt_var],
                                [full_cost_expr], list_names+["opt_var"],
                                ["cost"], self.options["function_opts"])
        cnstr_func = cs.Function("cnstr", list_vars+[self._opt_var],
                                 [cnstr_expr], list_names+["opt_var"],
                                 ["cnstr"], self.options["function_opts"])
        # lb and ub are for numerics
        lb_cnstr_func = cs.Function("lb_cnstr", list_vars, [lb_cnstr_expr],
                                    list_names, ["lb_cnstr"],
                                    self.options["function_opts"])
        ub_cnstr_func = cs.Function("ub_cnstr", list_vars, [ub_cnstr_expr],
                                    list_names, ["ub_cnstr"],
                                    self.options["function_opts"])
        self.cost_func = cost_func
        self.cnstr_func = cnstr_func
        self.lb_cnstr_func = lb_cnstr_func
        self.ub_cnstr_func = ub_cnstr_func

    def setup_initial_problem_solver(self):
        """Sets up the initial problem solver, for finding slack and virtual variables
        before the solver solver should run."""
        # Test if we don't need to do anything
        shortcut = self.skill_spec._has_virtual is None
        shortcut = shortcut and self.skill_spec.slack_var is None
        if shortcut:
            # If no slack, and no virtual, nothing to initialize
            self._has_initial = False
            return None

        # Prepare variables
        time_var = self.skill_spec.time_var
        robot_var = self.skill_spec.robot_var
        robot_vel_var = self.skill_spec.robot_vel_var
        virtual_var = self.skill_spec.virtual_var
        virtual_vel_var = self.skill_spec.virtual_vel_var
        input_var = self.skill_spec.input_var
        slack_var = self.skill_spec.slack_var
        nvirt = self.skill_spec.n_virtual_var
        nslack = self.skill_spec.n_slack_var

        # Optimization variable and parameters to opt.prob.
        list_par = [time_var, robot_var, robot_vel_var]
        list_names = ["time_var", "robot_var", "robot_vel_var"]
        opt_var = []
        if self.skill_spec._has_virtual:
            opt_var += [virtual_vel_var]
            list_par += [virtual_var]
            list_names += ["virtual_var"]
        if nvirt > 0:
            opt_var += [slack_var]
        if self.skill_spec._has_input:
            list_par += [input_var]
            list_names += ["input_var"]

        # Prepare cost expression
        cost_expr = self.get_regularised_cost_expr()

        # Prepare constraints expressions
        cnstr_expr_list = []
        lb_cnstr_expr_list = []
        ub_cnstr_expr_list = []
        slack_ind = 0
        virt_ind = 0
        for cnstr in self.skill_spec.constraints:
            cnstr_expr = 0.0
            found_virt = False
            found_slack = False
            expr_size = cnstr.expression.size()
            # Look for virtual variables in expr
            if nvirt > 0:
                J_virt = cnstr.jtimes(virtual_var, virtual_vel_var)
                if J_virt.nnz() > 0:
                    cnstr_expr += J_virt
                    found_virt = True
                    virt_ind += 1
            # Setup bounds/functions for numerics
            rob_der = cnstr.jtimes(robot_var, robot_vel_var)
            lb_cnstr_expr = -cnstr.jacobian(time_var) - rob_der
            ub_cnstr_expr = -cnstr.jacobian(time_var) - rob_der
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
            if nslack > 0:
                if cnstr.constraint_type == "soft":
                    cnstr_expr += -slack_var[slack_ind:slack_ind + expr_size[0]]
                    slack_ind += expr_size[0]
                    found_slack = True
            # Only care about this constraint if it's actually relevant
            if (found_virt or found_slack):
                cnstr_expr_list += [cnstr_expr]
                lb_cnstr_expr_list += [lb_cnstr_expr]
                ub_cnstr_expr_list += [ub_cnstr_expr]
        if slack_ind == 0 and virt_ind == 0:
            # Didn't find any of them.. return
            self._has_initial = False
            return None
        cnstr_expr_full = cs.vertcat(*cnstr_expr_list)
        lb_cnstr_expr_full = cs.vertcat(*lb_cnstr_expr_list)
        ub_cnstr_expr_full = cs.vertcat(*ub_cnstr_expr_list)
        lb_cnstr_func = cs.Function("lb_cnstr", list_par,
                                    [lb_cnstr_expr_full],
                                    list_names, ["lb_cnstr"],
                                    self.options["function_opts"])
        ub_cnstr_func = cs.Function("ub_cnstr", list_par,
                                    [ub_cnstr_expr_full],
                                    list_names, ["ub_cnstr"],
                                    self.options["function_opts"])
        self._initial_problem = {
            "nlp": {
                "x": cs.vertcat(*opt_var),
                "p": cs.vertcat(*list_par),
                "f": cost_expr,
                "g": cnstr_expr_full
            },
            "num": {
                "lb": lb_cnstr_func,
                "ub": ub_cnstr_func
            }
        }
        self.initial_solver = cs.nlpsol("solver",
                                        self.options["solver_name"],
                                        self._initial_problem["nlp"],
                                        self.options["solver_opts"])
        self._has_initial = True

    def solve_initial_problem(self,  time_var0, robot_var0,
                              virtual_var0=None, robot_vel_var0=None,
                              input_var0=None):
        """Solves the initial problem, finding slack and virtual variables."""
        # Test if we don't need to do anything
        nvirt = self.skill_spec.n_virtual_var
        if not self.skill_spec._has_virtual:
            nvirt = 0
        nslack = self.skill_spec.n_slack_var
        ninput = self.skill_spec.n_input_var
        if not self._has_initial:
            # if no slack, and no virtual, nothing to initialize
            return None, None
        if robot_vel_var0 is None:
            robot_vel_var0 = [0.0]*self.skill_spec.n_robot_var
        currvals = [time_var0, robot_var0, robot_vel_var0]
        if self.skill_spec._has_virtual:
            if virtual_var0 is None:
                virtual_var0 = [0.0]*nvirt
            currvals += [virtual_var0]
        if self.skill_spec._has_input:
            if input_var0 is None:
                input_var0 = [0.0]*ninput
            currvals += [input_var0]
        lb_num = self._initial_problem["num"]["lb"](*currvals)
        ub_num = self._initial_problem["num"]["ub"](*currvals)
        res = self.initial_solver(lbg=lb_num,
                                  ubg=ub_num,
                                  p=cs.vertcat(*currvals))
        res_virt = None
        res_slack = None
        if nvirt > 0:
            res_virt = res["x"][:nvirt]
        if nslack > 0:
            res_slack = res["x"][nvirt:nvirt+nslack]
        return res_virt, res_slack

    def solve(self, time_var, robot_var,
              virtual_var=None,
              input_var=None,
              warmstart_robot_vel_var=None,
              warmstart_virtual_vel_var=None,
              warmstart_slack_var=None):
        """Solve the skill specification."""
        # Useful sizes
        nrob = self.skill_spec.n_robot_var
        nvirt = self.skill_spec.n_virtual_var
        nslack = self.skill_spec.n_slack_var
        has_virtual = self.skill_spec._has_virtual
        has_input = self.skill_spec._has_input
        # Pack current values
        currvals = [time_var, robot_var]
        if virtual_var is not None and has_virtual:
            currvals += [virtual_var]
        if input_var is not None and has_input:
            currvals += [input_var]
        # Get numerics
        lb_num = self.lb_cnstr_func(*currvals)
        ub_num = self.ub_cnstr_func(*currvals)
        # Do we have warmstart?
        ws_rob = warmstart_robot_vel_var is not None
        ws_virt = warmstart_virtual_vel_var is not None and has_virtual
        ws_slack = warmstart_slack_var is not None and nslack > 0
        if not (ws_rob or ws_virt or ws_slack):
            # If no warmstart, then jsut calculate results
            self.res = self.solver(ubg=ub_num, lbg=lb_num, p=cs.vertcat(*currvals))
        else:
            # Pack warmstart vector
            warmstart = []
            use_warmstart = False
            if ws_rob:
                warmstart += [warmstart_robot_vel_var]
            else:
                warmstart += [cs.DM.zeros(nrob)]
            if ws_virt:
                warmstart += [warmstart_virtual_vel_var]
            else:
                if nvirt > 0:
                    warmstart += [cs.DM.zeros(nvirt)]
            if ws_slack:
                warmstart += [warmstart_slack_var]
            else:
                if nslack > 0:
                    warmstart += [cs.DM.zeros(nslack)]
            # Calculate results

            self.res = self.solver(x0=cs.vertcat(*warmstart),
                                   ubg=ub_num, lbg=lb_num,
                                   p=cs.vertcat(*currvals))
        res_robot_vel = self.res["x"][:nrob]
        if nvirt > 0 and has_virtual:
            res_virtual_vel = self.res["x"][nrob:nrob+nvirt]
        else:
            res_virtual_vel = None
        if nslack > 0:
            if not has_virtual:
                # Handles user error when user adds virtual error
                # but it's not actually in the expressions
                nvirt = 0
            res_slack = self.res["x"][nrob+nvirt: nrob+nvirt+nslack]
        else:
            res_slack = None
        return res_robot_vel, res_virtual_vel, res_slack

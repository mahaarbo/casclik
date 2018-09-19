"""Module for Reactive QP Controller, see class description.
"""

import casadi as cs
from casclik.constraints import EqualityConstraint, SetConstraint
from casclik.constraints import VelocityEqualityConstraint, VelocitySetConstraint
from casclik.controllers.base_controller import BaseController


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
    have a "weight_shifter", or regularization constant in the
    cost. It is to shift the weight between v and the slack
    variables. If you want a more complex cost, you can overload the
    H_func with ANY function that relies on the current values.

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
    weight_shifter = 0.001  # See eTaSL paper, corresponds to mu symbol

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
            weights = cs.vertcat([1.]*self.skill_spec.n_robot_var)
        elif isinstance(weights, cs.GenericCommonMatrix):
            if weights.size2 != 1:
                raise ValueError("robot_var_weights must be a vector.")
            elif weights.size1 != self.skill_spec.n_robot_var:
                raise ValueError("robot_var_weights and robot_var dimensions"
                                 + " do not match.")
        elif isinstance(weights, (list, cs.np.ndarray)):
            if len(weights) != self.skill_spec.n_robot_var:
                raise ValueError("robot_var_weights and robot_var dimensions"
                                 + " do not match")
            weights = cs.vertcat(weights)
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
            weights = cs.vertcat([1.]*self.skill_spec.n_virtual_var)
        elif isinstance(weights, cs.GenericCommonMatrix):
            if weights.size2 != 1:
                raise ValueError("virtual_var_weights must be a vector.")
            elif weights.size1 != self.skill_spec.n_virtual_var:
                raise ValueError("virtual_var_weights and virtual_var dimensions"
                                 + " do not match.")
        elif isinstance(weights, (list, cs.np.ndarray)):
            if len(weights) != self.skill_spec.n_virtual_var:
                raise ValueError("virtual_var_weights and virtual_var dimensions"
                                 + " do not match")
            weights = cs.vertcat(weights)
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
    def options(self):
        """Get or set the options. See ReactiveQPController.options_info for
        more information."""
        return self._options

    @options.setter
    def options(self, opt):
        if opt is None or not isinstance(opt, dict):
            opt = {}
        if "solver_name" not in opt:
            opt["solver_name"] = "qpoases"
        if "solver_opts" not in opt:
            opt["solver_opts"] = {}
        solver_opts = opt["solver_opts"]
        if "print_time" not in solver_opts:
            solver_opts["print_time"] = False
        if opt["solver_name"] == "qpoases":
            if "printLevel" not in solver_opts:
                solver_opts["printLevel"] = "none"
        elif opt["solver_name"] == "ooqp":
            if "print_level" not in solver_opts:
                solver_opts["print_level"] = 0
        if "jit" not in solver_opts:
            solver_opts["jit"] = True
        if "jit_options" not in solver_opts:
            solver_opts["jit_options"] = {"compiler": "shell",
                                          "flags": "-O2"}
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

    def get_cost_expr(self):
        """Returns a casadi expression describing the cost.

        Return:
             H for min_opt_var opt_var^T*H*opt_var"""
        nvirt = self.skill_spec.n_virtual_var
        nslack = self.skill_spec.n_slack_var
        mu = self.weight_shifter
        opt_weights = [mu*self.robot_var_weights]
        if nvirt > 0:
            opt_weights += [mu*self.virtual_var_weights]
        if nslack > 0:
            opt_weights += [mu + self.slack_var_weights]
        H = cs.diag(cs.vertcat(*opt_weights))
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
        self.solver = cs.conic("solver",
                               self.options["solver_name"],
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
        if virtual_var is not None and self.skill_spec._has_virtual:
            list_vars += [virtual_var]
            list_names += ["virtual_var"]
        input_var = self.skill_spec.input_var
        if input_var is not None and self.skill_spec._has_input:
            list_vars += [input_var]
            list_names += ["input_var"]
        H_func = cs.Function("H_func", list_vars, [H_expr],
                             list_names, ["H"],
                             self.options["function_opts"])
        A_func = cs.Function("A_func", list_vars, [A_expr],
                             list_names, ["A"],
                             self.options["function_opts"])
        Blb_func = cs.Function("Blb_expr", list_vars, [Blb_expr],
                               list_names, ["Blb"],
                               self.options["function_opts"])
        Bub_func = cs.Function("Bub_expr", list_vars, [Bub_expr],
                               list_names, ["Bub"],
                               self.options["function_opts"])
        self.H_func = H_func
        self.A_func = A_func
        self.Blb_func = Blb_func
        self.Bub_func = Bub_func

    def setup_initial_problem_solver(self):
        """Sets up the initial problem solver, for finding slack and virtual
        variables before the solver should run."""
        # Test if we don't need to do anything
        shortcut = self.skill_spec._has_virtual is None
        shortcut = shortcut and self.skill_spec.slack_var is None
        if shortcut:
            # If no slack, and no virtual, nothing to initializes
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
        mu = self.weight_shifter

        # Prepare cost expression
        opt_var = []
        opt_weights = []
        if nvirt > 0:
            opt_var += [virtual_vel_var]
            opt_weights += [mu*self.virtual_var_weights]
        if nslack > 0:
            opt_var += [slack_var]
            opt_weights += [(1+mu)*self.slack_var_weights]
        H_expr = cs.diag(cs.vertcat(*opt_weights))

        # Prepare constraints expressions
        cnstr_expr_list = []
        lb_cnstr_expr_list = []
        ub_cnstr_expr_list = []
        slack_ind = 0
        virt_ind = 0
        for cnstr in self.skill_spec.constraints:
            found_virt = False
            found_slack = False
            expr_size = cnstr.expression.size()
            # Look for virtual variables
            if nvirt > 0:
                J_virt = cs.jacobian(cnstr.expression, virtual_var)
                if J_virt.nnz() > 0:  # if it has non-zero elements
                    cnstr_expr = J_virt
                    found_virt = True
                    virt_ind += 1
                else:
                    cnstr_expr = cs.DM.zeros((expr_size[0], nvirt))
            # Setup bounds/functions for numerics
            rob_der = cnstr.jtimes(robot_var, robot_vel_var)
            lb_cnstr_expr = -cnstr.jacobian(time_var) - rob_der
            ub_cnstr_expr = -cnstr.jacobian(time_var) - rob_der
            if isinstance(cnstr, EqualityConstraint):
                lb_cnstr_expr += -cs.mtimes(cnstr.gain, cnstr.expression)
                ub_cnstr_expr += -cs.mtimes(cnstr.gain, cnstr.expression)
            elif isinstance(cnstr, SetConstraint):
                lb_cnstr_expr += cs.mtimes(cnstr.gain,
                                           cnstr.set_min - cnstr.expression)
                ub_cnstr_expr += cs.mtimes(cnstr.gain,
                                           cnstr.set_max - cnstr.expression)
            elif isinstance(cnstr.VelocityEqualityConstraint):
                lb_cnstr_expr += cnstr.target
                ub_cnstr_expr += cnstr.target
            elif isinstance(cnstr.VelocitySetConstraint):
                lb_cnstr_expr += cnstr.set_min
                ub_cnstr_expr += cnstr.set_max
            # Look for slack variables
            if nslack > 0:
                slack_mat = cs.DM.zeros((expr_size[0], nslack))
                if cnstr.constraint_type == "soft":
                    slack_mat[:, slack_ind:slack_ind + expr_size[0]] = -cs.DM.eye(expr_size[0])
                    slack_ind += expr_size[0]
                    found_slack = True
                if nvirt > 0:
                    cnstr_expr = cs.horzcat(cnstr_expr, slack_mat)
                else:
                    cnstr_expr = slack_mat
            # Only care about this expression if it's actually relevant
            if (found_virt or found_slack):
                cnstr_expr_list += [cnstr_expr]
                lb_cnstr_expr_list += [lb_cnstr_expr]
                ub_cnstr_expr_list += [ub_cnstr_expr]
        if slack_ind == 0 and virt_ind == 0:
            # Didn't find any of them.. return
            self._has_initial = False
            return None
        A_expr = cs.vertcat(*cnstr_expr_list)
        Blb_expr = cs.vertcat(*lb_cnstr_expr_list)
        Bub_expr = cs.vertcat(*ub_cnstr_expr_list)
        currval_vars = [time_var, robot_var, robot_vel_var]
        currval_names = ["time_var", "robot_var", "robot_vel_var"]
        if self.skill_spec._has_virtual:
            currval_vars += [virtual_var]
            currval_names += ["virtual_var"]
        if self.skill_spec._has_input:
            currval_vars += [input_var]
            currval_names += ["input_var"]
        func_opts = self.options["function_opts"]
        self._initial_problem = {"H": cs.Function("H_initial", currval_vars,
                                                  [H_expr], currval_names,
                                                  ["H"], func_opts),
                                 "A": cs.Function("A_initial", currval_vars,
                                                  [A_expr], currval_names,
                                                  ["A"], func_opts),
                                 "Blb": cs.Function("Blb_initial",
                                                    currval_vars,
                                                    [Blb_expr],
                                                    currval_names,
                                                    ["Blb"], func_opts),
                                 "Bub": cs.Function("Bub_initial",
                                                    currval_vars,
                                                    [Bub_expr],
                                                    currval_names,
                                                    ["Bub"], func_opts)}
        self.initial_solver = cs.conic("solver",
                                       self.options["solver_name"],
                                       {"h": H_expr.sparsity(),
                                        "a": A_expr.sparsity()},
                                       self.options["solver_opts"])
        self._has_initial = True

    def solve_initial_problem(self, time_var0, robot_var0,
                              virtual_var0=None, robot_vel_var0=None,
                              input_var0=None):
        """Solves the initial problem, finding slack and virtual variables."""
        # Test if we don't need to do anything
        nvirt = self.skill_spec.n_virtual_var
        nslack = self.skill_spec.n_slack_var
        ninput = self.skill_spec.n_input_var
        if not self._has_initial:
            # If no slack, and no virtual, nothing to initializes
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
        H = self._initial_problem["H"](*currvals)
        A = self._initial_problem["A"](*currvals)
        Blb = self._initial_problem["Blb"](*currvals)
        Bub = self._initial_problem["Bub"](*currvals)
        res = self.initial_solver(h=H, a=A, lba=Blb, uba=Bub)
        res_virt = None
        res_slack = None
        if nvirt > 0:
            res_virt = res["x"][:nvirt]
        if nslack > 0:
            res_slack = res["x"][nvirt:nvirt+nslack]
        return res_virt, res_slack

    def solve(self, time_var,
              robot_var,
              virtual_var=None,
              input_var=None,
              warmstart_robot_vel_var=None,
              warmstart_virtual_vel_var=None,
              warmstart_slack_var=None):
        """Solve the skill specification.
        """
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
        H = self.H_func(*currvals)
        A = self.A_func(*currvals)
        Blb = self.Blb_func(*currvals)
        Bub = self.Bub_func(*currvals)
        # Do we have warmstart?
        ws_rob = warmstart_robot_vel_var is not None
        ws_virt = warmstart_virtual_vel_var is not None and has_virtual
        ws_slack = warmstart_slack_var is not None and nslack > 0
        if not (ws_rob or ws_virt or ws_slack):
            # If no warmstart, then just calculate results
            self.res = self.solver(h=H, a=A, lba=Blb, uba=Bub)
        else:
            # Pack warmstart vector
            warmstart = []
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
                                   h=H, a=A, lba=Blb, uba=Bub)
        res_robot_vel = self.res["x"][:nrob]
        if nvirt > 0 and has_virtual:
            res_virtual_vel = self.res["x"][nrob: nrob+nvirt]
        else:
            res_virtual_vel = None
        if nslack > 0:
            if not has_virtual:
                # handles user error when user adds virtual_var
                # but it's not actually in the expressions
                nvirt = 0
            res_slack = self.res["x"][nrob+nvirt: nrob+nvirt+nslack]
        else:
            res_slack = None

        return res_robot_vel, res_virtual_vel, res_slack

"""Pseudo inverse controller, see class doc.
"""

import casadi as cs
from casclik.constraints import EqualityConstraint, SetConstraint
from casclik.constraints import VelocityEqualityConstraint
from casclik.controllers.base_controller import BaseController


class PseudoInverseController(BaseController):
    """Pseudo Inverse controller.

    The pseudo inverse controller is based on the Set-Based task
    controller of Signe Moe, and reminiscent of the stack-of-tasks
    controller. It uses the moore-penrose pseudo-inverse to calculate
    robot_var speeds, and the in_tangent_cone to handle set-based
    constraints. As it is a reactive controller resolved to the speeds
    of robot_var, it can handle inputs such as force sensors with a
    dampening effect.

    Args:
        skill_spec (SkillSpecification): skill specification
        options (dict): options dictionary, see self.options_info

    """
    controller_type = "PseudoInverseController"
    options_info = """TODO"""

    def __init__(self, skill_spec,
                 options=None):
        self.skill_spec = skill_spec
        self.options = options
    
    @property
    def options(self):
        """Get or set the options, See
        EqualityPseudoInverseController.options_info.

        """
        return self._options

    @options.setter
    def options(self, opt):
        if opt is None:
            opt = {}
        if "feedforward" not in opt:
            opt["feedforward"] = True
        if "multidim_sets" not in opt:
            opt["multidim_sets"] = False
        if "converge_final_set_to_max" not in opt:
            opt["converge_final_set_to_max"] = False
        if "pinv_method" not in opt:
            # Options: standard, damped
            opt["pinv_method"] = "damped"
        if "damping_factor" not in opt:
            opt["damping_factor"] = 1e-7
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

    @property
    def skill_spec(self):
        """Get or set the skill specification. This also sets the number of
         modes."""
        return self._skill_spec

    @skill_spec.setter
    def skill_spec(self, spec):
        cnstr_count = spec.count_constraints()
        self.n_set_constraints = cnstr_count["set"]
        self.n_modes = 2**cnstr_count["set"]
        state_var = [spec.robot_var]
        n_state_var = spec.n_robot_var
        cntrl_var = [spec.robot_vel_var]
        if spec.virtual_var is not None:
            state_var += [spec.virtual_var]
            n_state_var += spec.n_virtual_var
            cntrl_var += [spec.virtual_vel_var]
        self.state_var = cs.vertcat(*state_var)
        self.cntrl_var = cntrl_var
        self.n_state_var = n_state_var
        self._skill_spec = spec
        self.create_activation_map()

    def pinv(self, J):
        if self.options["pinv_method"] == "standard":
            pJ = cs.pinv(J)
        elif self.options["pinv_method"] == "damped":
            dmpng_fctr = self.options["damping_factor"]
            if J.size2() >= J.size1():
                inner = cs.mtimes(J, J.T)
                inner += dmpng_fctr*cs.DM.eye(J.size1())
                pJ = cs.solve(inner, J).T
            else:
                inner = cs.mtimes(J.T, J)
                inner += dmpng_fctr*cs.DM.eye(J.size2())
                pJ = cs.solve(inner, J.T)
        return pJ

    def create_activation_map(self):
        """Create the activation map.

        Activation of set constraints forms a binary decision tree. We
        exploit this to create an activation mapping by first creating a
        list of the binary representations of each number between 0 and
        2^{n_sets} - 1 reversed, and then sort that list according to
        the sum of ones in the binary representation.
        """
        # Create a list of lists of bits for binary numbers
        n_sets = self.n_set_constraints
        if n_sets == 0:
            self.activation_map = []
            return
        binmaps = []
        for mode_idx in range(2**n_sets):
            mode_bin_total = [0 for i in range(n_sets)]
            mode_idx_bin = [int(i) for i in bin(mode_idx)[2:]]
            for idx, val in enumerate(reversed(mode_idx_bin)):
                mode_bin_total[-(idx+1)] = val
            mode_bin_total = list(reversed(mode_bin_total))
            binmaps += [mode_bin_total]
        # Sort them according to how many are active
        self.activation_map = sorted(binmaps, key=lambda s: cs.np.sum(s))

    def get_in_tangent_cone_function(self, cnstr):
        """Returns a casadi function for the SetConstraint instance."""
        if not isinstance(cnstr, SetConstraint):
            raise TypeError("in_tangent_cone is only available for"
                            + " SetConstraint")
            return None
        time_var = self.skill_spec.time_var
        robot_var = self.skill_spec.robot_var
        list_vars = [time_var, robot_var]
        list_names = ["time_var", "robot_var"]
        robot_vel_var = self.skill_spec.robot_vel_var
        opt_var = [robot_vel_var]
        opt_var_names = ["robot_vel_var"]
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
            opt_var_names += ["virtual_vel_var"]
            dexpr += cs.jtimes(expr, virtual_var, virtual_vel_var)
        if input_var is not None:
            list_vars += [input_var]
            list_names += ["input_var"]
        if_low_inc = cs.if_else(
            dexpr > 0,
            True,
            False,
            True
        )
        if_high_dec = cs.if_else(
            dexpr < 0,
            True,
            False,
            True
        )
        leq_high = cs.if_else(
            expr <= set_max,
            True,
            if_high_dec,
            True
        )
        in_tc = cs.if_else(
            set_min <= expr,
            leq_high,
            if_low_inc,
            True
        )
        return cs.Function("in_tc_"+cnstr.label.replace(" ", "_"),
                           list_vars+opt_var,
                           [in_tc],
                           list_names+opt_var_names,
                           ["in_tc_"+cnstr.label])

    def get_in_tangent_cone_function_multidim(self, cnstr):
        """Returns a casadi function for the SetConstraint instance when the
        SetConstraint is multidimensional."""
        if not isinstance(cnstr, SetConstraint):
            raise TypeError("in_tangent_cone is only available for"
                            + " SetConstraint")
        time_var = self.skill_spec.time_var
        robot_var = self.skill_spec.robot_var
        list_vars = [time_var, robot_var]
        list_names = ["time_var", "robot_var"]
        robot_vel_var = self.skill_spec.robot_vel_var
        opt_var = [robot_vel_var]
        opt_var_names = ["robot_vel_var"]
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
            opt_var_names += ["virtual_vel_var"]
            dexpr += cs.jtimes(expr, virtual_var, virtual_vel_var)
        if input_var is not None:
            list_vars += [input_var]
            list_vars += ["input_var"]
        le = expr - set_min
        ue = expr - set_max
        le_good = le >= 1e-12
        ue_good = ue <= 1e-12
        above = cs.dot(le_good - 1, le_good - 1) == 0
        below = cs.dot(ue_good - 1, ue_good - 1) == 0
        inside = cs.logic_and(above, below)
        out_dir = (cs.sign(le) + cs.sign(ue))/2.0
        # going_in = cs.dot(out_dir, dexpr) <= 0.0
        same_signs = cs.sign(le) == cs.sign(ue)
        corner = cs.dot(same_signs - 1, same_signs - 1) == 0
        dists = (cs.norm_2(dexpr)+1e-10)*cs.norm_2(out_dir)
        corner_handler = cs.if_else(
            cs.dot(out_dir, dexpr) < 0.0,
            cs.fabs(cs.dot(-out_dir, dexpr))/dists < cs.np.cos(cs.np.pi/4),
            False,
            True
        )
        going_in = cs.if_else(
            corner,
            corner_handler,
            cs.dot(out_dir, dexpr) < 0.0,
            True
        )

        in_tc = cs.if_else(
            inside,  # Are we inside?
            True,  # Then true.
            going_in,  # If not, check if we're "going_in"
            True
        )
        return cs.Function("in_tc_"+cnstr.label.replace(" ", "_"),
                           list_vars+opt_var,
                           [in_tc],
                           list_names+opt_var_names,
                           ["in_tc_"+cnstr.label])

    def get_problem_expressions(self):
        """Initialize the desired control variables."""
        time_var = self.skill_spec.time_var
        state_var = self.state_var
        n_state_var = self.n_state_var
        n_modes = self.n_modes
        modes = [{} for i in range(n_modes)]
        # Figure out how to do modes here
        for mode_idx, mode in enumerate(modes):
            set_idx = 0
            cntrl_var_expr = cs.MX.zeros(n_state_var)
            J_active_list = []
            rJ_active_list = []
            in_tc_list = []
            active_set_names = []
            for cnstr_idx, cnstr in enumerate(self.skill_spec.constraints):
                # Identifiers
                is_first = len(J_active_list) == 0
                is_last = cnstr == self.skill_spec.constraints[-1]
                is_set = isinstance(cnstr, SetConstraint)
                is_eq = isinstance(cnstr, EqualityConstraint)
                is_veleq = isinstance(cnstr, VelocityEqualityConstraint)
                conv_last = self.options["converge_final_set_to_max"]
                use_multidim = self.options["multidim_sets"]

                # General jacobians
                Jt = cnstr.jacobian(time_var)
                Ji = cnstr.jacobian(state_var)

                # Activation matrix for multidim set constraints
                if use_multidim and is_set:
                    # The jacobian is only active in set active
                    # directions
                    above = cnstr.expression - cnstr.set_max > 0.0
                    below = cnstr.expression - cnstr.set_min < 0.0
                    active = cs.logic_or(
                        above,
                        below
                        )
                    S = cs.diag(active)
                if not use_multidim and is_set:
                    expr_dim = cnstr.expression.size()[0]
                    if expr_dim > 1:
                        raise NotImplementedError("PseudoInverseController"
                                                  + " does not yet have gu"
                                                  + "aranteed stable suppo"
                                                  + "rt for multidimension"
                                                  + "al SetConstraints. Si"
                                                  + "ze("+cnstr.label+")="
                                                  + str(expr_dim) + ". Set"
                                                  + " the multidim_sets fi"
                                                  + "eld in options to Tru"
                                                  + "e for experimental su"
                                                  + "pport.")
                ########################################
                # Case-by-case for the constraint types:
                ########################################
                # First has no null-space effect
                if is_first and is_eq:
                    cnstr_des = -cs.mtimes(cnstr.gain,
                                           cnstr.expression)
                    if self.options["feedforward"]:
                        cnstr_des += -Jt
                    cntrl_var_expr += cs.mtimes(
                        self.pinv(Ji), cnstr_des
                    )
                    J_active_list += [Ji]
                    rJ_active_list += [Ji]

                # Allow convergence of last set
                elif is_set and is_last and conv_last:
                    if self.activation_map[mode_idx][set_idx]:
                        cnstr_des = cs.mtimes(cnstr.gain,
                                              cnstr.set_max - cnstr.expression)
                        if self.options["feedforward"]:
                            cnstr_des += -Jt
                        J0toi = cs.vertcat(*J_active_list)
                        rJ0toi = cs.vertcat(*rJ_active_list)
                        N0toi = cs.MX.eye(n_state_var) - cs.mtimes(
                            self.pinv(J0toi),
                            rJ0toi
                        )
                        NJ = cs.mtimes(N0toi, self.pinv(Ji))
                        cntrl_var_expr += cs.mtimes(NJ, cnstr_des)
                        J_active_list += [Ji]
                        if use_multidim:
                            rJ_active_list += [cs.mtimes(S, Ji)]
                        else:
                            rJ_active_list += [Ji]
                        active_set_names += [cnstr.label]
                    else:
                        expr_dim = cnstr.expression.size()[0]
                        if expr_dim == 1:
                            in_TC = self.get_in_tangent_cone_function(cnstr)
                            in_tc_list += [in_TC]
                        elif use_multidim:
                            in_TC = self.get_in_tangent_cone_function_multidim(
                                cnstr
                            )
                            in_tc_list += [in_TC]
                        else:
                            raise NotImplementedError("PseudoInverseController"
                                                      + " does not yet have gu"
                                                      + "aranteed stable suppo"
                                                      + "rt for multidimension"
                                                      + "al SetConstraints. Si"
                                                      + "ze("+cnstr.label+")="
                                                      + str(expr_dim) + ". Set"
                                                      + " the multidim_sets fi"
                                                      + "eld in options to Tru"
                                                      + "e for experimental su"
                                                      + "pport.")
                    set_idx += 1

                # Others
                elif is_eq:
                    cnstr_des = -cs.mtimes(cnstr.gain,
                                           cnstr.expression)
                    if self.options["feedforward"]:
                        cnstr_des += -Jt
                    J0toi = cs.vertcat(*J_active_list)
                    rJ0toi = cs.vertcat(*rJ_active_list)
                    N0toi = cs.MX.eye(n_state_var) - cs.mtimes(
                        self.pinv(J0toi),
                        rJ0toi
                    )
                    NJ = cs.mtimes(N0toi, self.pinv(Ji))
                    cntrl_var_expr += cs.mtimes(NJ, cnstr_des)
                    J_active_list += [Ji]
                    rJ_active_list += [Ji]

                elif is_set:
                    if self.activation_map[mode_idx][set_idx]:
                        J_active_list += [Ji]
                        if use_multidim:
                            rJ_active_list += [cs.mtimes(S, Ji)]
                        else:
                            rJ_active_list += [Ji]
                        active_set_names += [cnstr.label]
                    else:
                        expr_dim = cnstr.expression.size()[0]
                        if expr_dim == 1:
                            in_TC = self.get_in_tangent_cone_function(cnstr)
                            in_tc_list += [in_TC]
                        elif use_multidim:
                            in_TC = self.get_in_tangent_cone_function_multidim(
                                cnstr
                            )
                            in_tc_list += [in_TC]
                        else:
                            raise NotImplementedError("PseudoInverseController"
                                                      + " does not yet have gu"
                                                      + "aranteed stable suppo"
                                                      + "rt for multidimension"
                                                      + "al SetConstraints. Si"
                                                      + "ze("+cnstr.label+")="
                                                      + str(expr_dim) + ". Set"
                                                      + " the multidim_sets fi"
                                                      + "eld in options to Tru"
                                                      + "e for experimental su"
                                                      + "pport.")
                    set_idx += 1

                elif is_veleq:
                    cnstr_des = cnstr.target
                    if self.options["feedforward"]:
                        cnstr_des += -Jt
                    J0toi = cs.vertcat(*J_active_list)
                    rJ0toi = cs.vertcat(*rJ_active_list)
                    N0toi = cs.MX.eye(n_state_var) - cs.mtimes(
                        self.pinv(J0toi),
                        rJ0toi
                    )
                    NJ = cs.mtimes(N0toi, self.pinv(Ji))
                    cntrl_var_expr += cs.mtimes(NJ, cnstr_des)
                    J_active_list += [Ji]
                    rJ_active_list += [Ji]

            # End of constraints for loop
            mode["cntrl_var_expr"] = cntrl_var_expr
            mode["in_tangent_cone_func_list"] = in_tc_list
            mode["active_set_names"] = active_set_names
        # End of modes loop
        self.modes = modes
        return modes

    def setup_problem_functions(self):
        """Setup problem functions. Separate from get_problem_expressions for
        future compilation of functions for speed-up and to generate
        all the modes.
        """
        self.get_problem_expressions()
        func_opts = self.options["function_opts"]
        # All variables
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

        for mode_idx, mode in enumerate(self.modes):
            cntrl_var_expr = mode["cntrl_var_expr"]
            mode["cntrl_var_func"] = cs.Function(
                "cntrl_var_"+str(mode_idx),
                list_vars,
                [cntrl_var_expr],
                list_names,
                ["cntrl_var"],
                func_opts
            )

    def setup_initial_problem_solver(self):
        """Setup the initial problem solver. This does not do anything yet.
        """
        pass

    def solve_initial_problem(self, time_var0, robot_var0,
                              virtual_var0=None, robot_vel_var0=None,
                              input_var0=None):
        """Solve the initial problem. This does not do anything yet."""
        if self.skill_spec.slack_var is not None:
            res_slack = cs.DM.zeros(self.skill_spec.slack_var.size())
        if virtual_var0 is not None:
            res_virt = cs.DM.zeros(self.skill_spec.virtual_var.size())
        else:
            res_virt = None
        if self.skill_spec.slack_var is not None:
            res_slack = cs.DM.zeros(self.skill_spec.slack_var.size())
        else:
            res_slack = None
        return res_virt, res_slack

    def setup_solver(self):
        """Sets up the solver. (just runs the get_problem_expressions
        and setup_problem_functions)"""
        self.get_problem_expressions()
        self.setup_problem_functions()

    def solve(self, time_var,
              robot_var,
              virtual_var=None,
              input_var=None,
              warmstart_robot_vel_var=None,
              warmstart_virtual_vel_var=None,
              warmstart_slack_var=None):
        currvals = [time_var, robot_var]
        nrob = self.skill_spec.n_robot_var
        if virtual_var is not None and self.skill_spec._has_virtual:
            nvirt = self.skill_spec.n_virtual_var
            currvals += [virtual_var]
        else:
            nvirt = 0
        if input_var is not None and self.skill_spec._has_input:
            currvals += [input_var]
        # Check all modes
        NONEOKAY = True
        for mode_idx, mode in enumerate(self.modes):
            cntrl_var = mode["cntrl_var_func"](*currvals)
            cntrl_rob = cntrl_var[:nrob]
            suggested = currvals + [cntrl_rob]
            if nvirt > 0:
                cntrl_virt = cntrl_var[nrob:]
                suggested += [cntrl_virt]
            else:
                cntrl_virt = None
            ALLOKAY = True
            intcstr = ""
            for in_TC_func in mode["in_tangent_cone_func_list"]:
                in_TC = in_TC_func(*suggested)
                intcstr += str(int(in_TC))
                if not in_TC:
                    ALLOKAY = False
                    break
            if ALLOKAY:
                NONEOKAY = False
                self.current_mode = mode_idx
                break
        if NONEOKAY:
            self.current_mode = -1
            cntrl_rob = cs.DM.zeros(nrob)
            if nvirt > 0:
                cntrl_virt = cs.DM.zeros(nvirt)
        return cntrl_rob, cntrl_virt, None

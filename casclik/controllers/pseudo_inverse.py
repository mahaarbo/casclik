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
            list_vars += ["input_var"]
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
        le_good = le >= 0.0
        ue_good = ue <= 0.0
        above = cs.dot(le_good - 1, le_good - 1) == 0
        below = cs.dot(ue_good - 1, ue_good - 1) == 0
        inside = cs.logic_and(above, below)
        out_dir = (cs.sign(le) + cs.sign(ue))/2.0
        going_in = cs.dot(out_dir, dexpr) < 0.0
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
        """Initializes the constraint derivatives, Jacobians and null-space
        functions."""
        time_var = self.skill_spec.time_var
        state_var = self.state_var
        n_state_var = self.n_state_var
        n_modes = self.n_modes
        n_sets = self.n_set_constraints
        # Prepare all the modes
        modes = [{"J_expr_list": [],
                  "Jt_expr_list": [],
                  "J0i_expr_list": [],
                  "N0i_expr_list": [],
                  "des_dconstr_expr_list": [],
                  "in_tangent_cone_func_list": []} for i in range(n_modes)]
        mode_count = 0
        for cnstr in self.skill_spec.constraints:
            # We need this for everyone
            Ji = cnstr.jacobian(state_var)
            Jti = cnstr.jacobian(time_var)
            if isinstance(cnstr, EqualityConstraint):
                # For equality constraints, we always do the same
                for mode in modes:
                    mode["J_expr_list"] += [Ji]
                    mode["Jt_expr_list"] += [Jti]
                    J0i = cs.vertcat(*mode["J_expr_list"])
                    mode["J0i_expr_list"] += [J0i]
                    N0i = cs.MX.eye(n_state_var) - cs.mtimes(cs.pinv(J0i), J0i)
                    mode["N0i_expr_list"] += [N0i]
                    des_dconstri = -cs.mtimes(cnstr.gain, cnstr.expression)
                    if self.options["feedforward"]:
                        des_dconstri += -Jti
                    mode["des_dconstr_expr_list"] += [des_dconstri]

            elif isinstance(cnstr, SetConstraint):
                # For set constraints, things are a little more
                # complicated
                # The modes are like a binary tree, so we have to
                # split the list of modes to make that fit, and only affect
                # correct ones Find indices of affected modes:
                amode_idxs = []
                mode_count += 1
                section_size = 2**(n_sets - mode_count)
                sect_ind = 0
                for i in range(2**mode_count):
                    # Every even section stays the same
                    # Every odd section is affected by null-space
                    if i % 2 == 1:
                        amode_idxs += range(sect_ind, sect_ind+section_size)
                    sect_ind += section_size
                # The affected modes are the modes where the constraint is
                # active. Then loop over the modes.
                # If a mode is affected, we modify the expressions
                # If it isn't affected, we give it the in_tc_func
                for mode_idx, mode in enumerate(modes):
                    if mode_idx in amode_idxs:
                        mode["J_expr_list"] += [Ji]
                        mode["Jt_expr_list"] += [Jti]
                        J0i = cs.vertcat(*mode["J_expr_list"])
                        N0i = cs.MX.eye(n_state_var) - cs.mtimes(cs.pinv(J0i), J0i)
                        mode["N0i_expr_list"] += [N0i]
                        # If the last constraint is a set constraint,
                        # activation means that we will desire to converge to
                        # it rather than activating the null-space
                        if cnstr == self.skill_spec.constraints[-1]:
                            des_dconstri = -cs.mtimes(cnstr.gain,
                                                      cnstr.expression)
                        else:
                            des_dconstri = cs.MX.zeros(cnstr.expression.size()[0])
                        mode["des_dconstr_expr_list"] += [des_dconstri]
                    else:
                        if cnstr.expression.size()[0] == 1:
                            in_tc_func = self.get_in_tangent_cone_function(cnstr)
                        elif self.options["multidim_sets"]:
                            in_tc_func = self.get_in_tangent_cone_function_multidim(cnstr)
                        else:
                            raise NotImplementedError("PseudoInverseController"
                                                      + " does not yet have gu"
                                                      + "aranteed stable suppo"
                                                      + "rt for multidimension"
                                                      + "al SetConstraints. Si"
                                                      + "ze("+cnstr.label+")="
                                                      + str(cnstr.expression.size())
                                                      + ". Set the multidim_se"
                                                      + "ts field in options t"
                                                      + "o True for experiment"
                                                      + "al support.")
                        mode["in_tangent_cone_func_list"] += [in_tc_func]
            elif isinstance(cnstr, VelocityEqualityConstraint):
                # Velocity Equality constraints: des_dconstri is just set_min
                for mode in modes:
                    mode["J_expr_list"] += [Ji]
                    mode["Jt_expr_list"] += [Jti]
                    J0i = cs.vertcat(*mode["J_expr_list"])
                    mode["J0i_expr_list"] += [J0i]
                    N0i = cs.MX.eye(n_state_var) - cs.mtimes(cs.pinv(J0i), J0i)
                    mode["N0i_expr_list"] += [N0i]
                    des_dconstri = cnstr.target
                    if self.options["feedforward"]:
                        des_dconstri += -Jti
                    mode["des_dconstr_expr_list"] += [des_dconstri]
            else:
                raise NotImplementedError("PseudoInverseController only knows"
                                          + " of EqualityConstraint, "
                                          + "SetConstraint, and Velocity"
                                          + "EqualityConstraint. The "
                                          + "constraint named "
                                          + str(cnstr.label) + " of type: "
                                          + str(type(cnstr)) + ".")
        return modes

    def setup_problem_functions(self):
        """Setup problem functions. Separate from get_problem_expressions for
        future compilation of functions for speed-up and to generate
        all the modes.
        """
        #raise NotImplementedError("Setup problem functions is not done yet.")
        pass
    
    def setup_solver(self):
        func_opts = self.options["function_opts"]
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
        # Get all the modes and their expressions
        modes = self.get_problem_expressions()
        n_state_var = self.n_state_var
        mode_ind = 0
        full_cntrl_var_des_expr = []
        for mode in modes:
            cntrl_var_des_expr = cs.MX.zeros(n_state_var)
            for i in range(len(mode["J_expr_list"])):
                Ji = mode["J_expr_list"][i]
                dconstri = mode["des_dconstr_expr_list"][i]
                if i == 0:
                    cntrl_var_des_expr += cs.mtimes(cs.pinv(Ji), dconstri)
                else:
                    N0i = mode["N0i_expr_list"][i-1]
                    cntrl_var_des_expr += cs.mtimes(N0i,
                                                    cs.mtimes(cs.pinv(Ji),
                                                              dconstri))
            mode["fcntrl_var_des"] = cs.Function("fdes"+str(mode_ind),
                                                 list_vars,
                                                 [cntrl_var_des_expr],
                                                 list_names,
                                                 ["cntrl_var_des"],
                                                 func_opts)
            full_cntrl_var_des_expr += [cntrl_var_des_expr]
        full_cntrl_var_des_expr = cs.vertcat(*full_cntrl_var_des_expr)
        full_cntrl_var_des_func = cs.Function("fdes",
                                              list_vars,
                                              [full_cntrl_var_des_expr],
                                              list_names,
                                              ["cntrl_var_des"],
                                              func_opts)
        self.modes = modes
        self.full_cntrl_var_des_func = full_cntrl_var_des_func

    def solve(self, time_var,
              robot_var,
              virtual_var=None,
              input_var=None):
        currvals = [time_var, robot_var]
        nrob = self.skill_spec.n_robot_var
        nvirt = self.skill_spec.n_virtual_var
        if virtual_var is not None and nvirt > 0:
            currvals += [virtual_var]
        if input_var is not None:
            currvals += [input_var]
        # Check all modes
        for i, mode in enumerate(self.modes):
            # The first one with all okay is the one we return
            ALLOKAY = True
            dcntrl_var = mode["fcntrl_var_des"](*currvals)
            dcntrl_rob = dcntrl_var[:nrob]
            suggested = currvals + [dcntrl_rob]
            if nvirt > 0:
                dcntrl_virt = dcntrl_var[nrob:]
                suggested += [dcntrl_virt]
            for in_tangent_cone in mode["in_tangent_cone_func_list"]:
                if not in_tangent_cone(*(suggested)):
                    ALLOKAY = False
                    break
            if ALLOKAY:
                self.current_mode = i
                break
        # Split robot_vel_var and virtual_vel_var
        res_robot_vel = dcntrl_var[:nrob]
        if nvirt > 0:
            res_virtual_vel = dcntrl_var[nrob:nrob+nvirt]
        else:
            res_virtual_vel = None
        # Same return pattern as everything else
        # But there are no slack variables (as of yet)
        return res_robot_vel, res_virtual_vel, None

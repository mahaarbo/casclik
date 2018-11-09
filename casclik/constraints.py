"""Constraints and tools for the constraint objects.
Todo:
    * Add list functionality to set.
    * Add a good list check to _check_sizes
    * Add @property to the gain, set_min, & set_max that runs _check_sizes
    * Add a "combine" method to vertcat constraints.
    * Allow SetConstraints to have set_min & set_max that are expressions
"""
import casadi as cs


class BaseConstraint(object):
    """Base constraint object
    Args:
        label (str): Name of the constraint
        expression (cs.MX): expression of the constraint
        gain (float,MX,DM,numpy.ndarray): gain in the constraint
    """
    constraint_class = "BaseConstraint"

    def __init__(self, label, expression, gain):
        self.label = label
        self.expression = expression
        self.gain = gain

    def __repr__(self):
        return self.label+"<"+self.constraint_class+" at 0x"+str(id(self))+">"

    def size(self):
        return self.expression.size()

    def _check_sizes(self):
        expr_size = self.size()
        if expr_size[1] != 1:
            return False
        if isinstance(self.gain, cs.MX):
            gs = self.gain.size()
            if gs[0] == gs[1]:
                if gs[1] == expr_size[0] or gs[1] == 1:
                    return True
        elif isinstance(self.gain, cs.DM):
            gs = self.gain.size()
            if gs[0] == gs[1]:
                if gs[1] == expr_size[0] or gs[1] == 1:
                    return True
        elif isinstance(self.gain, cs.np.ndarray):
            gs = self.gain.shape
            if gs[0] == gs[1] and gs[1] == expr_size[0]:
                return True
        elif isinstance(self.gain, float):
            return True
        elif isinstance(self.gain, list):
            for val in self.gain:
                if not isinstance(val, float):
                    if not isinstance(val, int):
                        raise TypeError("Unknown gain type in " + self.label
                                        + ". Supported are: float, MX, DM, "
                                        + "numpy.ndarray, and list of floats"
                                        + "/ints")
            if len(self.gain) == expr_size[0]:
                return True
        else:
            raise TypeError("Unknown gain type in " + self.label + "."
                            + "Supported are: float, MX, DM, numpy."
                            + ".ndarray, and list of floats/ints.")
        return False

    def jacobian(self, var):
        """Returns the partial derivative of the expression with respect to
        var.
        Return:
            cs.MX: expression of partial derivative
        """
        return cs.jacobian(self.expression, var)

    def jtimes(self, varA, varB):
        """Returns the partial derivative of the expression with respect to
        varA, times varB.
        Return:
            cs.MX: expression of partial derivative"""
        return cs.jtimes(self.expression, varA, varB)

    def nullspace(self, var):
        """Returns I-pinv(J)*J where J is dexpression/dvar."""
        J = self.jacobian(var)
        return cs.MX.eye(var.size()[0]) - cs.mtimes(cs.pinv(J), J)


class EqualityConstraint(BaseConstraint):
    """Equality constraints can be hard or soft, they can have different
    priorities but they all try to converge to zero. Gain can be an
    expression but must not contain robot_vel_var, or virtual_vel_var.

    Equality constraints in the optimization solver is of the form:

    dexpr/drob*rob_vel + dexpr/dvirt*virt_vel = -gain*expr - dexpr/dt

    where we have the partial derivatives of the expression w.r.t. the
    robot_var and virtual_var. Using gain times expression gives us
    exponential stability if gain is a constant >0, and the partial
    derivative of the constraint expression w.r.t time_var is used as
    a feedforward.

    Args:
        label (str): Name of the constraint
        expression (cs.MX): expression of the constraint
        gain (float,cs.MX,cs.DM,cs.np.ndarray): gain in the constraint
        constraint_type (str): "hard" or "soft", for opt.prob. controllers
        priority (int): sorting key of constraint, for pseudoinv. controllers
        slack_weight (float): Optional specification of slack_var weight
    """
    constraint_class = "EqualityConstraint"

    def __init__(self, label,
                 expression,
                 gain=1.0,
                 constraint_type="hard",
                 priority=1,
                 slack_weight=1.0):
        BaseConstraint.__init__(self, label, expression, gain)
        self.constraint_type = constraint_type
        self.priority = priority
        self.slack_weight = slack_weight
        if not self._check_sizes():
            raise ValueError("Gain and expression dimensions do not match.")

    def __add__(self, cnstrB):
        """Concatenate two constraints of equal priority and type."""
        if not self.priority == cnstrB.priority:
            raise TypeError("Added constraints must have same priority.")
        if not self.constraint_type == cnstrB.constraint_type:
            raise TypeError("Added constrains must have same constraint type")
        A_size = self.expression.size()[0]
        B_size = self.expression.size()[0]
        exprA = self.expression
        exprB = cnstrB.expression
        expr = cs.vertcat(exprA, exprB)
        gain = cs.MX.zeros(A_size + B_size,
                           A_size + B_size)
        gain[:A_size, :A_size] = self.gain
        gain[:-B_size, :-B_size] = cnstrB.gain
        return EqualityConstraint(self.label+"+"+cnstrB.label,
                                  expression=expr,
                                  gain=gain,
                                  constraint_type=self.constraint_type,
                                  priority=self.priority)


class SetConstraint(BaseConstraint):
    """Set constraints can be hard or soft, they can have different
    priorities but they all try to converge to a set of values. The
    gain, set_min, and set_max must not contain the robot_vel_var or
    virtual_vel_var.

    Set constraints in the optimization problems are handled as:

    gain*(set_min - expr) <= dexpr/dt <= gain*(set_max - expr)

    where we take the total derivative of expression w.r.t. time to
    get the constraint in terms the optimization variables. See
    EqualityConstraint.

    In the pseudoinverse controllers it is handled differently
    depending on the priority. Generally they have an in_tangent_cone
    function that works as follows:

    if  set_min < expression < set_max:
        return True
    elif expression <= set_min and dexpression/dt > 0:
        return True
    elif expression >= set_max and dexpression/dt < 0:
        return True
    else:
        return False

    Args:
        label (str): Name of the constraint
        expression (cs.MX): expression of the constraint
        gain (float,cs.MX,cs.DM,cs.np.ndarray): gain in the constraint
        set_min (list, cs.MX, cs.DM, cs.np.ndarray): minimum value of set
        set_max (list, cs.MX, cs.DM, cs.np.ndarray): maximum value of set
        constraint_type (str): "hard" or "soft", for opt.prob. controllers
        priority (int): sorting key of constraint, for pseudoinv. controllers
        slack_weight (float): Optional slack_var weight
    """
    constraint_class = "SetConstraint"

    def __init__(self, label,
                 expression,
                 gain=1.0,
                 set_min=None,
                 set_max=None,
                 constraint_type="hard",
                 priority=1,
                 slack_weight=1.0):
        BaseConstraint.__init__(self, label,
                                expression, gain)
        self.constraint_type = constraint_type
        self.priority = priority
        if set_min is None:
            self.set_min = -1e10*cs.np.ones(expression.size()[0])
        else:
            self.set_min = set_min
        if set_max is None:
            self.set_max = 1e10*cs.np.ones(expression.size()[0])
        else:
            self.set_max = set_max
        self.slack_weight = slack_weight
        if not self._check_sizes():
            raise ValueError("Gain, set limits, or expression dimensions"
                             + " do not match in " + self.label)

    def _check_sizes(self):
        """Checks sizes of the expression, set, and gain."""
        # Todo: add support for lists as well
        expr_size = self.size()
        rmin = False
        rmax = False
        rgain = BaseConstraint._check_sizes(self)
        if isinstance(self.set_min, float):
            if expr_size[0] == 1:
                rmin = True
        elif isinstance(self.set_min, cs.MX):
            if self.set_min.is_symbolic():
                rmin = False
            else:
                szi = self.set_min.size()
                if szi[0] == expr_size[0] and szi[1] == 1:
                    rmin = True
        elif isinstance(self.set_min, cs.DM):
            szi = self.set_min.size()
            if szi[0] == expr_size[0] and szi[1] == 1:
                rmin = True
        elif isinstance(self.set_min, cs.np.ndarray):
            szi = self.set_min.shape
            if len(szi) == 1 and szi[0] == expr_size[0]:
                rmin = True
            elif szi[0] == expr_size[0] and szi[1] == 1:
                rmin = True
        else:
            raise TypeError("Unknown set_min type in " + self.label
                            + ". Supported are float, MX, DM, and "
                            + "numpy.ndarray")

        if isinstance(self.set_max, float):
            if expr_size[0] == 1:
                rmax = True
        elif isinstance(self.set_max, cs.MX):
            if self.set_min.is_symbolic():
                rmax = False
            else:
                sza = self.set_max.size()
                if sza[0] == expr_size[0] and sza[1] == 1:
                    rmax = True
        elif isinstance(self.set_max, cs.DM):
            sza = self.set_max.size()
            if sza[0] == expr_size[0] and sza[1] == 1:
                rmax = True
        elif isinstance(self.set_max, cs.np.ndarray):
            sza = self.set_max.shape
            if len(sza) == 1 and sza[0] == expr_size[0]:
                rmax = True
            elif sza[0] == expr_size[0] and szi == 1:
                rmax = True

        else:
            raise TypeError("Unknown set_max type in " + self.label
                            + ". supported are float, MX, DM, and "
                            + "numpy.ndarray")
        return rgain and rmin and rmax

    def __add__(self, cnstrB):
        """Concatenate two constraints of equal priority and type."""
        if not self.priority == cnstrB.priority:
            raise TypeError("Added constraints must have same priority.")
        if not self.constraint_type == cnstrB.constraint_type:
            raise TypeError("Added constrains must have same constraint type")
        A_size = self.expression.size()[0]
        B_size = self.expression.size()[0]
        exprA = self.expression
        exprB = cnstrB.expression
        expr = cs.vertcat(exprA, exprB)
        gain = cs.MX.zeros(A_size + B_size,
                           A_size + B_size)
        gain[:A_size, :A_size] = self.gain
        gain[:-B_size, :-B_size] = cnstrB.gain
        set_min = cs.vertcat(self.set_min,
                             cnstrB.set_min)
        set_max = cs.vertcat(self.set_max,
                             cnstrB.set_max)
        return SetConstraint(self.label+"+"+cnstrB.label,
                             expression=expr,
                             gain=gain,
                             set_min=set_min,
                             set_max=set_max,
                             constraint_type=self.constraint_type,
                             priority=self.priority)


class VelocityEqualityConstraint(BaseConstraint):
    """VelocityEqualityConstraints are made to set a constant speed. They
    can be hard or soft, have different priorities but they set a certain
    expression derivative up to be an equality constraint.
    For joint speed resolved controllers, the gain serves no purpose.

    In the optimization controllers this is handled as:
    dexpr/dq*q_der + dexpr/dx*x_der = target - dexpr/dt

    In the pseudoinverse controllers, the :
    [q_der', x_der']' = pinv([dexpr/dq, dexpr/dx)*(target - dexpr/dt)

    WARNING: This doesn't run _check_sizes.
    Args:
        label (str): Name of the constraint
        expression (cs.MX): expression of the constraint
        gain (float,cs.MX,cs.DM,cs.np.ndarray): gain in the constraint
        constraint_type (str): "hard" or "soft", for opt.prob.controllers
        priority (int): sorting key of constraints, for pseudoinv.controllers
        target (float,cs.MX,cs.DM,cs.np.ndarray): target value of velocity
        slack_weight (float): Optional slack_var weight
    """

    def __init__(self, label,
                 expression,
                 gain=1.0,
                 constraint_type="hard",
                 priority=1,
                 target=0.0,
                 slack_weight=1.0):
        BaseConstraint.__init__(self, label, expression, gain)
        self.constraint_type = constraint_type
        self.priority = priority
        self.target = target
        self.slack_weight = slack_weight


class VelocitySetConstraint(BaseConstraint):
    """VelocitySetconstraints are made to set an upper and lower speed.
    They can be hard or soft, have different priorities, but they
    define a max and min for an expression derivative by setting it up
    as a max-min constraint in the optimization problem.
    For joint speed resolved controllers, the gain serves no purpose.
    WARNING: This doesn't run _check_sizes.

    Args:
        label (str): Name of the constraint
        expression (cs.MX): expression of the constraint
        gain (float,cs.MX,cs.DM,cs.np.ndarray): gain in the constraint
        set_min (list,cs.MX,cs.DM,cs.np.ndarray): minimum value of velocity
        set_max (list,cs.mx,cs.DM,cs.np.ndarray): maximum value of velocity
        constraint_type (str): "hard" or "soft", for opt.prob.controllers
        priority (int): sorting key of constraint
        slack_weight (float): optional slack_var weight
    """
    def __init__(self, label,
                 expression,
                 gain=1.0,
                 set_min=-1e10,
                 set_max=1e10,
                 constraint_type="hard",
                 priority=1,
                 slack_weight=1.0):
        BaseConstraint.__init__(self, label,
                                expression, gain)
        self.constraint_type = constraint_type
        self.priority = priority
        self.set_min = set_min
        self.set_max = set_max
        self.slack_weight = slack_weight

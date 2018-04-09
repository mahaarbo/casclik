"""Constraints and tools for the constraint objects.
Todo:
    * Add list functionality to set.
    * Add @property to the gain, set_min, & set_max that runs _check_sizes
    * Add a "combine" method to vertcat constraints.
    * Allow SetConstraints to have set_min & set_max that are expressions
"""
import casadi as cs
import logging


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
        # Todo add list check as well
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
                        raise TypeError("Unknown gain type. Supported are:"
                                        + " float, MX, DM, numpy.ndarray, and"
                                        + "list of floats/ints")
            if len(self.gain) == expr_size[0]:
                return True
        else:
            raise TypeError("Unknown gain type. Supported are: float, MX, DM,"
                            + " numpy.ndarray, and list of floats/ints.")
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

    """
    constraint_class = "EqualityConstraint"

    def __init__(self, label,
                 expression,
                 gain=1.0,
                 constraint_type="hard",
                 priority=1):
        BaseConstraint.__init__(self, label, expression, gain)
        self.constraint_type = constraint_type
        self.priority = priority
        if not self._check_sizes():
            raise ValueError("Gain and expression dimensions do not match.")


class SetConstraint(BaseConstraint):
    """Set constraints can be hard or soft, they can have different priorities
    but they all try to converge to a set of values.
    Args:
        label (str): Name of the constraint
        expression (cs.MX): expression of the constraint
        gain (float,cs.MX,cs.DM,cs.np.ndarray): gain in the constraint
        set_min (list, cs.MX, cs.DM, cs.np.ndarray): minimum value of set
        set_max (list, cs.MX, cs.DM, cs.np.ndarray): maximum value of set
        constraint_type (str): "hard" or "soft", for opt.prob. controllers
        priority (int): sorting key of constraint, for pseudoinv. controllers
    """
    constraint_class = "SetConstraint"

    def __init__(self, label,
                 expression,
                 gain=1.0,
                 set_min=0.0,
                 set_max=1.0,
                 constraint_type="hard",
                 priority=1):
        BaseConstraint.__init__(self, label,
                                expression, gain)
        self.constraint_type = constraint_type
        self.priority = priority
        self.set_min = set_min
        self.set_max = set_max
        if not self._check_sizes():
            raise ValueError("Gain, set limits, or expression dimensions"
                             + " do not match.")

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
            if szi[0] == expr_size[0] and szi[1] == 1:
                rmin = True
        else:
            raise TypeError("Unknown set_min type. Supported are float,"
                            + " MX, DM, and numpy.ndarray")
                
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
        elif isinstance(self.set_maz, cs.np.ndarray):
            sza = self.set_max.shape
            if sza[0] == expr_size[0] and szi == 1:
                rmax = True
        else:
            raise TypeError("Unknown set_max type. supported are float,"
                            + " MX, DM, and numpy.ndarray")
        return rgain and rmin and rmax


class VelocityEqualityConstraint(EqualityConstraint):
    pass


class VelocitySetConstraint(SetConstraint):
    pass

"""@package tasks
This package contains tools for tasks and the task classes.

"""
import casadi as cs

class Task(object):
    # Todo:
    #  - repr
    #  - add?
    #  - rank?
    #  - gain function?
    #  - Cone?
    pass


class EqualityTask(Task):
    def __init__(self, label, equation, gain):
        self.label = label
        self.equation = equation
        self.gain = gain
        self.jacobian_t = cs.jacobian

class SetTask(Task):
    def __init__(self, label, equation, gain, eq_min, eq_max):
        self.label = label
        self.equation = equation
        self.gain = gain
        self.eq_min = eq_min
        self.eq_max = eq_max

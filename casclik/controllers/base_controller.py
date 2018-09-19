class BaseController(object):
    def __init__(self, skill_spec):
        pass

    def __repr__(self):
        return self.controller_type+"<"+self.skill_spec.label+">"

import casadi as cs

def Rx(angle):
    

def axis_angle_to_rotation(axis, angle):
    # Source:
    # https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
    res = cs.MX.eye(3)
    + cs.sin(angle)*cs.skew(axis)
    +(1-cs.cos(angle))*cs.mtimes(cs.skew(axis), cs.skew(axis))
    return res


def rotation_to_angle(R):
    # Source:
    # https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
    return cs.arccos((cs.trace(R) - 1.0)/2.0)


def rotation_to_axis(R):
    denominator = cs.sqrt()
    vec = cs.vertcat(R[3, 2] - R[2, 3],
                     R[1, 3] - R[3, 1],
                     R[2, 1] - R[1, 2])
    
def manipulability_measure(J, version="yoshikawa"):
    """Returns a manipulability measure of the jacobian J.
    if H = JJ^T and describes the manipulation ellipsoid (ME)
    versions:
      "yoshikawa" - sqrt(det(H)), proportional to volume of ME
      "inverse"  - sn/s1, smallest/largest singular value of ME
      "smallest_singular_value" - simply the smallest singular value of ME
      "determinant"- determinant of J
      "smallest" - smallest value of abs(J)
    """
    # Convert everything to numpy things
    if isinstance(J, cs.MX):
        try:
            J = J.to_DM().toarray()
        except NotImplementedError:
            raise NotImplementedError("condition_number does not take symbolic"
                                      + " jacobians (type of J is MX)")
    elif isinstance(J, cs.SX):
        try:
            J = cs.DM(J).toarray()
        except NotImplementedError:
            raise NotImplementedError("condition_number does not take symbolic"
                                      + " jacobians (type of J is SX")
    elif isinstance(J, list):
        J = cs.np.array(J)
    H = cs.np.dot(J, J.T)

    if version.lower() == "yoshikawa":
        return cs.np.sqrt(cs.np.linalg.det(H))
    elif version.lower() == "inverse":
        U, S, V = cs.np.linalg.svd(H)
        return S[-1]/S[0]
    elif version.lower() == "smallest_singular_value":
        U, S, V = cs.np.linalg.svd(H)
        return S[-1]
    elif version.lower() == "determinant":
        return cs.np.linalg.det(J)
    elif version.lower() == "smallest":
        return J.min()

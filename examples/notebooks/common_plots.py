from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import casadi as cs


def joints(simres, axs=None, max_speed=None, label_suffix="", lstyle="-"):
    """Plot joint states wrt time from a simres dictionary.

    If given 2 axes, it will plot joints in the first and joint speeds
    in the second, else it will make a new figure. If given a single
    float max speed, it will show the limits.
    """
    nq = simres["q_sim"].shape[1]
    if axs is None:
        fig, axs = plt.subplots(2, 1)
    for i in xrange(nq):
        axs[0].plot(simres["t_sim"], simres["q_sim"][:, i],
                    label="q"+str(i)+label_suffix,ls=lstyle)
        axs[1].plot(simres["t_sim"], simres["dq_sim"][:, i],
                    label="dq"+str(i)+label_suffix,ls=lstyle)
    axs[0].set_ylabel("Position [rad]")
    axs[0].set_xlabel("time [s]")
    axs[0].legend()
    axs[1].set_ylabel("Speed [rad/s]")
    axs[0].set_ylabel("position [rad]")
    if max_speed is not None:
        axs[1].plot([min(simres["t_sim"]), max(simres["t_sim"])],
                    [max_speed, max_speed], 'k--')
        axs[1].plot([min(simres["t_sim"]), max(simres["t_sim"])],
                    [-max_speed, -max_speed], 'k--')
    return axs


def pos_point(simres, ax=None, p_des=None, label_suffix="", lstyle="-"):
    """Plot position wrt time from a simres dictionary."""
    ns = simres["p_sim"].shape[1]
    cmap = plt.get_cmap("tab10")
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    for i in xrange(ns):
        ax.plot(simres["t_sim"], simres["p_sim"][:, i],
                color=cmap(i), label=chr(ord("x")+i)+label_suffix,
                ls=lstyle)
        if p_des is not None:
            if not isinstance(p_des, cs.Function):
                ax.plot([min(simres["t_sim"]), max(simres["t_sim"])],
                        [p_des[i], p_des[i]], color=cmap(i), ls="--",
                        label=chr(ord("x")+i)+"_des")
            else:
                p_des_sim = [p_des(t).toarray()[i] for t in simres["t_sim"]]
                ax.plot(simres["t_sim"], p_des_sim, color=cmap(i),
                        ls="--", label=chr(ord("x")+i)+"_des")
    ax.legend()
    return ax


def pos_point_3d(simres, ax=None, p_des=None):
    """Plot 3D of converging to the desired point"""
    if ax is None:
        ax = Axes3D(plt.figure())
    ax.plot(simres["p_sim"][:, 0],
            simres["p_sim"][:, 1],
            simres["p_sim"][:, 2])
    if p_des is not None:
        ax.scatter(p_des[0], p_des[1], p_des[2], s=20, color="k")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    return ax


def frame_3d(simres,  ax=None, p_des=None, T_des=None):
    """Plot 3D of converging to the desired frame."""
    if ax is None:
        ax = Axes3D(plt.figure())
    ax.plot(simres["p_sim"][:, 0],
            simres["p_sim"][:, 1],
            simres["p_sim"][:, 2], "k")
    # make the end point
    if p_des is not None:
        ax.scatter(p_des[0], p_des[1], p_des[2], s=20, c="k")
    elif T_des is not None:
        ax.scatter(T_des[0, 3], T_des[1, 3], T_des[2, 3], s=20, c="k")
    r_sim = cs.np.zeros_like(simres["p_sim"])
    g_sim = cs.np.zeros_like(simres["p_sim"])
    b_sim = cs.np.zeros_like(simres["p_sim"])
    for i in range(len(simres["t_sim"])):
        r_sim[i, :] = simres["p_sim"][i, :] + cs.np.dot(simres["R_sim"][i, :, :], cs.np.array([0.1, 0., 0.]))
        g_sim[i, :] = simres["p_sim"][i, :] + cs.np.dot(simres["R_sim"][i, :, :], cs.np.array([0., 0.1, 0.]))
        b_sim[i, :] = simres["p_sim"][i, :] + cs.np.dot(simres["R_sim"][i, :, :], cs.np.array([0., 0., 0.1]))
    ax.plot(r_sim[:, 0],
            r_sim[:, 1],
            r_sim[:, 2],
            "r--")
    ax.plot(g_sim[:, 0],
            g_sim[:, 1],
            g_sim[:, 2],
            "g--")
    ax.plot(b_sim[:, 0],
            b_sim[:, 1],
            b_sim[:, 2],
            "b--")
    if T_des is not None:
        r_des_x = T_des[0, 3] + T_des[0, 0]*.1
        r_des_y = T_des[1, 3] + T_des[1, 0]*.1
        r_des_z = T_des[2, 3] + T_des[2, 0]*.1
        g_des_x = T_des[0, 3] + T_des[0, 1]*.1
        g_des_y = T_des[1, 3] + T_des[1, 1]*.1
        g_des_z = T_des[2, 3] + T_des[2, 1]*.1        
        b_des_x = T_des[0, 3] + T_des[0, 2]*.1
        b_des_y = T_des[1, 3] + T_des[1, 2]*.1
        b_des_z = T_des[2, 3] + T_des[2, 2]*.1
        ax.scatter([r_des_x, g_des_x, b_des_x],
                   [r_des_y, g_des_y, b_des_y],
                   [r_des_z, g_des_z, b_des_z],
                   s=20, c=["r", "g", "b"])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    return ax

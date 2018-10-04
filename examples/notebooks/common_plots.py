from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def joints(simres, axs=None, max_speed=None):
    """Plot joint states wrt time from a simres dictionary.

    If given 2 axes, it will plot joints in the first and joint speeds
    in the second, else it will make a new figure. If given a single
    float max speed, it will show the limits.
    """
    nq = simres["q_sim"].shape[1]
    if axs is None:
        fig, axs = plt.subplots(2, 1)
    for i in xrange(nq):
        axs[0].plot(simres["t_sim"], simres["q_sim"][:, i], label="q"+str(i))
        axs[1].plot(simres["t_sim"], simres["dq_sim"][:, i], label="dq"+str(i))
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


def pos_point(simres, ax=None, p_des=None):
    """Plot position wrt time from a simres dictionary."""
    ns = simres["p_sim"].shape[1]
    cmap = plt.get_cmap("tab10")
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    for i in xrange(ns):
        ax.plot(simres["t_sim"], simres["p_sim"][:, i],
                color=cmap(i), label=chr(ord("x")+i))
        if p_des is not None:
            ax.plot([min(simres["t_sim"]), max(simres["t_sim"])],
                    [p_des[i], p_des[i]], color=cmap(i), ls="--",
                    label=chr(ord("x")+i)+"_des")
    ax.legend()
    return ax


def pos_point_3d(simres, ax=None, p_des=None):
    """Plot 3D of converging to the desired point"""
    if ax is None:
        ax = Axes3D(plt.figure())
    ax.plot(simres["p_sim"][:, 0],
            simres["p_sim"][:, 0],
            simres["p_sim"][:, 0])
    if p_des is not None:
        ax.scatter(p_des[0], p_des[1], p_des[2], s=20, color="k")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    return ax

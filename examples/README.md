# CASCLIK EXAMPLES
The notebook folder contains jupyter notebook examples of some standard uses and comparison of controllers. We recommend using matplotlib >3.0.2, and ffmpeg needs to be installed to see animations.

## cart\_on\_track\_1D\_comparison\_of\_controllers
In this example we control a 1D cart on a track. This is the smallest example showing the different controllers at work. The cart is made to move to a point, move to a trajectory, and move to follow a path. The track has a limited length, and the example showcases the combination of equality constraints and set constraints. The path following example shows a virtual variable (the path following variable) that the cart uses to figure out which along-path speed it should use. The example does not require any additional libraries. 

## double\_pendulum\_2D\_comparison\_of\_controllers
In this example we are controlling a fully actuated two-link manipulator situated on a table. two-link manipulator moves to a point while avoiding the table, tracks a trajectory with a nonlinear cost equal to the kinetic energy of the system. As this is a minimal example, the result of using the nonlinear cost in the NLP does not result in a highly different joint speed from the QP controller. The example does not require any additional libraries.

*Currently incomplete, feel free to contribute a more complete example.*

## ur5\_dual\_quaternion\_vs\_transformation\_matrix
In this example we are exploring some different underlying representations and constraint formulations for use with a UR5 robot. We try to perform trajectory tracking and moving to a pose in space, either using dual quaternions or transformation matrices as the underlying representation. The example requires [urd2fcasadi](https://github.com/mahaarbo/urdf2casadi) which requires ROS' [urdf\_parser\_py](https://github.com/ros/urdf_parser_py).

## ur5\_input\_experiment
In this example the UR5 is made to track an position input. The input is initiated after 10 seconds and has a sinusoidal motion.

## ur5\_moe2016\_example2
This is an implementation of the second example from [Moe2016](https://www.frontiersin.org/articles/10.3389/frobt.2016.00016/full). The example shows a UR5 tracking a trajectory while remaining in a box for the different controllers available in CASCLIK. The example requires [urd2fcasadi](https://github.com/mahaarbo/urdf2casadi) which requires ROS' [urdf\_parser\_py](https://github.com/ros/urdf_parser_py).

### Troubleshooting
1. If you get a `failed to remove temp` etc. error, you may be trying to run multiple examples at the same time. This seems to cause an issue with the compiler/casadi. Try again with a single example. If that doesn't work, file an issue.

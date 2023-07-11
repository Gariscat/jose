import mujoco
import time

import mujoco
import mujoco.viewer
import numpy as np
import gym

env = gym.make('Humanoid-v2')

xml_path = 'humanoid.xml'
frame_skip = 5
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)


### print(dir(m))

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()
    action = env.action_space.sample()
    print()
    d.ctrl[:] = action
    mujoco.mj_step(m, d, nstep=frame_skip)

    # As of MuJoCo 2.0, force-related quantities like cacc are not computed
    # unless there's a force sensor in the model.
    # See https://github.com/openai/gym/issues/1541
    mujoco.mj_rnePostConstraint(m, d)
    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    #### mujoco.mj_step(m, d)
    print(d.geom('head'))

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
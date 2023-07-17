from dm_control.mujoco import wrapper, Physics
import mujoco
import imageio
import gym
from tqdm import tqdm, trange
import numpy as np

env = gym.make('Humanoid-v2')

xml_path = 'humanoid.xml'
frame_skip = 5
m = mujoco.MjModel.from_xml_path(xml_path)

m = wrapper.MjModel(m)
d = wrapper.MjData(m)

physics = Physics(d)

views = {'ego': [], 'track': []}

print(physics.control().shape)

for _ in range(5):
    physics.reset()
    for step in trange(100, desc='simulating:'):
        action = np.zeros(18)
        action[-1] = np.random.uniform(-0.4, 0.4)
        # d.ctrl[:] = action
        physics.set_control(action)
        physics.step(nstep=5)
        for camera_name in views:
            img = physics.render(camera_id=camera_name)
            views[camera_name].append(img)
        # print(img.shape)
        
for camera_name in views:
    with imageio.get_writer(f'{camera_name}.gif', mode='I') as writer:
        for img in tqdm(views[camera_name], desc='saving gif:'):
            writer.append_data(img)
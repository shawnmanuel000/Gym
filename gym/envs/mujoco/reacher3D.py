import io
import copy
import numpy as np
from ...utils import EzPickle
from .mujoco_env import MujocoEnv
from mujoco_py import load_model_from_xml
import xml.etree.ElementTree as ET

target_geom = """
    <geom conaffinity="0" contype="0" name="target" pos="0 0 0" rgba="0.8 0.2 0.4 0.8" size=".002" type="sphere"/>
"""

class ReacherEnv3D(MujocoEnv, EzPickle):
    def __init__(self):
        EzPickle.__init__(self)
        MujocoEnv.__init__(self, 'reacher3D.xml', 2)

    def load_model(self, path):
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()
        return self.new_model()

    def new_model(self):
        root = copy.deepcopy(self.root)
        worldbody = root.find("worldbody")
        self.create_path(worldbody)
        return self.load_model_from_xml(root)

    def load_model_from_xml(self, root):
        with io.StringIO() as string:
            string.write(ET.tostring(root, encoding="unicode"))
            
            model = load_model_from_xml(string.getvalue())
            return model

    def create_path(self, worldbody):
        self.range = 0.5*np.array([-0.25, -0.6, 0.15, 0.25, -0.2, 0.25])
        mul = 2*np.random.random() - 1
        amp = np.random.random()
        X = mul*np.linspace(self.range[0], self.range[3], 20)
        Y = 0.5*(self.range[1]+self.range[4]) + amp*(self.range[1]-self.range[4])*np.sin((X+self.range[0])/(self.range[3]-self.range[0])/mul*2*np.pi)/2
        Z = 0.2 * np.ones_like(Y)

        self.path_names = []
        ele = ET.Element("body")
        ele.set("name", "path")
        ele.set("pos", "0 0 0")
        for i,(x,y,z) in enumerate(zip(X,Y,Z)):
            ele.append(self.create_target(f"path{i}", f"{x} {y} {z}"))
            self.path_names.append(f"path{i}")
        target = worldbody.findall("./*[@name='target']")[0]
        target.set("pos", f"{X[-1]} {Y[-1]} {Z[-1]}")
        worldbody.append(ele)

    def create_target(self, name, pos):
        ele = ET.Element("body")
        ele.set("name", name)
        ele.set("pos", pos)
        geo = ET.fromstring(target_geom)
        geo.set("name", name)
        ele.append(geo)
        return ele

    def step(self, a):
        ef_pos = self.get_body_com("fingertip")
        target_dist = ef_pos-self.get_body_com("target")
        path_dists = [ef_pos-self.get_body_com(name) for name in self.path_names]
        reward_dist = - np.linalg.norm(target_dist)
        reward_path = - np.min(np.linalg.norm(path_dists, axis=-1))
        # reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_path
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_path)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        self.model = self.new_model()
        qpos = self.np_random.uniform(low=-0.2, high=0.2, size=self.model.nq) + self.init_qpos
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        # while True:
        #     self.goal = self.np_random.uniform(low=-.2, high=.2, size=3)
        #     if np.linalg.norm(self.goal) < 0.2:
        #         break
        # qpos[-3:] = self.goal
        # qvel[-3:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:-3],
            self.sim.data.qvel.flat[:-3],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
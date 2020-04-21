import io
import re
import copy
import random
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
        self.path_names = []
        ele = ET.Element("body")
        ele.set("name", "path")
        ele.set("pos", "0 0 0")
        for i in range(10):
            ele.append(self.create_target(f"path{i}"))
            self.path_names.append(f"path{i}")
        # target = worldbody.findall("./*[@name='target']")[0]
        # target_pos = origin+0.8*np.random.uniform(start, end)
        # target.set("pos", ' '.join([str(p) for p in target_pos]))
        worldbody.append(ele)

    def create_target(self, name, pos="0 0 0"):
        ele = ET.Element("body")
        ele.set("name", name)
        ele.set("pos", pos)
        geo = ET.fromstring(target_geom)
        geo.set("name", name)
        ele.append(geo)
        return ele

    def step(self, a):
        ef_pos = self.get_body_com("fingertip")
        target_pos = self.get_body_com("target")
        path = [self.get_body_com(name) for name in self.path_names]
        target_dist = ef_pos-target_pos
        path_dists = [ef_pos-path_pos for path_pos in path]
        reward_dist = - np.linalg.norm(target_dist)
        reward_path = - np.min(np.linalg.norm(path_dists, axis=-1))
        reward = reward_dist + reward_path
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs(ef_pos, target_pos, path)
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_path)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        self.start = np.array([-0.25, -0.1, 0.15])
        self.end = np.array([0.25, -0.3, 0.25])
        origin = (self.start+self.end)/2
        start = self.start-origin
        end = self.end-origin
        target_pos = origin+0.6*np.random.uniform(start, end)
        self.model.body_pos[self.model.body_names.index("target")] = target_pos

        qpos = 0.5*self.np_random.uniform(low=-1, high=1, size=self.model.nq) + self.init_qpos
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.sim.step()

        ef_pos = self.get_body_com("fingertip")
        target_pos = self.get_body_com("target")
        points = np.linspace(ef_pos, target_pos, len(self.path_names))
        path_indices = [self.model.body_names.index(name) for name in self.path_names]
        for i,point in zip(path_indices, points):
            self.model.body_pos[i] = point
        path = [self.get_body_com(name) for name in self.path_names]
        return self._get_obs(ef_pos, target_pos, path)

    def _get_obs(self, pos, target, path):
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat, pos, target, *path])
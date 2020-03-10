import mujoco_py
import os
import matplotlib.pyplot as plt
from gym.envs.mujoco import hopper_v3
from lxml import etree
import numpy as np
import random
class BlockHopperEnv(hopper_v3.HopperEnv):
	def __init__(self, xml_file="./hopper.xml"):
		self.current_xml = xml_file
		tree = etree.parse(xml_file)
		for elem in tree.findall('worldbody/body/geom'):
			if elem.attrib['name'] == 'box_geom':
				self.box_position = float(elem.attrib['fromto'].split(" ")[3])
		hopper_v3.HopperEnv.__init__(self, xml_file=xml_file)
		self.model = mujoco_py.load_model_from_path(xml_file)
		self.sim = mujoco_py.MjSim(self.model)
	
	def reset_model(self):
		super().reset_model()
		tree = etree.parse(self.current_xml)
		for elem in tree.findall('worldbody/body/geom'):
			if elem.attrib['name'] == 'box_geom':
				xpos = round(random.uniform(-2.5,-.3),2)
				elem.attrib['fromto'] = f'{xpos} 0 0 {round(xpos+0.1,2)} 0 0'
				elem.attrib['rgba'] = f'{random.uniform(0,1)} {random.uniform(0,1)} {random.uniform(0,1)} 1'
				self.box_position = round(xpos+0.1,2)
		if not os.path.isdir("./extra"):
			os.mkdir("./extra")
		self.current_xml = f"./extra/hopper{random.randint(0,2**10)}.xml"
		with open(self.current_xml, 'wb') as f:
			f.write(etree.tostring(tree, pretty_print=True, encoding='utf8'))
		self.model = mujoco_py.load_model_from_path(self.current_xml)
		self.sim = mujoco_py.MjSim(self.model)
	def _get_image(self, size=1024):
		return self.sim.render(camera_name='track',height=size,width=size)[::-1, :]

	def step(self, action, size=128):
		observation, reward, done, info = super().step(action)
		xpos = self.sim.data.qpos[0]
		dist_from_box = np.sqrt(np.power(xpos - self.box_position,2))
		reward -= np.log(abs(dist_from_box+1e-9))
		return observation, reward, done, info



if __name__ == "__main__":
	env = BlockHopperEnv()
	obs = env._get_obs()

	for i in range(5):
		observation, reward, done, info = env.step([1,0,1])
		img= env._get_image()
		print(reward)
		plt.imshow(img)		
		plt.show()
	
	env.reset_model()
	observation, reward, done, info = env.step([1,1,1])
	img= env._get_image()
	plt.imshow(img)		
	plt.show()
	#while(1): env.render()


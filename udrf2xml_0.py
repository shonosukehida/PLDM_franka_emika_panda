import mujoco

model = mujoco.MjModel.from_xml_path("panda.urdf")  # 拡張子が.urdfでもOK
mujoco.mj_saveLastXML("panda.xml", model)

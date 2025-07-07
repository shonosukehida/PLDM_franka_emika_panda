from dm_control import mjcf

model = mjcf.from_path("panda.urdf")
xml_str = mjcf.to_xml_string(model)

with open("panda.xml", "w") as f:
    f.write(xml_str)

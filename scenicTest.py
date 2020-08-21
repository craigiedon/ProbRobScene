import scenic3d

scenario = scenic3d.scenarioFromFile("simpleCube.scenic")

for i in range(1):
    ex_world, used_its = scenario.generate()
    ex_world.show()

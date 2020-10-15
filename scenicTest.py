import scenic3d

scenario = scenic3d.scenarioFromFile("scenarios/simpleCube.scenic")

for i in range(10):
    ex_world, used_its = scenario.generate(verbosity=2)
    ex_world.show_3d()

import scenic3d

scenario = scenic3d.scenarioFromFile("scenarios/tableCube.scenic")

for i in range(1):
    print(f"Generation {i}")
    ex_world, used_its = scenario.generate(verbosity=2)
    ex_world.show_3d()

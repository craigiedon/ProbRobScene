import probRobScene
import sys

if __name__ == "__main__":

    if sys.argv != 3:
        print("Usage: python runScenarioRaw.py <scenario-file> <max-generations>")
        sys.exit(0)

    scenario_file = sys.argv[1]
    max_generations = sys.argv[2]

    scenario = probRobScene.scenario_from_file(scenario_file)

    for i in range(max_generations):
        print(f"Generation {i}")
        ex_world, used_its = scenario.generate(verbosity=2)
        ex_world.show_3d()

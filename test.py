import traci

if __name__ == '__main__':
    traci.start(['sumo-gui', '-c', 'TwoLoops/TwoLoops.sumocfg'])

    # To avoid re-applying the same strategy
    strategy = 1

    # To not immediately change logic in the next step
    # Give sometime to the current logic to run
    last = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        north = traci.edge.getLastStepVehicleNumber("-E3")
        south = traci.edge.getLastStepVehicleNumber("-E2")
        east = traci.edge.getLastStepVehicleNumber("-E5")
        west = traci.edge.getLastStepVehicleNumber("E0")

        if north + south + east + west < 30 and strategy != 1 and last > 30:
            print("Using strategy 1")
            traci.trafficlight.setProgram("J1", "1")
            strategy = 1
            last = 0
        if north + south + east + west < 50 and strategy != 2 and last > 30:
            print("Using strategy 2")
            traci.trafficlight.setProgram("J1", "2")
            strategy = 2
            last = 0
        elif north + south + east + west >= 50 and strategy != 3 and last > 30:
            print("Using strategy 3")
            traci.trafficlight.setProgram("J1", "3")
            strategy = 3
            last = 0

        last = last + 1

    traci.close()

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

sim.startSimulation()

objs = sim.getObjectsInTree(sim.handle_scene)

for obj in objs:

    try:
        print(sim.getObjectAlias(obj))
    except:
        pass

sim.stopSimulation()
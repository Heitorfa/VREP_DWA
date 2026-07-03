from coppeliasim_zmqremoteapi_client import RemoteAPIClient

print("Tentando conectar...")

client = RemoteAPIClient()

sim = client.require('sim')

print("Conectado!")

print(sim.getSimulationState())
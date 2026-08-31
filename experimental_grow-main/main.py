import gromo
from gromo.graph_network.dag_growing_network import GraphGrowingNetwork

if __name__ == "__main__":
    print(gromo.__version__)

    net = GraphGrowingNetwork()
    print(net.device)
    print(net.logger.enabled)

    net = GraphGrowingNetwork(device="cpu")
    print(net.device)

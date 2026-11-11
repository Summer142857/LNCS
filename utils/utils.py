"""Text-attributed instance prompts and data helpers."""

import numpy as np
import torch
import networkx as nx
from scipy.spatial import cKDTree
from lncs.problems import KP_CAPACITIES


def calculate_top_k_nearest_nodes(nodes, k=2):
    # Build a k-d tree from the list of nodes
    kdtree = cKDTree(nodes)

    top_k_nearest_nodes = []

    # Query for the top-k nearest nodes for each node
    for node in nodes:
        distances, indices = kdtree.query(
            node, k + 1
        )  # k + 1 to include the node itself
        sorted_indices = indices[
            1:
        ]  # Exclude the first index (index of the node itself)
        distances = distances[1:]
        top_k_nearest_nodes.append(
            [(j, distances[i]) for i, j in enumerate(sorted_indices)]
        )

    return top_k_nearest_nodes


def get_rank(instance, problem="KP"):
    if problem == "KP":
        ratios = torch.where(
            instance[:, 0] == 0,
            torch.where(instance[:, 1] > 0, torch.inf, 0.0),
            instance[:, 1] / instance[:, 0],
        )
        v2w = [round(i, 4) for i in ratios.tolist()]
        rank = [sorted(v2w, reverse=True).index(x) + 1 for x in v2w]
        return v2w, rank
    elif problem == "MVC":
        # instance -> the adj matrix
        degrees = np.sum(instance, axis=-1).tolist()
        rank = [sorted(degrees, reverse=True).index(x) + 1 for x in degrees]
        return degrees, rank
    elif problem == "SMTWTP":
        # instance -> the due times
        return [sorted(instance).index(x) + 1 for x in instance]
    else:
        raise NotImplementedError


def _core_tag_prompt(instance, problem="TSP", k_nn=3, task_description=True):
    """
    build text-attribute graph for instances
    """
    if problem == "TSP":
        p_size = instance.shape[0]
        description = [
            "For a traveling salesman problem (TSP),there will be a list of nodes distributed in an unit square, representing a series of cities."
            "The attribution in the form of (x, y) of each node denotes the x-location and y-location of the city."
            "The goal is to find the shortest route that visits each city exactly once and returns to the origin city."
            "The following are the descriptions of " + str(p_size) + " nodes of a TSP: "
        ]
        for i in range(instance.shape[0]):
            description.append(
                "Node ({}).Attribution:".format(i)
                + str([round(a, 4) for a in instance[i].tolist()])
            )
            if i != 0:
                description[i] = description[i] + "."
        description[-1] = description[-1] + "."
        if k_nn != 0:
            nns = calculate_top_k_nearest_nodes(instance.cpu().numpy(), k_nn)
            for i in range(1, instance.shape[0] + 1):
                description[i] = description[
                    i
                ] + "The {} nearest nodes and distances:".format(k_nn)
                description[i] = description[i] + str(
                    ["({}): {:.4f}".format(n[0], n[1]) for n in nns[i - 1]]
                ).replace("'", "")
                if i != instance.shape[0] - 1:
                    description[i] = description[i] + ";"
                else:
                    description[i] = description[i] + "."
    elif problem == "CVRP":
        p_size = instance[1].shape[0]
        description = [
            "For a Capacitated Vehicle Routing Problem (CVRP),there will be a depot node and a list of customer nodes distributed in an unit square."
            "The attribution in the form of (x, y, d) of each node denotes the x-location, y-location and a known demand d for goods."
            "Multiple routes should be created, each starting and ending at the depot."
            "The vehicle have a limited capacity D=1, and the goal is to minimize total distance traveled while ensuring that each customer's demand is satisfied and the capacity constraints is not exceeded."
            "The following are the descriptions of a depot node and "
            + str(p_size)
            + " nodes of a CVRP: "
        ]
        description.append(
            "Node (0). Depot node. Attribution: "
            + str([round(a, 4) for a in instance[0][0].tolist()])
        )
        description[-1] = description[-1] + ";"
        for i in range(instance[1].shape[0]):
            attribution = [round(a, 4) for a in instance[1][i].tolist()] + [
                round(instance[2][i].tolist(), 4)
            ]
            description.append(
                "Node ({}). Customer node. Attribution:".format(i + 1)
                + str(attribution)
                + "."
            )
        if k_nn != 0:
            nns = calculate_top_k_nearest_nodes(
                torch.concat([instance[0], instance[1]], dim=0).cpu().numpy(), k_nn
            )
            for i in range(1, instance[1].shape[0] + 2):
                description[i] = description[
                    i
                ] + " The {} nearest nodes and distances:".format(k_nn)
                description[i] = description[i] + str(
                    ["({}): {:.4f}".format(n[0], n[1]) for n in nns[i - 1]]
                ).replace("'", "")
                if i != instance[1].shape[0] - 1:
                    description[i] = description[i] + ";"
                else:
                    description[i] = description[i] + "."
    elif problem == "KP":
        p_size = instance.shape[0]
        capacity = KP_CAPACITIES[p_size]
        description = [
            "For a knapsack problem (KP),there will be a list of nodes distributed in an unit square,representing a series of items."
            "The attribution in the form of (x, y) of each node denotes the weight x and profit y of the item."
            "Given a bag with capacity "
            + str(capacity)
            + ", the goal is to put the items into the bag such that the sum of profits is the maximum possible."
            "The following are the descriptions of " + str(p_size) + " nodes of a KP: "
        ]
        for i in range(instance.shape[0]):
            description.append(
                "Node ({}). Attribution:".format(i)
                + str([round(a, 4) for a in instance[i].tolist()])
            )
            if i != 0:
                description[i] = description[i] + "."
        description[-1] = description[-1] + "."
        if k_nn != 0:
            v2w, rank = get_rank(instance, problem="KP")
            for i in range(1, instance.shape[0] + 1):
                description[i] = (
                    description[i] + "Value-to-weight ratio and importance rank:"
                )
                description[i] = description[i] + str([v2w[i - 1], rank[i - 1]])
                if i != instance.shape[0]:
                    description[i] = description[i] + ";"
                else:
                    description[i] = description[i] + "."
    elif problem == "MVC":
        n_nodes = instance.number_of_nodes()
        n_edges = instance.number_of_edges()
        adj = nx.adjacency_matrix(instance).todense()
        description = [
            "For a minimum vertex cover (MVC) problem,there will be a graph with "
            + str(n_nodes)
            + " nodes and "
            + str(n_edges)
            + "edges. "
            "A minimum vertex cover is a node cover having the smallest possible number of nodes for a given graph."
            "The attribution in the form of (x1,x2,...,x_"
            + str(n_nodes)
            + ") of a node denotes the adjacency relationship of itself and other nodes."
            "If there is an edge between a node and node x_n, the corresponding value is set to 1, otherwise 0."
            "The following are the descriptions of "
            + str(n_nodes)
            + " nodes of a MVC problem: "
        ]
        for i in range(0, n_nodes):
            attr = "Node ({}). Attribution:".format(i) + str(adj[i])
            description.append(attr.replace("\n", ""))
            if i != 0:
                description[i] = description[i] + "."
        description[-1] = description[-1] + "."
        if k_nn != 0:
            degree, rank = get_rank(adj, problem="MVC")
            for i in range(1, n_nodes + 1):
                description[i] = description[i] + "Node degree and importance rank:"
                description[i] = description[i] + str([degree[i - 1], rank[i - 1]])
                if i != n_nodes + 1:
                    description[i] = description[i] + ";"
                else:
                    description[i] = description[i] + "."
    elif problem == "SMTWTP":
        due_time, weights, processing_time = instance
        p_size = due_time.shape[0]
        description = [
            "For a single machine total weighted tardiness problem (SMTWTP),there will be a list of nodes,representing a set of jobs must be processed by a single machine."
            "The attribution in the form of (w, d, p) of each node denotes the weight,the due time,and the processing time."
            "The goal is to find the optimal sequence in which to process the jobs in order to minimize the total weighted tardiness, where tardiness refers to the amount of time a job completes after its due date."
            "The following are the description of "
            + str(p_size)
            + " nodes of a SMTWTP: "
        ]
        for i in range(0, p_size):
            attr = "Node ({}). Attribution:".format(i) + str(
                [
                    round(weights[i].tolist(), 4),
                    round(due_time[i].tolist(), 4),
                    round(processing_time[i].tolist(), 4),
                ]
            )
            description.append(attr.replace("\n", ""))
            if i != 0:
                description[i] = description[i] + "."
        description[-1] = description[-1] + "."
        if k_nn != 0:
            rank = get_rank(due_time, problem="SMTWTP")
            for i in range(1, p_size + 1):
                description[i] = description[i] + "Node importance rank:"
                description[i] = description[i] + str([rank[i - 1]])
                if i != p_size:
                    description[i] = description[i] + ";"
                else:
                    description[i] = description[i] + "."
    else:
        raise NotImplementedError("Not define problem type yet.")
    if task_description:
        return description
    else:
        return description[1:]


def tag_prompt(instance, problem="TSP", k_nn=3, task_description=True):
    """Use the original five-task prompts, plus explicit transfer-task prompts."""
    if problem == "MIS":
        description = _core_tag_prompt(instance, "MVC", k_nn, True)
        n = instance.number_of_nodes()
        description[0] = (
            "For a maximum independent set (MIS) problem, there will be a graph "
            f"with {n} nodes and {instance.number_of_edges()} edges. "
            "The goal is to select the largest set of nodes such that no two "
            "selected nodes are connected by an edge. The attribution of each "
            "node is its adjacency relationship with all nodes: 1 for an edge "
            "and 0 otherwise. The following are the descriptions of the nodes: "
        )
    elif problem == "VRPB":
        description = _core_tag_prompt(instance, "CVRP", k_nn, True)
        description[0] = (
            "For a vehicle routing problem with backhauls (VRPB), there will be "
            "a depot and customer nodes in a unit square. Each customer has "
            "attribution (x, y, d): coordinates and demand. Positive demand "
            "requires unloading and negative demand requires loading goods. "
            "Each route starts and ends at the depot. Vehicle capacity is 1. "
            "The goal is to minimize total travel distance while visiting every "
            "customer once and respecting vehicle capacity. "
            f"The following describe a depot and {instance[1].shape[0]} customers: "
        )
    else:
        return _core_tag_prompt(instance, problem, k_nn, task_description)
    return description if task_description else description[1:]

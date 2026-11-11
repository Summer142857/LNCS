"""Problem definitions shared by generation, prompts and CLI validation."""

TASKS = ("TSP", "CVRP", "KP", "MVC", "SMTWTP", "VRPB", "MIS")
SIZES = (20, 50, 100)
KP_CAPACITIES = {20: 5, 50: 12.5, 100: 25}
VRP_CAPACITIES = {20: 30, 50: 40, 100: 50}

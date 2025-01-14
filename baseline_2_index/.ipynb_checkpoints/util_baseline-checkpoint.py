# import required packages
import math
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def travel_time_computer(u, w, node_data):
    """
    Computes the travel time between two nodes `u` and `w`.

    This function calculates the travel time as the sum of the Euclidean distance
    between nodes `u` and `w` and the service time of node `u`. Adding the service
    is based on the paper assumption

    Args:
        u (int): The identifier for the starting node.
        w (int): The identifier for the destination node.

    Returns:
        float: The computed travel time.

    """
    travel_time = euclidean_dist(
                    node_data[u]["x"], node_data[u]["y"],
                    node_data[w]["x"], node_data[w]["y"]
                ) +  node_data[u]["service"]
    return travel_time

def euclidean_dist(x1, y1, x2, y2):
    """
    Calculates the Euclidean distance between two points in a 2D plane.

    Args:
        x1 (float): The x-coordinate of the first point.
        y1 (float): The y-coordinate of the first point.
        x2 (float): The x-coordinate of the second point.
        y2 (float): The y-coordinate of the second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return round(distance, 2)

def route_feasible(alpha_node, u_node, w_node, bar_alpha_node, capacity):
    """
    Returns False if u -> v route is impossible due to capacity or time windows;
    True otherwise.
    """

    # Checking capacity constraint
    if u_node["demand"] + w_node["demand"] > capacity:
        return False

    # checking time window constraints
    dist_alpha_u = euclidean_dist(alpha_node["x"], alpha_node["y"], u_node["x"], u_node["y"])
    arrival_u = alpha_node["early"] + alpha_node["service"] + dist_alpha_u
    start_u = max(u_node["early"], arrival_u)
    if start_u > u_node["late"]:
        return False
    finish_u = start_u + u_node["service"]

    # Arrive at w from u
    dist_u_w = euclidean_dist(u_node["x"], u_node["y"], w_node["x"], w_node["y"])
    arrival_w = finish_u + dist_u_w
    start_w = max(w_node["early"], arrival_w)
    
    if start_w > w_node["late"]:
        return False
    finish_w = start_w + w_node["service"]

    # Step C: Arrive at bar_alpha from w
    dist_w_bar = euclidean_dist(w_node["x"], w_node["y"], bar_alpha_node["x"], bar_alpha_node["y"])
    arrival_bar_alpha = finish_w + dist_w_bar
    if arrival_bar_alpha > bar_alpha_node["late"]:
        return False

    # If none of these checks fail => feasible
    return True


def generate_E_star(all_nodes, node_data, alpha, bar_alpha, d0):
    """
    This function generate feasible arcs
    """
    E_star = {}
    for u in all_nodes:
        for w in all_nodes:
            if u != w:
                # Skip arcs into alpha or out of bar_alpha
                if w == alpha: 
                    continue
                if u == bar_alpha:
                    continue
    
                # Check capacity/time feasibility of the "mini-route" alpha->u->w->bar_alpha
                feasible = route_feasible(
                    node_data[alpha], node_data[u],
                    node_data[w], node_data[bar_alpha], d0
                )
                if feasible:
                    # If feasible, keep (u,w)
                    dist_uv = euclidean_dist(
                        node_data[u]["x"], node_data[u]["y"],
                        node_data[w]["x"], node_data[w]["y"]
                    )
                    E_star[(u,w)] = dist_uv
    return E_star

def solve_model(all_nodes, customers, node_data, E_star, t0, d0, alpha, bar_alpha):
    """
    Solves the model using Gurobi solver

    Args:
        all_nodes: The identifier for the starting node.
        customers: The identifier for the destination node.
        node_data: The data for nodes
        E_star: feasible arcs
        t0: the time of vehicle at the starting depot
        d0: the capacity of vehicle at the starting depot
        alpha: staring depot
        bar_alpha: ending depot

    Returns:
        float: The objective function value

    """
    # initialing the model
    model = gp.Model("VRPTW_TwoIndex_with_PrunedArcs")
    
    
    # Defining x which are routing decisions
    x = {}
    for (u,w) in E_star.keys():
        x[(u,w)] = model.addVar(vtype=GRB.BINARY, name=f"x_{u}_{w}")
    
    # defining tau_u decision variables
    tau = {}
    for u in all_nodes:
        tau[u] = model.addVar(lb=node_data[u]["t_minus_budget"], ub=node_data[u]["t_plus_budget"], vtype=GRB.CONTINUOUS, name=f"tau_{u}")
    
    # defining delta_u decision variables
    delta = {}
    for u in all_nodes:
        delta[u] = model.addVar(lb=node_data[u]['demand'], ub=d0, vtype=GRB.CONTINUOUS, name=f"delta_{u}")
    
    
    # defining objective function
    obj_expr = gp.quicksum(E_star[(u,w)] * x[(u,w)] for (u,w) in E_star.keys())
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Force start depot to have tau[alpha] = t0 (this is based on paper budget remaining logic)
    model.addConstr(tau[alpha] == t0, "start_time_resource")
    
    # Force start depot to have delta[alpha] = d0 (this is based on paper budget remaining logic)
    model.addConstr(delta[alpha] == d0, name="start_capacity")
    
    # Each customer has exactly 1 incoming arc
    for u in customers:
        model.addConstr(
            gp.quicksum(x[(u,v)] for v in all_nodes if (u,v) in E_star.keys()) == 1,
            name=f"arrive_once_{u}"
        )
    
    # Each customer has exactly 1 outgoing arc
    for u in customers:
        model.addConstr(
            gp.quicksum(x[(v,u)] for v in all_nodes if (v,u) in E_star.keys()) == 1,
            name=f"depart_once_{u}"
        )
    
    
    # constraint for linking routing and capacity
    for (v,u) in E_star.keys():
        if (u in customers):
            dv = node_data[v]["demand"]
            du = node_data[u]["demand"]
            model.addConstr(
                delta[v] - dv >= delta[u] - (d0 + dv)*(1 - x[(v,u)]),
                name=f"cap_{u}_{w}"
            )



    # constraint for linking routing and time
    for (v,u) in E_star.keys():
        if (u in customers):
            t_vu = travel_time_computer(v,u, node_data)
            # Big-M 
            M_vu = t_vu + node_data[u]["t_plus_budget"]
    
            model.addConstr(
                tau[v] - t_vu >= (tau[u]) - M_vu*(1 - x[(v,u)]),
                name=f"time_{v}_{u}"
            )


    
    
    # ## bound constraints for time remaining at each customer based on time window
    # for u in customers:
    
    #     model.addConstr(tau[u] <= node_data[u]["t_plus_budget"], name=f"tw_min_{u}")
    #     model.addConstr(tau[u] >= node_data[u]["t_minus_budget"], name=f"tw_max_{u}")
    
    
    # # bound constraint for capacity remaining at each customer
    # for u in customers:
    #     du = node_data[u]["demand"]
    #     model.addConstr(delta[u] >= du, name=f"delta_lower_{u}")
    
    
    # constraint for minimum number of vehicles
    total_demand = sum(node_data[u]["demand"] for u in customers)
    min_vehicles = math.ceil(total_demand / d0)
    
    model.addConstr(
        gp.quicksum(x[(alpha,w)] for w in customers if (alpha,w) in E_star.keys()) >= min_vehicles,
        name="min_vehicle_start"
    )
    

    
    # defining time limit for the solver
    model.Params.TimeLimit = 1000


    #solving the model

    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        solution_time = model.Runtime 
        print(f"Optimal solution found in {solution_time:.4f} seconds.")
        print(f"Optimal objective value: {model.objVal:.3f}")
        # Which arcs are used?
        used_arcs = [(u,w) for (u,w) in E_star.keys() if x[(u,w)].X > 0.5]
        print("Used arcs:")
        for (u,w) in used_arcs:
            print(f"  {u} -> {w}, cost={E_star[(u,w)]:.2f}")
    
    else:
        print(f"No optimal solution found. Gurobi status={model.status}")
    
    
        
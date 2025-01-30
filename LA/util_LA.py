import math
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from itertools import chain, combinations
from collections import defaultdict

def travel_time_computer(data, u, w):
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
                    data[u]["x"], data[u]["y"],
                    data[w]["x"], data[w]["y"]
                ) + data[u]["service"]
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
    Returns False if u -> v route is impossible due to capacity or time windows limitations;
    True otherwise.
    """

    # Checking capacity constraint
    if u_node["demand"] + w_node["demand"] > capacity:
        return False

    # checking time window constraints
    dist_alpha_u = euclidean_dist(alpha_node["x"], alpha_node["y"], u_node["x"], u_node["y"])
    arrival_u = alpha_node["early"] + alpha_node["service"] + dist_alpha_u
    start_u = max(u_node["early"], arrival_u)
    # checking if time window at node u is violated or not
    if start_u > u_node["late"]:
        return False
    finish_u = start_u + u_node["service"]

    # Arrive at w from u
    dist_u_w = euclidean_dist(u_node["x"], u_node["y"], w_node["x"], w_node["y"])
    arrival_w = finish_u + dist_u_w
    start_w = max(w_node["early"], arrival_w)
    # checking if time window at node w is violated or not
    if start_w > w_node["late"]:
        return False
    finish_w = start_w + w_node["service"]

    # Step C: Arrive at bar_alpha from w
    dist_w_bar = euclidean_dist(w_node["x"], w_node["y"], bar_alpha_node["x"], bar_alpha_node["y"])
    arrival_bar_alpha = finish_w + dist_w_bar
    # checking if time window at ending depot is violated or not
    if arrival_bar_alpha > bar_alpha_node["late"]:
        return False

    # If none of these checks fail => feasible
    return True

def generate_E_star(all_nodes, data, alpha, bar_alpha, d0):
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
                    data[alpha], data[u],
                    data[w], data[bar_alpha], d0
                )
                if feasible:
                    # If feasible, keep (u,w)
                    dist_uv = euclidean_dist(
                        data[u]["x"], data[u]["y"],
                        data[w]["x"], data[w]["y"]
                    )
                    E_star[(u,w)] = dist_uv
    return E_star

def create_buckets(lower_bound, increment, upper_bound):
    """
    Creates demand buckets for a customer, starting at their demand.

    Parameters:
    - lower_bound (int): The total demand of the customer (du[u]).
    - increment (int): The fixed increment for each bucket.
    - upper_bound (int): The overall capacity (upper limit for d_plus).


    Returns:
    - List[Tuple[int, int]]: A list of tuples representing the (d_minus, d_plus) of each bucket.
    """
    buckets = []
    
    current = lower_bound
    i = 1
    while current <= upper_bound:
        d_minus = current
        d_plus = current + i * increment - 1
        
        if d_plus > upper_bound:
            d_plus = upper_bound

        # Ensure that d_minus does not exceed d_plus
        if d_minus > d_plus:
            break

        buckets.append((d_minus, d_plus))
        current += i * increment
        i += 1
    return buckets


def merge_buckets(node, buckets, shadow_prices):
    """
    Merge buckets for a given node based on shadow prices.

    Parameters:
    - node: The node for which to merge buckets.
    - buckets: A list of intervals (buckets) for the node.
    - shadow_prices: A dictionary of shadow prices.

    Returns:
    - Updated list of buckets for the node.
    """
    if len(buckets) < 2:
        return buckets[:]  # Return a copy to avoid in-place issues

    i = 0  # Start from the first bucket

    while i < len(buckets) - 1:  # Stop before the last bucket
        current_bucket = buckets[i]
        next_bucket = buckets[i + 1]

        # Check if both keys exist and shadow prices are identical
        if (node, i) in shadow_prices and (node, i + 1) in shadow_prices:
            merge_condition = shadow_prices[(node, i)] == shadow_prices[(node, i + 1)]
        else:
            merge_condition = False

        if merge_condition:
            # Merge current and next buckets
            merged_bucket = (current_bucket[0], next_bucket[1])  # Merge intervals
            buckets[i] = merged_bucket  # Update current bucket
            buckets.pop(i + 1)  # Remove the next bucket after merging
        else:
            i += 1  # Move to the next bucket only if no merge occurred

    return buckets


def expand_capacity_buckets(Du, zD, du, d0, alpha, bar_alpha):
    """
    Expands capacity bucket ranges based on flow data.

    Parameters:
        Du (dict): A dictionary representing the capacity bucket ranges for each customer.
        zD (dict): A dictionary of flow decision variables for capacity.
        du (dict): A dictionary containing the demand values for each customer.
        d0 (float): The capacity of the depot (alpha).

    Returns:
        dict: Updated capacity buckets.
    """
    thresholds_D = {customer: [bucket[1] for bucket in buckets] for customer, buckets in Du.items()}
    # expanded_Du = Du.copy()
    expanded_Du = {}

    for (from_node, to_node), flow in zD.items():
        if flow > 0:
            from_customer, from_bucket = from_node
            to_customer, to_bucket = to_node

            if to_customer == bar_alpha or from_customer == to_customer:
                continue

            if from_customer == alpha:
                from_d_plus = d0
            else:
                if from_customer not in Du or from_bucket - 1 < 0 or from_bucket - 1 >= len(Du[from_customer]):
                    continue
                from_d_plus = Du[from_customer][from_bucket - 1][1]

            demand_u = du[from_customer] if from_customer != alpha else 0
            new_d_plus_j = from_d_plus - demand_u

            if new_d_plus_j not in thresholds_D[to_customer]:
                thresholds_D[to_customer].append(new_d_plus_j)
                thresholds_D[to_customer].sort()

    expanded_Du = {customer: [(buckets[0][0], thresholds_D[customer][0])] +
                              [(thresholds_D[customer][i-1] + 1, thresholds_D[customer][i]) 
                               for i in range(1, len(thresholds_D[customer]))]
                   for customer, buckets in Du.items()}

    return expanded_Du


def expand_time_buckets(Tu, zT, travel_time_computer, data, t0, alpha, bar_alpha):
    """
    Expands time bucket ranges based on flow data considering travel time between nodes.

    Parameters:
        Tu (dict): A dictionary representing the time bucket ranges for each customer.
        zT (dict): A dictionary of flow decision variables for time.
        travel_time_computer (function): Function to compute travel time between nodes.
        data (any): Data required by the travel_time_computer function.
        t0 (float): The time capacity of the depot (alpha).

    Returns:
        dict: Updated time buckets.
    """
    import copy
    import math
    thresholds_T = {customer: [bucket[1] for bucket in buckets] for customer, buckets in Tu.items()}
    # expanded_Tu = Tu.copy()
    # expanded_Tu = copy.deepcopy(Tu)
    expanded_Tu = {}
    for (from_node, to_node), flow in zT.items():
        if flow > 0:
            from_customer, from_bucket = from_node
            to_customer, to_bucket = to_node

            if to_customer == bar_alpha or from_customer == to_customer:
                continue

            if from_customer == alpha:
                from_t_plus = t0
                from_customer_id = 1
            else:
                if from_customer not in Tu.keys() or from_bucket - 1 < 0 or from_bucket - 1 >= len(Tu[from_customer]):
                    continue
                from_t_plus = Tu[from_customer][from_bucket - 1][1]
                from_customer_id = from_customer

            travel_time = math.floor(travel_time_computer(data, from_customer_id, to_customer))
            if to_bucket - 1 >= len(Tu[to_customer]) or to_bucket - 1 < 0:
                continue

            new_t_plus_j = min(from_t_plus - travel_time, Tu[to_customer][to_bucket - 1][1])

            if new_t_plus_j not in thresholds_T[to_customer]:
                thresholds_T[to_customer].append(new_t_plus_j)
                thresholds_T[to_customer].sort()

    expanded_Tu = {customer: [(buckets[0][0], thresholds_T[customer][0])] +
                              [(thresholds_T[customer][i-1] + 1, thresholds_T[customer][i]) 
                               for i in range(1, len(thresholds_T[customer]))]
                   for customer, buckets in Tu.items()}

    return expanded_Tu



def construct_capacity_graph(customers, Du, du, d0, alpha, bar_alpha):
    """
    Builds a directed graph (GD) and defines a binary parameter for allowed arcs based on rules.
    
    Parameters:
    - customers (list): A list of customers.
    - Du (dict): A dictionary containing buckets for each customer [(d_minus, d_plus), ...].
    - du (dict): A dictionary mapping each customer to their demand.
    - d0 (int/float): The total available capacity.
    
    Returns:
    - nodes (list): A list of all nodes in the graph GD.
    - arc_allowed (dict): A binary parameter for each arc (i, j), where 1 = allowed, 0 = disallowed.
    """
    nodes = []
    arc_allowed = {}  # Binary parameter for each arc (i, j)

    # Define starting and ending depot nodes
    alpha = (alpha, 1)            
    bar_alpha = (bar_alpha, 1)

    # Appending starting depot to node list. Based on the paper, d_minus = d_plus = d0
    nodes.append(alpha)
    
    # Appending ending depot to node list. Based on the paper, d_minus = 0 and d_plus = d0
    nodes.append(bar_alpha)

    # Add nodes for each customer and their demand buckets
    for u in customers:
        buckets = Du[u]
        for k, (d_minus, d_plus) in enumerate(buckets, start=1):
            node = (u, k)
            nodes.append(node)

    # Generate all possible arcs and determine if they are allowed
    for i in nodes:
        for j in nodes:
            if i == j:
                continue  # Skip self-loops
            
            # Default to disallowed
            arc_allowed[(i, j)] = 0
            
            # Define rules for allowed arcs
            if i == alpha:
                # Rule 1: From alpha to the last bucket of a customer
                if j[0] in customers and j[1] == len(Du[j[0]]):
                    arc_allowed[(i, j)] = 1
            elif j == bar_alpha:
                # Rule 2: From the first bucket of a customer to bar_alpha
                if i[0] in customers and i[1] == 1:
                    arc_allowed[(i, j)] = 1
            elif i[0] in customers and j[0] in customers and i[0] != j[0]:
                # Rule 3: Between buckets of different customers
                u, v = i[0], j[0]
                if u != v:  # Skip same customer
                    k = i[1]  # Current bucket of u
                    m = j[1]  # Current bucket of v

                    d_plus_i = Du[u][k - 1][1]  # Get d_plus for node i
                    d_minus_j = Du[v][m - 1][0]  # Get d_minus for node j

                    # If the arc is valid without considering prior buckets
                    if (d_plus_i - du[u]) >= d_minus_j:
                        if k > 1:
                            # Check capacity of the (k-1)-th bucket
                            d_u_k_minus_1_plus = Du[u][k - 2][1]  # d_plus of (k-1)-th bucket
                            if not (d_minus_j > (d_u_k_minus_1_plus - du[u])):
                                continue  # Skip if this condition is not satisfied

                        # Arc is allowed
                        arc_allowed[(i, j)] = 1
            elif i[0] in customers and i[1] > 1 and i[0] == j[0]:
                # Rule 4: From higher bucket k -> lower bucket (k-1) for the same customer
                if i[1] == j[1] + 1:
                    arc_allowed[(i, j)] = 1


    return nodes, arc_allowed




def construct_time_graph(data, customers, Tu, t0, alpha, bar_alpha):
    """
    Builds the directed graph (GT) using time-based remaining budget logic and defines a binary parameter.
    
    Parameters:
    - data: Data needed to calculate travel times between customers.
    - customers (list): A list of customer identifiers.
    - Tu (dict): A dictionary of time windows for each customer [(t_minus, t_plus), ...].
    - t0: Maximum amount of time (initial time budget).
    
    Returns:
    - nodes (list): A list of all nodes in the graph GT.
    - arc_allowed (dict): A binary parameter for each arc (i, j), where 1 = allowed, 0 = disallowed.
    - node_attributes (dict): A dictionary containing attributes for each node.
    """
    nodes = []
    arc_allowed = {}  # Binary parameter for each arc (i, j)
    node_attributes = {}

    # Define starting and ending depot nodes
    alpha = (alpha, 1)
    bar_alpha = (bar_alpha, 1)
    
    # Add starting depot
    nodes.append(alpha)
    node_attributes[alpha] = {'ui': alpha, 't_minus': t0, 't_plus': t0}

    # Add ending depot
    nodes.append(bar_alpha)
    node_attributes[bar_alpha] = {'ui': bar_alpha, 't_minus': 0, 't_plus': t0}

    # Add nodes for each customer and their time buckets
    for u in customers:
        buckets = Tu[u]
        for k, (t_minus, t_plus) in enumerate(buckets, start=1):
            node = (u, k)
            nodes.append(node)
            # node_attributes[node] = {'ui': u, 't_minus': t_minus, 't_plus': t_plus}

    # Generate all possible arcs and determine if they are allowed
    for i in nodes:
        for j in nodes:
            if i == j:
                continue  # Skip self-loops
            
            # Default to disallowed
            arc_allowed[(i, j)] = 0
            
            # Define rules for allowed arcs
            if i == alpha:
                # Rule 1: From alpha to the last bucket of a customer
                if j[0] in customers and j[1] == len(Tu[j[0]]):
                    arc_allowed[(i, j)] = 1
            elif j == bar_alpha:
                # Rule 2: From the first bucket of a customer to bar_alpha
                if i[0] in customers and i[1] == 1:
                    arc_allowed[(i, j)] = 1
            elif i[0] in customers and j[0] in customers:
                # Rule 3: Between buckets of different customers
                u, v = i[0], j[0]
                if u != v:  # Skip same customer
                    k = i[1]  # Bucket index for u
                    m = j[1]  # Bucket index for v

                    t_plus_i = Tu[u][k - 1][1]  # Get t_plus for node i
                    t_minus_j = Tu[v][m - 1][0]  # Get t_minus for node j


                    travel_time = travel_time_computer(data, u, v)

                    # Condition 1: t^+_u - travel_time >= t^-_v
                    if t_plus_i - travel_time >= t_minus_j:
                        
                        # # Arc is allowed
                        arc_allowed[(i, j)] = 1
            elif i[0] in customers and i[1] > 1 and i[0] == j[0]:
                # Rule 4: From higher bucket k -> lower bucket (k-1) for the same customer
                if i[1] == j[1] + 1:
                    arc_allowed[(i, j)] = 1

    return nodes, arc_allowed



def calculate_distance(item1, item2):
    """
    This function calculates euclidean distance
    item1: The tuple showing coordination of x and y for the first node
    item2: The tuple showing coordination of x and y for the second node
    """
    return round(math.sqrt((item1['x'] - item2['x'])**2 + (item1['y'] - item2['y'])**2),2)

# Creating neighborhoods
def create_neighborhoods(data, starting_depot, ending_depot, k, total_time, total_capacity):
    """
    The function aims to generate neighborhoods for each cusomter
    inputs:
    data: The node data
    starting_depot: starting depot
    ending depot: ending depot
    k: the number k-closest neighborhoods
    total_time: maximum amount of time of each vehicle
    total_capacity: maximum capacity of each vehicle

    """
    neighborhoods = {}
    start = 2
    end = len(data)-1
    # creating nodes for each customer
    for u in range(start, end + 1):
        neighbors = []
        distance_du = calculate_distance(data[1], data[u])  # distance between starting depot and u
        travel_time_du = distance_du  

        # checking feasiblity of starting time at depot
        arrival_time_at_u = max(data[u]["early"], data[1]["early"] + travel_time_du + data[1]["service"])
        if arrival_time_at_u > data[u]["late"]:
            continue  # infeasible
        
        for v in data:
            if u != v and v != 1 and v != len(data):  
                distance_uv = calculate_distance(data[u], data[v])  # Distance between u and v
                distance_vd = calculate_distance(data[v], data[len(data)])  # Distance from v to ending depot

                # time feasibility
                travel_time_uv = distance_uv  
                travel_time_vd = distance_vd  
                arrival_time_at_v = max(data[v]['early'], arrival_time_at_u + travel_time_uv + data[u]['service'])
                return_time_to_depot = arrival_time_at_v + data[v]['service'] + travel_time_vd
                
                if arrival_time_at_v > data[v]['late'] or return_time_to_depot > total_time:
                    continue  # Not feasible due to time window or depot return constraint

                # capacity feasibility
                remaining_capacity = total_capacity - data[u]['demand']
                if remaining_capacity < data[v]['demand']:
                    continue  # Not feasible due to insufficient capacity

                # adding node and distance
                neighbors.append((v, distance_uv))
        
        # sorting and recording the k-nearest neighborhoods
        neighbors_sorted = sorted(neighbors, key=lambda x: x[1])[:k]
        neighborhoods[u] = [v for v, _ in neighbors_sorted]

    return neighborhoods


from itertools import chain, combinations
def powerset(s):
    """
    generates all subsets of a given set s.
    inputs:
    s: set

    """
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def generate_P_plus(neighborhoods, node_data):
    """
    returns P_plus for all customers using their neighborhoods.
    inputs:
    neighborhoods: a dictionary containing customers as key and neighborhoods as value
    """
    all_nodes = [i for i in range(1, len(node_data)+1)]
    P_plus = []
    for u, neighbors in neighborhoods.items():
        # Compute the set of valid candidates for v
        candidates_for_v = set(all_nodes) - set(neighbors) - {u} - {1}
        for N_hat in powerset(neighbors):  # Generate all subsets of neighbors
            for v in candidates_for_v:  # Iterate over valid candidates for v
                if v not in N_hat:  # Ensure v is not in the subset
                    P_plus.append((u, frozenset(N_hat), v))  # Add arc to P_plus
    return P_plus


def time_windows(data, u):
    """
    returns time windows for each customer
    """
    return (data[u]["t_plus_budget"], data[u]["t_minus_budget"])


def base_case_feasilibtiy(data, u, v, vehicle_capacity):
    """
    checks if the route "starting_depot, u, v, ending_depot" is feasible 
    it returns true if the route is feasible
    """

    
    # check capacity
    total_demand = data[u]["demand"] + data[v]["demand"]
    if total_demand > vehicle_capacity:
        return False
    

    return True



def compute_c_and_phis(data, route):
    """
    This function calculates c_r, phi_r, and phihat_r for a given route recursively

    """
    route_key = tuple(route)

    # Base case: route with two customers => [u, v]
    if len(route) == 2:
        u, v = route
        t_plus_u, t_minus_u = time_windows(data, u)
        t_plus_v, t_minus_v = time_windows(data, v)

        # Compute base-case cost and phis
        c = travel_time_computer(data, u,v)
        phi = min(t_plus_u, t_plus_v + travel_time_computer(data, u,v))  # equation 13a
        phi_hat = max(t_minus_u, t_minus_v + travel_time_computer(data, u,v))  # equation 13b
        return c, phi, phi_hat
        
    
    u = route[0]
    w = route[1]
    r_minus = route[1:]
    t_plus_u, t_minus_u = time_windows(data, u)
    t_plus_w, t_minus_w = time_windows(data, w)

    # recursively calculating c_(r_minus), phi_(r_minus), phiHat_(r_minus)
    c_minus, phi_minus, phihat_minus = compute_c_and_phis(data, r_minus)
    
    phi   = min(t_plus_u, phi_minus + travel_time_computer(data, u,w))  # equation 13c
    phi_hat = max(t_minus_u, phihat_minus + travel_time_computer(data, u,w))  # equation 13d
    c     = c_minus + travel_time_computer(data, u,w)
    return c, phi, phi_hat


R_dict = defaultdict(list)  # maps p -> list of feasible routes (R_p)

def creating_efficient_frontier(data, P_plus):
    """
    this function obtains the efficient frontier for all p in P_plus.

    """
    # import pdb
    # pdb.set_trace()
    # adding feasible base cases to the efficient frontier
    for p in P_plus:
        u_p, N_p, v_p = p
        if u_p == 26:
            breakpoint()
        if len(N_p) == 0:
            r = [u_p, v_p]
            if not base_case_feasilibtiy(data, u_p, v_p, vehicle_capacity=200):
                continue
            else:
                R_dict[p].append(r)

    # sorting p based on |N_p| (|N_p|>0)
    sorted_p = sorted(
        [p for p in P_plus if len(p[1]) > 0],
        key=lambda x: len(x[1])
    )


    # getting efficient frontier for each p
    for p in sorted_p:
        if p not in R_dict:
            R_dict[p] = []
        u_p, N_p, v_p = p
        # R_dict[p] = []  # initialize empty
        t_plus = time_windows(data, u_p)[0]  # Extract time window for u_p
        t_minus = time_windows(data, u_p)[1]

        
        # lines 3-9 of pseudocode: build new routes from R_(p_hat)
        for w in N_p:

                    
            # p_hat = (w, N_p - {w}, v_p)
            # ensure we use a consistent structure (like tuple or frozenset)
            # N_p_minus_w = tuple(sorted(set(N_p) - {w}))
            N_p_minus_w = frozenset(N_p - {w})
            p_hat = (w, N_p_minus_w, v_p)

            # Predecessor routes
            r_minus_list = R_dict.get(p_hat, [])
            
            if not r_minus_list:
                continue  # infeasible
            
            for r_minus in r_minus_list:
                # build new route r by adding u_p
                r = [u_p] + r_minus
                
                # calculate (c_r, phi_r, phihat_r)
                c_r, phi_r, phihat_r = compute_c_and_phis(data, r)
                

                # checking feasiblity
                if phihat_r <= t_plus:
                    R_dict[p].append(r)
        
        # remove dominated routes
        to_remove = set()
        
        # Precompute route stats to avoid repeated recursion
        route_stats = {}
        for r in R_dict[p]:
            c_r, phi_r, phihat_r = compute_c_and_phis(data, r)
            route_stats[tuple(r)] = (c_r, phi_r, phihat_r)
        
        # Pdominancy check

        all_routes = R_dict[p]
        for i, r in enumerate(all_routes):
            if i in to_remove:
                continue
            c_r, phi_r, phihat_r = route_stats[tuple(r)]
            for j, r_hat in enumerate(all_routes):
                if j == i or j in to_remove:
                    continue
                c_rhat, phi_rhat, phihat_rhat = route_stats[tuple(r_hat)]
                
                condition1 = (c_r >= c_rhat)
                condition2 = (phihat_r >= phihat_rhat)
                condition3 = ((phi_r - c_r) <= (phi_rhat - c_rhat))
                any_strict = (
                    (c_r > c_rhat) or
                    (phihat_r > phihat_rhat) or
                    ((phi_r - c_r) < (phi_rhat - c_rhat))
                )
                
                if condition1 and condition2 and condition3 and any_strict:
                    to_remove.add(i)
                    break
        
        # Keep only non-dominated routes
        non_dominated = []
        for i, r in enumerate(R_dict[p]):
            if i not in to_remove:
                non_dominated.append(r)
        
        R_dict[p] = non_dominated

    return R_dict


def construct_R(u, R_dict, neighborhood_u, ending_depot, max_subset_size=6):
    """
    This function generates R by finding a match between nodes and R_dict.
    Max size of subsets is defined to manage computational expense.

    """
    from itertools import chain, combinations

    def powerset_limited(iterable, max_size):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(min(len(s), max_size) + 1))

    # Preprocess R_dict to remove entries with empty routes
    R_dict = {key: routes for key, routes in R_dict.items() if routes}

    R = set()
    neighborhood_with_depot = neighborhood_u + [ending_depot]

    for v_p in neighborhood_with_depot:
        for subset in powerset_limited(set(neighborhood_u) - {v_p}, max_size=max_subset_size):
            p = (u, frozenset(subset), v_p)

            if p in R_dict:
                for route in R_dict[p]:
                    R.add(tuple(route))

    R_clipped = {route[:-1] for route in R}

    return R_clipped



def compute_k(u, N_u, pi_uwk, pi_uwvk, E_star):
    """
    Computes k for a single customer u, given:
      - N_u[u]: the list (or set) of neighbors for u
      - pi_uwk[u, w, k]: dual variable 
      - pi_uwvk[u, w, v, k]: dual variable 
      - E_star: set of edges (w,v) in the augmented graph
    
    Returns an integer k_u (0 if none are positive).
    """
    # Let's sort neighbors by some criterion if needed, 
    # or assume N_u[u] is already in ascending distance order:
    neighbors = list(N_u)
    num_neighbors = len(neighbors)

    # We will track the largest k (1..num_neighbors) 
    # that yields a strictly positive sum of dual variables.
    k_u_value = 0
    
    # For each k in 1..|N_u[u]|, check the sum of the relevant duals
    for k in range(1, num_neighbors + 1):
        # Gather the partial sums:
        
        # 1) sum_{w in N_u} of pi_uwk
        sum_pi_uwk = 0.0
        for w in neighbors:
            # If (u,w,k) not in pi_uwk, default to 0
            sum_pi_uwk += pi_uwk.get((u, w, k), 0.0)
        
        # 2) sum_{w in N_u^{k+}, v in N_u^k \ {w}} of pi_uwvk
        #    "N_u^k" can be the first k neighbors, "N_u^{k+} = N_u^k U {u}"
        #    Typically w in that union, v in the first k neighbors minus w.
        
        # Let's define N_u^k as the first k neighbors in 'neighbors'.
        Nk = neighbors[:k]  # the k closest neighbors
        Nk_plus = set(Nk) | {u}
        
        sum_pi_uwvk = 0.0
        for w in Nk_plus:
            for v in Nk:
                if v != w and (w, v) in E_star.keys(): 
                    sum_pi_uwvk += pi_uwvk.get((u, w, v, k), 0.0)
        
        # total sum for this k
        total_dual_sum = sum_pi_uwk + sum_pi_uwvk
        
        # check if strictly positive
        if total_dual_sum > 0:
            k_u_value = k
        else:
            k_u_value = 1
    
    return k_u_value

    

def compute_a_k_wvr(u, routes_u, N_k):
    """
    Computes a_{w,v,r}^k for a single customer 'u', 
    given:
      - routes_u: the set (or list) of route tuples that visit 'u' (i.e., R[u])
      - neighbor_list_u: the sorted list of neighbors for 'u' (i.e., neighborhoods[u])
    
    Returns a dict: {(k, w, v, r_tuple): 0 or 1}
    """
    a_k_wvr = {}  # keyed by (k, w, v, route_tuple)
    
    # Helper to check partial route membership
    def partial_route_in_set(route_tuple, node_x, valid_set):
        if node_x not in route_tuple:
            return False
        idx = route_tuple.index(node_x)
        for nd in route_tuple[:idx+1]:
            if nd not in valid_set:
                return False
        return True
    

    # N_u^{k+} = N_k ∪ {u}
    N_k_plus = set(N_k) | {u}
    
    # For each route tuple in routes_u
    for route_tuple in routes_u:
        # For each w in N_k_plus and v in N_k
        for w in N_k_plus:
            for v in N_k:
                # Check partial route up to w in N_k_plus
                cond_w = partial_route_in_set(route_tuple, w, N_k_plus)
                # Check partial route up to v in N_k_plus
                cond_v = partial_route_in_set(route_tuple, v, N_k_plus)
                
                val = 1 if (cond_w and cond_v) else 0
                a_k_wvr[(w, v, route_tuple)] = val
    
    return a_k_wvr



def calculate_a_w_star_k(u, w, routes_for_u, N_k):
    """
    Computes a_{w*r}^k for a SINGLE (u, k, w) across all route tuples in routes_for_u.
    
    Each route in routes_for_u is a tuple, e.g. (2, 4, 10, 13).
    
    Returns a dict: route_tuple -> {0 or 1}, i.e. a_{w*r}^k for that route.

    Based on:
      a_{w*r}^k = 1 if 
         (partial route up to w is in N_u^{k+}) AND
         ( w is final in the route  OR  sum_{v not in (N_u^k - {w})} [w immediately precedes v] >= 1 )
      else 0.
    """

    # 1) Build the sets N_u^k and N_u^{k+}
    # neighbor_list_u is the sorted list of neighbors for u
    N_k_plus = set(N_k) | {u}
    excluded = set(N_k) - {w}         # we'll use this to check "v not in (N_u^k - {w})"

    # Helper: Check if partial route up to w is in N_k_plus
    def partial_route_in_set(route_tuple, w, valid_set):
        """
        Returns True if from the start of route_tuple up to (including) w,
        all visited nodes are in valid_set.
        """
        if w not in route_tuple:
            return False
        idx = route_tuple.index(w)
        for node in route_tuple[:idx+1]:
            if node not in valid_set:
                return False
        return True

    # Helper: Check if w is final in route_tuple
    def w_is_final(route_tuple, w):
        """
        Returns True if w is the last node in route_tuple.
        """
        if len(route_tuple) == 0:
            return False
        return (route_tuple[-1] == w)

    # Helper: a_{w,v,r} = 1 if w immediately precedes v in route_tuple
    def w_immediately_precedes_v(route_tuple, w, v):
        """
        Returns True if in route_tuple, w is directly followed by v.
        """
        if w not in route_tuple:
            return False
        idx = route_tuple.index(w)
        # w precedes v if v is at index idx+1
        if idx < len(route_tuple) - 1 and route_tuple[idx+1] == v:
            return True
        return False

    # 2) We'll store results in a dictionary: route_tuple -> 0/1
    a_w_star_k_for_routes = {}

    # 3) Iterate over each route (tuple) in routes_for_u
    for route_tuple in routes_for_u:
        # (a) Check if partial route up to w is valid
        a_wrk = 1 if partial_route_in_set(route_tuple, w, N_k_plus) else 0
        if a_wrk == 0:
            # If partial route up to w fails, then a_{w*r}^k=0
            a_w_star_k_for_routes[route_tuple] = 0
            continue

        # (b) Check if w is final
        if w_is_final(route_tuple, w):
            # Then condition is satisfied => a_{w*r}^k=1
            a_w_star_k_for_routes[route_tuple] = 1
            continue

        # (c) Otherwise, we compute sum_{v not in (N_u^k - {w})} a_{w,v,r}
        # That means sum over all v in route_tuple's successors except those in "excluded".
        # We'll check if at least one successor v is outside excluded.
        sum_outside = 0
        # Let's find all possible v that could follow w
        # A quick way is to check every node in route_tuple after w's index, but let's do:
        for v in route_tuple:
            if v not in excluded:  # i.e. v not in (N_k - {w})
                # check adjacency
                if w_immediately_precedes_v(route_tuple, w, v):
                    sum_outside += 1
        
        if sum_outside >= 1:
            a_w_star_k_for_routes[route_tuple] = 1
        else:
            a_w_star_k_for_routes[route_tuple] = 0

    # 4) Return the dictionary keyed by route_tuple
    return a_w_star_k_for_routes


    

def create_E_u_star_dict(N_u, u, E_star):
    """
    Create the edge set E_u_star for a given node u as a dictionary, based on E_star.

    Args:
        N_u: Reduced neighborhood for node u.
        u: The current node u.
        E_star: Dictionary of feasible edges (keys: tuples (w, v), values: edge attributes).

    Returns:
        E_u_star: A dictionary of edges for E_u_star.
    """
    N_plus_u = set(N_u) | {u}  # N^+_u = N_u ∪ {u}
    E_u_star = {}

    # Iterate through all keys (edges) in E^*
    for (w, v), attributes in E_star.items():
        if w in N_plus_u and v in N_u and v != w:
            E_u_star[(w, v)] = attributes

    return E_u_star


def calculate_a_wvr(R, N_u, u):
    """
    Calculate a_{w,v,r} for each ordering r in R, where w ∈ N_u^+ and v ∈ N_u - w.
    
    Args:
        R (set): Set of orderings for customer u.
        N_u (list): Reduced neighbors for u.
        u (int): The customer index.

    Returns:
        dict: a_wvr dictionary where keys are tuples (w, v, r), and values are 1 or 0.
    """
    a_wvr = {}
    N_u_plus = set(N_u) | {u}  # Include u itself in N_u^+

    for r in R:
        r_list = list(r)  # Convert ordering to a list for sequential checks
        for i in range(len(r_list) - 1):
            w = r_list[i]
            v = r_list[i + 1]
            if w in N_u_plus and v in set(N_u) - {w}:
                a_wvr[(w, v, r)] = 1
            else:
                a_wvr[(w, v, r)] = 0

    return a_wvr


def calculate_E_star_uw_dict(E_star, N_u, u):
    """
    Calculate E_star_uw.

    Args:
        E_star: Dictionary representing feasible edges (e.g., {(w, v): cost, ...}).
        N_u: Reduced neighbors for node u (e.g., {v1, v2, ...}).
        u: The node itself.

    Returns:
        E_star_uw: Dictionary where keys are (u, w) and values are sets of edges E^*_uw.
    """
    N_u_plus = set(N_u) | {u} # Include u itself in N_u^+
    N_u_to = {v for (_, v) in E_star.keys() if v not in N_u_plus|{1}}  # Next neighbors not in N_u^+
    
    E_star_uw = {}
    for w in N_u_plus:
        # Collect edges (w, v) where v ∈ N_u_to
        edges = {(w, v) for (w_, v) in E_star.keys() if w_ == w and v in N_u_to}
        E_star_uw[(u, w)] = edges

    return E_star_uw


def calculate_aw_star_r(R, N_u_plus, u):
    """
    Calculate aw_star_r for all r ∈ R and w ∈ N_u_plus.

    Args:
        R: Set of orderings (routes) for node u, where each ordering is a tuple of nodes.
        N_u_plus: Reduced neighbors including u itself (e.g., {u, n1, n2, ...}).
        u: The current node.

    Returns:
        aw_star_r: Dictionary where keys are (w, r) and values are 1 if w is the final customer in r, else 0.
    """
    aw_star_r = {}
    
    for r in R:
        if not r:  # Skip empty routes
            continue
        final_customer = r[-1]  # Last node in the ordering
        for w in N_u_plus:
            aw_star_r[(w, r)] = 1 if w == final_customer else 0

    return aw_star_r

def mip_solver(all_nodes, customers, customer_demands, node_data, E_star, t0, d0, alpha, bar_alpha, R, Du, Tu, N_u):
    
    # constructing capacity and time graph and obtaining nodes, edges and attributes of the graphs
    DG_nodes, DG_arc = construct_capacity_graph(customers, Du, customer_demands, d0, alpha, bar_alpha)
    TG_nodes, TG_arc = construct_time_graph(node_data, customers, Tu, t0, alpha, bar_alpha)


    # initialing the model
    model = gp.Model("VRPTW_LA_Dis_MIP")
        
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
    
    # Define capacity flow variables for each edge in DG
    z_D = {}
    for i in DG_nodes:
        for j in DG_nodes:
            var_name = f"zD_{i}_{j}"
            z_D[(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, name=var_name)

    
    # Define time flow variables for each edge in TG
    z_T = {}
    for i in TG_nodes:
        for j in TG_nodes:
            var_name = f"zT_{i}_{j}"
            z_T[(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, name=var_name)


    # Define binary decision to order r
    y = {}
    for u, routes in R.items():
        for r in routes:
            var_name = f"y_{'_'.join(map(str, r))}"
            y[r] = model.addVar(vtype=GRB.CONTINUOUS, name=var_name)
    # import pdb
    # pdb.set_trace()
    
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
            t_vu = travel_time_computer(node_data, v,u)
            # Big-M 
            M_vu = t_vu + node_data[u]["t_plus_budget"]
    
            model.addConstr(
                tau[v] - t_vu >= (tau[u]) - M_vu*(1 - x[(v,u)]),
                name=f"time_{v}_{u}"
            )

    
    
    
    # constraint for minimum number of vehicles
    total_demand = sum(node_data[u]["demand"] for u in customers)
    min_vehicles = math.ceil(total_demand / d0)
    
    model.addConstr(
        gp.quicksum(x[(alpha,w)] for w in customers if (alpha,w) in E_star.keys()) >= min_vehicles,
        name="min_vehicle_start"
    )
    

    
    for u in customers:
        if len(R[u]) == 0:
            continue
        model.addConstr(
            gp.quicksum(y[r] for r in R[u]) == 1,
            name=f"order_selection_{u}"
        )
    
    
    for u in customers:
        # N_u = get_reduced_neighbors(R[u], neighborhoods[u], u)
        # N_u = set(neighborhoods[u])
        E_u_star = create_E_u_star_dict(N_u[u], u, E_star) ##
        a_wvr = calculate_a_wvr(R[u], N_u[u], u)##
        for (w,v) in E_u_star.keys():  # Iterate over feasible edges for u
                lhs = x[(w, v)]  # Left-hand side: x[w, v]
                rhs = gp.quicksum(a_wvr.get((w, v, r), 0) * y[r] for r in R[u])  # Right-hand side: sum(a_wvr * y[r])
                model.addConstr(lhs >= rhs, name=f"edge_order_consistency_u{u}_w{w}_v{v}")
    
    
    for u in customers:  # Iterate over all customers
        # N_u = get_reduced_neighbors(R[u], neighborhoods[u], u)
        # N_u = set(neighborhoods[u])  # Use the original neighborhood set
        N_u_plus = set(N_u[u]) | {u}  ##
        aw_star_r = calculate_aw_star_r(R[u], N_u_plus, u)  # Precompute a_w*r values
        E_star_uw = calculate_E_star_uw_dict(E_star, N_u[u], u)  ##
    
        for w in N_u_plus:  # Iterate over each w in N+_u
            if (u,w) in E_star_uw.keys():  # Ensure we have edges for w
                edges = E_star_uw[(u,w)]  # Get edges (w, v) for the current w
    
                # Define LHS: Sum of x[w, v] for all (w, v) in edges
                lhs = gp.quicksum(x[edge] for edge in edges)  # Edge format is (w, v)
    
                # Define RHS: Sum of a_w*r * y[r] for all r in R
                rhs = gp.quicksum(aw_star_r.get((w, r), 0) * y[r] for r in R[u])
    
                # Add the constraint
                model.addConstr(lhs >= rhs, name=f"final_order_consistency_u{u}_w{w}")
    


    
    for i in DG_nodes:
        if i[0] not in [alpha, bar_alpha]:
            # Sum of zD over outgoing edges equals sum of zD over incoming edges
            model.addConstr(
                gp.quicksum(DG_arc.get((i,j),0) * z_D[(i, j)] for j in DG_nodes) == gp.quicksum(DG_arc.get((j,i),0) * z_D[(j, i)] for j in DG_nodes if j != i),
                name=f"cap_flow_balance{i}"
            )

    for (u,w) in E_star.keys():
        # relevant_zD = [DG_arc.get((i,j),0) * z_D[(i, j)] for i in DG_nodes for j in DG_nodes if i[0] == u and j[0] == w]
        model.addConstr(
                x[(u, w)] == gp.quicksum(DG_arc.get((i,j),0) * z_D[(i, j)] for i in DG_nodes for j in DG_nodes if i[0] == u and j[0] == w),
                name=f"sum_zD_geq_x_{u}_{w}"
            )
    
    for i in TG_nodes:
        if i[0] not in [alpha, bar_alpha]:
            # Sum of zD over outgoing edges equals sum of zD over incoming edges
            model.addConstr(
                gp.quicksum(TG_arc.get((i,j),0) * z_T[(i, j)] for j in TG_nodes) == gp.quicksum(TG_arc.get((j,i),0) * z_T[(j, i)] for j in TG_nodes if j != i),
                name=f"time_flow_balance{i}"
            )
    
    
    for (u,w) in E_star.keys():
        # relevant_zT = [TG_arc.get((i,j),0) * z_T[(i, j)] for j in TG_nodes for i in TG_nodes if i[0] == u and j[0]==w]
        model.addConstr(
                gp.quicksum(TG_arc.get((i,j),0) * z_T[(i, j)] for i in TG_nodes for j in TG_nodes if i[0] == u and j[0]==w) == x[(u, w)],
                name="sum_zD_geq_x"
            )
     
     
    # defining time limit for the solver
    model.Params.TimeLimit = 1200


    #solving the model
    
    model.optimize()
    # import pdb
    # pdb.set_trace()
    
    if model.status == GRB.OPTIMAL:
        solution_time = model.Runtime 
        print(f"Optimal solution found in {solution_time:.4f} seconds.")
        print(f"Optimal objective value: {model.objVal:.3f}")
        used_arcs = [(u,w) for (u,w) in E_star.keys() if x[(u,w)].X > 0.5]
        print("Used arcs:")
        for (u,w) in used_arcs:
            print(f"  {u} -> {w}, cost={E_star[(u,w)]:.2f}")
    
    else:
        print(f"No optimal solution found. Gurobi status={model.status}")


def lp_relaxation_solver(all_nodes, customers, customer_demands, node_data, E_star, t0, d0, alpha, bar_alpha, R, Du, Tu, N_u):
    
    # constructing capacity and time graph and obtaining nodes, edges and attributes of the graphs
    DG_nodes, DG_arc = construct_capacity_graph(customers, Du, customer_demands, d0, alpha, bar_alpha)
    TG_nodes, TG_arc = construct_time_graph(node_data, customers, Tu, t0, alpha, bar_alpha)
    epsilon = 0.01
    # breakpoint()
    # initialing the model
    model = gp.Model("VRPTW_LA_Dis_LP")
    
    
    # Defining x which are routing decisions
    x = {}
    for (u,w) in E_star.keys():
        x[(u,w)] = model.addVar(vtype=GRB.CONTINUOUS, name=f"x_{u}_{w}")
    # defining tau_u decision variables
    tau = {}
    for u in all_nodes:
        tau[u] = model.addVar(lb=node_data[u]["t_minus_budget"], ub=node_data[u]["t_plus_budget"], vtype=GRB.CONTINUOUS, name=f"tau_{u}")
    
    # defining delta_u decision variables
    delta = {}
    for u in all_nodes:
        delta[u] = model.addVar(lb=node_data[u]['demand'], ub=d0, vtype=GRB.CONTINUOUS, name=f"delta_{u}")
    
    # Define capacity flow variables for each edge in DG
    z_D = {}
    for i in DG_nodes:
        for j in DG_nodes:
            var_name = f"zD_{i}_{j}"
            z_D[(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, name=var_name)

    
    # Define time flow variables for each edge in TG
    z_T = {}
    for i in TG_nodes:
        for j in TG_nodes:
            var_name = f"zT_{i}_{j}"
            z_T[(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, name=var_name)

    y = {}
    for u, routes in R.items():
        for r in routes:
            var_name = f"y_{'_'.join(map(str, r))}"
            y[r] = model.addVar(vtype=GRB.CONTINUOUS, name=var_name)
    
    
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
            t_vu = travel_time_computer(node_data, v,u)
            # Big-M 
            M_vu = t_vu + node_data[u]["t_plus_budget"]
    
            model.addConstr(
                tau[v] - t_vu >= (tau[u]) - M_vu*(1 - x[(v,u)]),
                name=f"time_{v}_{u}"
            )



    
    # constraint for minimum number of vehicles
    total_demand = sum(node_data[u]["demand"] for u in customers)
    min_vehicles = math.ceil(total_demand / d0)
    
    model.addConstr(
        gp.quicksum(x[(alpha,w)] for w in customers if (alpha,w) in E_star.keys()) >= min_vehicles,
        name="min_vehicle_start"
    )
    
    for u in customers:
        if len(R[u]) == 0:
            continue
        model.addConstr(
            gp.quicksum(y[r] for r in R[u]) == 1,
            name=f"order_selection_{u}"
        )
    


    for u in customers:

        # E_u_star = create_E_u_star_dict(N_u[u], u, E_star) ##        
        num_neighbors = len(N_u[u])
        
        for k in range(1, num_neighbors + 1):
            N_k = N_u[u][:k]  # first k neighbors
            # a_k_wvr = calculate_a_kwvr(R[u], N_k, u, k)
            a_k_wvr = compute_a_k_wvr(u, R[u], N_k)
            # 5) For each feasible edge (w, v) in E_u_star, add constraint:
            for (w, v) in E_star.keys():
                # LHS = epsilon * k + x[w,v]
                lhs = epsilon * k + x[(w, v)]                
                rhs = gp.quicksum(a_k_wvr.get((w, v, r), 0.0) * y[r] for r in R[u])                
                
                model.addConstr(
                    lhs >= rhs,
                    name=f"8a_penalty_u{u}_k{k}_w{w}_v{v}"
                )



    for u in customers:
        num_neighbors = len(N_u[u]) ##
        
        for k in range(1, num_neighbors + 1):
            N_k = N_u[u][:k]
            N_k_plus = set(N_k) | {u}
            for w in N_k_plus:

                outside_vs = []
                for (ww, vv) in E_star.keys():    ##
                    if ww == w and vv not in N_k_plus:
                        outside_vs.append(vv)
                
                lhs = epsilon * k + gp.quicksum(x[(w, v_)] for v_ in outside_vs)

                a_w_star_k = calculate_a_w_star_k(u, w, R[u], N_k)

                rhs = gp.quicksum(a_w_star_k.get(r, 0) * y[r] 
                                  for r in R[u])

                # 5) Add the constraint:
                model.addConstr(
                    lhs >= rhs,
                    name=f"8b_penalty_u{u}_k{k}_w{w}"
                )

    for i in DG_nodes:
        if i[0] not in [alpha, bar_alpha]:
            # Sum of zD over outgoing edges equals sum of zD over incoming edges
            model.addConstr(
                gp.quicksum(DG_arc.get((i,j),0) * z_D[(i, j)] for j in DG_nodes) == gp.quicksum(DG_arc.get((j,i),0) * z_D[(j, i)] for j in DG_nodes if j != i),
                name=f"cap_flow_balance{i}"
            )

    for (u,w) in E_star.keys():
        model.addConstr(
                x[(u, w)] == gp.quicksum(DG_arc.get((i,j),0) * z_D[(i, j)] for i in DG_nodes for j in DG_nodes if i[0] == u and j[0] == w),
                name=f"sum_zD_geq_x_{u}_{w}"
            )
    
    for i in TG_nodes:
        if i[0] not in [alpha, bar_alpha]:
            # Sum of zD over outgoing edges equals sum of zD over incoming edges
            model.addConstr(
                gp.quicksum(TG_arc.get((i,j),0) * z_T[(i, j)] for j in TG_nodes) == gp.quicksum(TG_arc.get((j,i),0) * z_T[(j, i)] for j in TG_nodes if j != i),
                name=f"time_flow_balance{i}"
            )
    
    
    for (u,w) in E_star.keys():
        model.addConstr(
                gp.quicksum(TG_arc.get((i,j),0) * z_T[(i, j)] for j in TG_nodes for i in TG_nodes if i[0] == u and j[0]==w) == x[(u, w)],
                name="sum_zD_geq_x"
            )
     
    # defining time limit for the solver
    model.setParam("OutputFlag", 0)
    model.Params.TimeLimit = 1000
    model.setParam('IntFeasTol', 1e-9)
    model.setParam('FeasibilityTol', 1e-9)
    model.setParam('OptimalityTol', 1e-9)

    #solving the model
    
    model.optimize()

    if model.status == GRB.OPTIMAL:
        objective_value = model.objVal
        capacity_flow_balance_duals = {}
        time_flow_balance_duals = {}
        pi_uwvk = {}
        pi_uwk = {}
        z_D_values = {}
        z_T_values = {}
        
        for i in DG_nodes:
            if i[0] not in [alpha, bar_alpha]:
                constr_name = f"cap_flow_balance{i}"
                constr1 = model.getConstrByName(constr_name)
                if constr1:
                    capacity_flow_balance_duals[i] = constr1.Pi  # Dual value (shadow price)

        for i in TG_nodes:
            if i[0] not in [alpha, bar_alpha]:
                constr_name = f"time_flow_balance{i}"
                constr1 = model.getConstrByName(constr_name)
                if constr1:
                    capacity_flow_balance_duals[i] = constr1.Pi  # Dual value (shadow price)
 

            
 
        for u in customers:
            # E_u_star = create_E_u_star_dict(N_u[u], u, E_star) ###
            num_neighbors = len(N_u)
            
            for k in range(1, num_neighbors + 1):
                N_k = N_u[u][:k]  # first k neighbors
                for (w, v) in E_star.keys():
                    constr_name = f"8a_penalty_u{u}_k{k}_w{w}_v{v}"
                    constr3 = model.getConstrByName(constr_name)
                    
                    if constr3:
                        pi_uwvk[u,w,v,k] = constr3.Pi 

        for u in customers:
            num_neighbors = len(N_u[u])  ###
            
            for k in range(1, num_neighbors + 1):
                N_k = N_u[u][:k]
                N_k_plus = set(N_k) | {u}
                
                for w in N_k_plus:
                    # Generate the constraint name
                    constr_name = f"8b_penalty_u{u}_k{k}_w{w}"
                    constr4 = model.getConstrByName(constr_name)
                    
                    # Store the dual value if the constraint exists
                    if constr4:
                        pi_uwk[(u, w, k)] = constr4.Pi


        for (i, j), var in z_D.items():
            # Store the value of the variable in the dictionary
            z_D_values[(i, j)] = var.X


        for (i, j), var in z_T.items():
            # Store the value of the variable in the dictionary
            z_T_values[(i, j)] = var.X

    return capacity_flow_balance_duals, time_flow_balance_duals, pi_uwvk, pi_uwk, z_D_values, z_T_values, objective_value


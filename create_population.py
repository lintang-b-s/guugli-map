import os
import math
import multiprocessing
import random
import pandas as pd
import sys
import concurrent.futures
import threading

experiments = 30

population_size = 300
random_population = 0.9
greedy_population = 0.1 
alpha = 100
beta = 0.001



customers_file = "./solomon_benchmark/R1/R101.csv"
fleets_file = "./solomon_benchmark/R1/R101_fleets.csv"

df_customers = pd.read_csv(customers_file)
df_fleets = pd.read_csv(fleets_file)
num_customers = df_customers.shape[0]-1
num_customers_with_depot = num_customers + 1

# create distance matrix
fleets_depot_coord = {'XCOORD.': df_fleets.loc[0, "fleet_start_x_coord"], 'YCOORD.': df_fleets.loc[0, "fleet_start_y_coord"]}
distance_matrix = []

def euclidean_distance(p1, p2):
    return math.sqrt((p1['XCOORD.'] - p2['XCOORD.'])**2 + (p1['YCOORD.'] - p2['YCOORD.'])**2)



distance_matrix = {}
for i in range(0, num_customers_with_depot):
    for j in range(0, num_customers_with_depot):
     
        left, right = df_customers.loc[i], df_customers.loc[j]

        if i not in distance_matrix:
            distance_matrix[i] = {}
            
        distance_matrix[i][j] =  euclidean_distance(left, right)
        

random_chromosome_num = math.floor(population_size*random_population)
greedy_chromosome_num = population_size - random_chromosome_num

max_dist = -1
min_dist = 999999

for i in range(0, num_customers):
    for j in range(0, num_customers):
        if i != j:
            max_dist = max(max_dist, distance_matrix[i][j])
            min_dist = min(min_dist, distance_matrix[i][j])

route_radius =  (max_dist - min_dist)/2

def allowed_neigbors_search(customer_size, customers, distance_matrix, available_customers, 
                     fleet_total_time, fllet_capacity, curr_time, curr_capacity,
                     curr_customer):
    # customers -> [0, 1, 2, ... 24]
    allowed_neigbors = [None] * len(available_customers) # -> 0, 1, 2, ... , 24 (customer)
    # available_customers -> 1, 2, 3, ..., 25

    available_customers_index = [i-1 for i in available_customers]
    for i in range(len(available_customers_index)):
        cust_index = available_customers_index[i]
        customer = customers[cust_index]
        allowed_neigbors[i] = {'id': customer['id'], 'demand': customer['demand'],  
                      'service_time': customer['service_time'] ,
                      'ready_time': customer['ready_time'],
                      'due_time': customer['due_time'],
                       'complete_time': customer['complete_time'] ,}

        allowed_neigbors[i]['demand'] += curr_capacity

        allowed_neigbors[i]['distance_to_curr_customer'] = distance_matrix[curr_customer][cust_index+1]

        allowed_neigbors[i]['arrival_time'] = allowed_neigbors[i]['distance_to_curr_customer'] + curr_time

        allowed_neigbors[i]['waiting_time'] = allowed_neigbors[i]['ready_time'] -  allowed_neigbors[i]['arrival_time']
        allowed_neigbors[i]['waiting_time'] = max(0,  allowed_neigbors[i]['waiting_time'])

        allowed_neigbors[i]['start_time'] = allowed_neigbors[i]['arrival_time'] + allowed_neigbors[i]['waiting_time']  

        allowed_neigbors[i]['finish_time'] = allowed_neigbors[i]['start_time'] + allowed_neigbors[i]['service_time']

        allowed_neigbors[i]['return_time'] = distance_matrix[cust_index+1][0] + allowed_neigbors[i]['finish_time']

    

    for i in range(len(allowed_neigbors)-1, -1, -1):
        neighbor = allowed_neigbors[i]
        if neighbor['demand'] > fleet_capacity or neighbor['return_time'] > fleet_total_time or allowed_neigbors[i]['start_time'] < allowed_neigbors[i]['ready_time'] or  allowed_neigbors[i]['finish_time'] > allowed_neigbors[i]['complete_time']:
                allowed_neigbors.pop(i)

    return allowed_neigbors
    


def create_random_chromosome(customers_size, customers, distance_matrix, fleet_capacity, 
                             fleet_total_time, customers_index):
    
    available_customers = customers_index.copy()    

    chromosome_routes = []
    chromosome_distances = []
    curr_route = [0] # start dari city-0
    curr_capacity = 0
    curr_time = 0
    curr_customer = 0
    curr_distance = 0

    while len(available_customers) != 0:
        allowed_neighbors = allowed_neigbors_search(customers_size, customers, distance_matrix, available_customers, 
                     fleet_total_time, fleet_capacity, curr_time, curr_capacity,
                     curr_customer)
        if len(allowed_neighbors) != 0:
            allowed_neighbors_id = []
            for i in range(len(allowed_neighbors)):
                allowed_neighbors_id.append(allowed_neighbors[i]['id'])
            curr_customer = int(random.choice(allowed_neighbors_id))
            curr_customer_idx = allowed_neighbors_id.index(curr_customer)

            curr_route.append(curr_customer)
            curr_distance += allowed_neighbors[curr_customer_idx]['distance_to_curr_customer']
            curr_capacity = allowed_neighbors[curr_customer_idx]['demand']
            curr_time = allowed_neighbors[curr_customer_idx]['finish_time']
            available_customers.remove(curr_customer)
        else:
            # rute baru
            curr_route.append(0)      
            chromosome_routes.append(curr_route)
            curr_distance += distance_matrix[curr_customer][0]
            chromosome_distances.append(curr_distance)

            curr_route = [0]      
            curr_capacity = 0
            curr_time = 0
            curr_customer = 0
            curr_distance = 0
        
    curr_route.append(0)
    chromosome_routes.append(curr_route)
    curr_distance += distance_matrix[curr_customer][0]
    chromosome_distances.append(curr_distance)

    total_distance = sum(chromosome_distances)

    return chromosome_routes, chromosome_distances, len(chromosome_routes), total_distance


def create_greedy_chromosome(customers_size, customers, distance_matrix, fleet_capacity, 
                             fleet_total_time, customers_index, route_radius):
    
    available_customers = customers_index.copy() # 1,2,3,4, ..., 25   

    chromosome_routes = []
    chromosome_distances = []
    curr_route = [0] # start dari city-0
    curr_capacity = 0
    curr_time = 0
    curr_customer = 0
    curr_distance = 0

    prev_random_choice = True

    while len(available_customers) != 0:
        allowed_neighbors = allowed_neigbors_search(customers_size, customers, distance_matrix, available_customers, 
                     fleet_total_time, fleet_capacity, curr_time, curr_capacity,
                     curr_customer)
        
        allowed_neighbors_id = []
        for i in range(len(allowed_neighbors)):
            allowed_neighbors_id.append(allowed_neighbors[i]['id'])


        
        if len(allowed_neighbors) != 0 and prev_random_choice == True:

            curr_customer = int(random.choice(allowed_neighbors_id))
            curr_customer_idx = allowed_neighbors_id.index(curr_customer)

           

            curr_route.append(curr_customer) # add ci to l
            curr_distance += allowed_neighbors[curr_customer_idx]['distance_to_curr_customer']
            curr_capacity = allowed_neighbors[curr_customer_idx]['demand']
            curr_time = allowed_neighbors[curr_customer_idx]['finish_time']
            available_customers.remove(curr_customer)# remove ci
            min_distance = 999999999


            allowed_neighbors = allowed_neigbors_search(customers_size, customers, distance_matrix, available_customers, 
                    fleet_total_time, fleet_capacity, curr_time, curr_capacity,
                    curr_customer)
        
            allowed_neighbors_id = []
            for i in range(len(allowed_neighbors)):
                allowed_neighbors_id.append(allowed_neighbors[i]['id'])


            nearest_customer = None
            nearest_customer_idx = None
            for index in range(len(allowed_neighbors)): # 0, 1, 2, 3, 4, ..., 24
                if allowed_neighbors[index]['distance_to_curr_customer'] < min_distance:
                    min_distance = allowed_neighbors[index]['distance_to_curr_customer']
                    nearest_customer = allowed_neighbors[index]['id']
                    nearest_customer_idx =   index


            exist = False
            if nearest_customer is not None:
                for route in chromosome_routes:
                    if nearest_customer in route:
                        exist = True
            else:
                exist = True

            if exist == False:
                curr_customer = nearest_customer
                curr_customer_idx = nearest_customer_idx
                curr_route.append(curr_customer)
                curr_distance += allowed_neighbors[curr_customer_idx]['distance_to_curr_customer']
                curr_capacity = allowed_neighbors[curr_customer_idx]['demand']
                curr_time = allowed_neighbors[curr_customer_idx]['finish_time']
                available_customers.remove(curr_customer)
                prev_random_choice = False
            else:
                prev_random_choice = True 
                continue

            
        elif len(allowed_neighbors) != 0 and prev_random_choice == False:
          

            nearest_customer = None
            nearest_customer_idx = None
            min_distance = 999999999

            for index in range(len(allowed_neighbors)):# 0, 1, 2, 3, 4, ..., 24
                if allowed_neighbors[index]['distance_to_curr_customer'] < min_distance:
                    min_distance = allowed_neighbors[index]['distance_to_curr_customer']
                    nearest_customer = allowed_neighbors[index]['id']
                    nearest_customer_idx = index
            
            exist = False
            if nearest_customer is not None:
                for route in chromosome_routes:
                    if nearest_customer in route:
                        exist = True
            else:
                exist = True

            if exist == False:
                curr_customer = nearest_customer
                curr_customer_idx = nearest_customer_idx
                curr_route.append(curr_customer)
                curr_distance += allowed_neighbors[curr_customer_idx]['distance_to_curr_customer']
                curr_capacity = allowed_neighbors[curr_customer_idx]['demand']
                curr_time = allowed_neighbors[curr_customer_idx]['finish_time']
                available_customers.remove(curr_customer)
                prev_random_choice = False
            else:
                prev_random_choice = True 
                continue
            
            
        else:
            # rute baru
            curr_route.append(0)      
            chromosome_routes.append(curr_route)
            curr_distance += distance_matrix[curr_customer][0]
            chromosome_distances.append(curr_distance)

            curr_route = [0]      
            curr_capacity = 0
            curr_time = 0
            curr_customer = 0
            curr_distance = 0
        
    curr_route.append(0)
    chromosome_routes.append(curr_route)
    curr_distance += distance_matrix[curr_customer][0]
    chromosome_distances.append(curr_distance)

    total_distance = sum(chromosome_distances)

    return chromosome_routes, chromosome_distances, len(chromosome_routes), total_distance



def initial_population(population_size, random_chromosome_num,  greedy_chromosome_num,
                          customers_size, customers, distance_matrix, fleet_capacity, fleet_total_time,
                          customers_index, route_radius):
    lock = threading.Lock()
   
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for _ in range(random_chromosome_num):
            futures.append(executor.submit(create_random_chromosome, customers_size, customers, distance_matrix, fleet_capacity, fleet_total_time, customers_index, ))
        for _ in range(greedy_chromosome_num):
            futures.append(executor.submit(create_greedy_chromosome, customers_size, customers, distance_matrix, fleet_capacity, fleet_total_time, customers_index, route_radius, ))
         
    for future in futures:
        with lock:
            results.append(future.result())
    

    population_routes = [None] * population_size
    population_distances = [None] * population_size
    population_num_routes =  [None] * population_size
    population_total_route_distances = [None]* population_size

   
    for i in range(len(results)):
        res = results[i]
        population_routes[i] = res[0]
        population_distances[i] = res[1]
        population_num_routes[i] = res[2] 
        population_total_route_distances[i] = res[3]
        
    return population_routes, population_distances, population_num_routes, population_total_route_distances

customers = []

def is_route_valid(route, distance_matrix, customers, fleet_total_time, fleet_capacity):
    route_valid = True

    city_one_idx = route[1]-1
    curr_demand = customers[city_one_idx]['demand']
    ready_time = customers[city_one_idx]['ready_time']
    due_time = customers[city_one_idx]['due_time']
    curr_time = distance_matrix[route[0]][route[1]]
    curr_time_dist = distance_matrix[route[0]][route[1]]

    for i in range(len(route)-3): # route exclude depot(0), exclude last city sebelum depot
        curr_time += distance_matrix[route[i]][route[i+1]]
        curr_time_dist += distance_matrix[route[i]][route[i+1]]
        next_city_idx = route[i+1]-1
        ready_time = customers[next_city_idx]['ready_time']
        due_time = customers[next_city_idx]['due_time']
        service_time = customers[next_city_idx]['service_time']
        curr_demand += customers[next_city_idx]['demand']
        if curr_time > due_time or curr_demand > fleet_capacity:
            route_valid = False
            break
        else:
            wait_time = max(0, ready_time-curr_time)
            if curr_time + wait_time < ready_time or curr_time + wait_time + service_time > fleet_total_time:
                route_valid = False
                break
            else:
                curr_time += wait_time + service_time
        
    if route_valid == True:
        curr_time += distance_matrix[route[-2]][route[-1]] # from last city ke depot
        curr_time_dist += distance_matrix[route[-2]][route[-1]]
        if curr_time > due_time:
            route_valid = False
    
    return route_valid, curr_time_dist


def fitness_function_weighted_sum(chromosome_distances, chromosome_routes ):
    total_distance = sum(chromosome_distances)
    num_vehicles = len(chromosome_routes)
    fitness = (alpha*num_vehicles) + (beta*total_distance) #  fitness untuk chromosome saat ini
    return fitness

def phase_two(alpha, beta, chromosome_routes, chromosome_distances, distance_matrix, customers, fleet_total_time, fleet_capacity):
    """
    In Phase 2, the last customer of each route ri , is relocated
    to become the first customer to route ri+1 . If this removal
    and insertion maintains feasibility for route ri+1, and the
    sum of costs of r1 and ri+1 at Phase 2 is less than sum of
    costs of ri + ri+1 at Phase 1, the routing configuration at
    Phase 2 is accepted, otherwise the network topology before
    Phase 2 (i.e., at Phase 1) is maintained.
    """
    indexes = [i for i in range(len(chromosome_routes)+1)]
    indexes[len(chromosome_routes)] = 0 # buat masangin route last vehicle dg route first vehicle

    for i in range(len(indexes)-1):
        route_one_idx = indexes[i]
        route_two_idx = indexes[i+1]
        route_one = chromosome_routes[route_one_idx].copy()
        route_two = chromosome_routes[route_two_idx].copy()
        distance_route_one = chromosome_distances[route_one_idx]
        distance_route_two = chromosome_distances[route_two_idx]
        last_customer_route_one = route_one[-2]
        route_two.insert(1, last_customer_route_one)
        is_route_two_valid, new_route_two_dist = is_route_valid(route_two, distance_matrix, customers, fleet_total_time, fleet_capacity)
        if is_route_two_valid == True:
            if len(route_one) > 3:
                new_route_one_dist =  distance_route_one - distance_matrix[route_one[-2]][0] + distance_matrix[route_one[-3]][0]
                if (new_route_one_dist + new_route_two_dist) < (distance_route_one + distance_route_two):
                    del route_one[-2]
                    chromosome_routes[route_one_idx] = route_one
                    chromosome_routes[route_two_idx] = route_two
                    chromosome_distances[route_one_idx] = new_route_one_dist
                    chromosome_distances[route_two_idx] = new_route_two_dist
            else:
                del chromosome_routes[route_one_idx]
                chromosome_routes[route_two_idx] = route_two
                del chromosome_distances[route_one_idx]
                chromosome_distances[route_two_idx] = new_route_two_dist

    total_distance = sum(chromosome_distances)
    num_vehicles = len(chromosome_routes)
    fitness = fitness_function_weighted_sum(chromosome_distances, chromosome_routes)
    return chromosome_routes, chromosome_distances, num_vehicles, total_distance, fitness

        

def fast_non_dominated_sort_fitness(values1, values2):
    S = [[] for _ in range(len(values1))]
    front = [[]]
    n = [0 for _ in range(len(values1))]
    rank = [0 for _ in range(len(values1))]

    for p in range(len(values1)):
        S[p] = [] # S[p] isinya semua solusi yang didominasi/lebih buruk oleh solusi p 
        n[p] = 0 # jumllah solusi lain yang mendominasi / lebih baik dari solusi p 
        for q in range(len(values1)):
            if (values1[p] < values1[q] and values2[p] < values2[q]) or \
               (values1[p] <= values1[q] and values2[p] < values2[q]) or \
               (values1[p] < values1[q] and values2[p] <= values2[q]):
                S[p].append(q) # p dominates q
            elif (values1[q] < values1[p] and values2[q] < values2[p]) or \
                 (values1[q] <= values1[p] and values2[q] < values2[p]) or \
                 (values1[q] < values1[p] and values2[q] <= values2[p]):
                n[p] += 1 #  ada solusi lain yang mendominasi solusi p, solusi lain yang mendominasi p ada n[p] buah
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                # q  lebih buruk dari p
                n[q] -= 1 # n[q] solusi yang lebih baik dari q 
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i += 1
        front.append(Q)
    del front[-1]

    rank = [0] * len(values1)
    for i in range(0, len(front)):
        for j in range(0, len(front[i])):
            rank[front[i][j]] = i + 1

    return rank
    # return front




def routing_phase_two(population_size, alpha, beta, population_routes, population_distances, distance_matrix, customers,
                      fleet_total_time, fleet_capacity):
    

    results = []
    for i in range(population_size):
        results.append(phase_two(alpha, beta, population_routes[i], population_distances[i], distance_matrix, customers, fleet_total_time, fleet_capacity))

    
    new_population_routes = [None] * population_size
    new_population_distances = [None] * population_size
    new_population_num_routes = [None] * population_size
    new_population_total_route_distances = [None] * population_size
    fitnesses = [None] * population_size

    for i in range(len(results)): 
        res = results[i]
        new_population_routes[i] = res[0]
        new_population_distances[i] = res[1]
        new_population_num_routes[i] = res[2]
        new_population_total_route_distances[i] = res[3]
        fitnesses[i] = res[4]

    
    return new_population_routes, new_population_distances, new_population_num_routes, new_population_total_route_distances, fitnesses

    

for i in range(1, num_customers_with_depot):
    customers.append({'id': i, 'demand': df_customers.loc[i, 'DEMAND'],  
                      'service_time': df_customers.loc[i, 'DUE DATE'] -  df_customers.loc[i, 'READY TIME'],
                      'ready_time': df_customers.loc[i, 'READY TIME'],
                      'due_time': df_customers.loc[i, 'DUE DATE'],
                       'complete_time': df_customers.loc[i, 'DUE DATE'] +  (df_customers.loc[i, 'DUE DATE'] -  df_customers.loc[i, 'READY TIME']) ,})
    
    fleet_total_time = df_fleets.loc[0, 'fleet_max_working_time']
    fleet_capacity = df_fleets.loc[0, 'fleet_capacity']

    customers_index = list(range(1, num_customers_with_depot))

for experiment in range(experiments):
    population_route_file = os.path.join("./solomon_benchmark/R1/experiments/R101_population_route_" + str(experiment) + ".csv")
    population_distance_matrix_file = os.path.join("./solomon_benchmark/R1/experiments/R101_population_distance_matrix_" + str(experiment) + ".csv")
    population_results_file = os.path.join("./solomon_benchmark/R1/experiments/R101_population_results_" + str(experiment) + ".csv")

    population_routes, population_distances, population_num_routes, population_total_route_distances = initial_population(population_size, random_chromosome_num,    greedy_chromosome_num, len(customers_index), customers, distance_matrix,
                                                                                            fleet_capacity, fleet_total_time,   customers_index,  route_radius
                                                                                            )
    population_routes, population_distances, population_num_routes, population_total_route_distances, fitnesses = routing_phase_two(population_size, alpha, beta, population_routes, population_distances, distance_matrix, customers, 
                                                                                                                 fleet_total_time, fleet_capacity)

    df_population_routes = pd.DataFrame({'routes': population_routes, 'distances': population_distances})
    df_population_routes.to_csv(population_route_file, index=False)

    # df_population_distances = pd.DataFrame({})
    # df_population_distances.to_csv(population_distance_matrix_file, index=False)


    df_initial_population_solution = pd.DataFrame({'num_vehicles': population_num_routes,
                                                    'total_distance': population_total_route_distances, 
                                                    'fitness': fitnesses})

    df_initial_population_solution.to_csv(population_results_file, index=False)

    print("membuat populasi experiment ke-", experiment)




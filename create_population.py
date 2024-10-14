import os
import math
import multiprocessing
import random
import pandas as pd
import sys
import concurrent.futures
import threading

generations = 50

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
for i in range(0, num_customers):
    for j in range(0, num_customers):
     
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
    allowed_neigbors = [] # -> 0, 1, 2, ... , 24 (customer)
    # available_customers -> 1, 2, 3, ..., 25
    available_customers_index = [i-1 for i in available_customers]
    for i in range(len(customers)):
        customer = customers[i]
        allowed_neigbors.append({'id': customer['id'], 'demand': customer['demand'],  
                      'service_time': customer['service_time'] ,
                      'ready_time': customer['ready_time'],
                      'due_time': customer['due_time'],
                       'complete_time': customer['complete_time'] ,})

        allowed_neigbors[i]['demand'] += curr_capacity

        allowed_neigbors[i]['distance_to_other'] = distance_matrix[curr_customer][i+1]

        allowed_neigbors[i]['arrival_time'] = allowed_neigbors[i]['distance_to_other'] + curr_time

        allowed_neigbors[i]['waiting_time'] = allowed_neigbors[i]['ready_time'] -  allowed_neigbors[i]['arrival_time']
        allowed_neigbors[i]['waiting_time'] = max(0,  allowed_neigbors[i]['arrival_time'])

        allowed_neigbors[i]['start_time'] = allowed_neigbors[i]['arrival_time'] + allowed_neigbors[i]['waiting_time']

        allowed_neigbors[i]['finish_time'] = allowed_neigbors[i]['start_time'] + allowed_neigbors[i]['service_time']

        allowed_neigbors[i]['return_time'] = distance_matrix[i+1][0] + allowed_neigbors[i]['finish_time']

    

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
            curr_distance += allowed_neighbors[curr_customer_idx]['distance_to_other']
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
            curr_distance += allowed_neighbors[curr_customer_idx]['distance_to_other']
            curr_capacity = allowed_neighbors[curr_customer_idx]['demand']
            curr_time = allowed_neighbors[curr_customer_idx]['finish_time']
            available_customers.remove(curr_customer)# remove ci
            min_distance = -1


            allowed_neighbors = allowed_neigbors_search(customers_size, customers, distance_matrix, available_customers, 
                    fleet_total_time, fleet_capacity, curr_time, curr_capacity,
                    curr_customer)
        
            allowed_neighbors_id = []
            for i in range(len(allowed_neighbors)):
                allowed_neighbors_id.append(allowed_neighbors[i]['id'])


            nearest_customer = None
            nearest_customer_idx = None
            for index in allowed_neighbors: # 0, 1, 2, 3, 4, ..., 25
                i = allowed_neighbors[index]['id']
                if allowed_neighbors[i]['distance_to_other'] < min_distance:
                    min_distance = allowed_neighbors[i]['distance_to_other']
                    nearest_customer = allowed_neighbors[i]['id']
                    nearest_customer_idx =  allowed_neighbors_id.index(curr_customer)


            exist = False
            for route in chromosome_routes:
                if nearest_customer in route:
                    exist = True

            if exist == False:
                curr_customer = nearest_customer
                curr_customer_idx = nearest_customer_idx
                curr_route.append(curr_customer)
                curr_distance += allowed_neighbors[curr_customer_idx]['distance_to_other']
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
            for index in allowed_neighbors: # 0, 1, 2, 3, 4, ..., 25
                i = allowed_neighbors[index]['id']
                if allowed_neighbors[i]['distance_to_other'] < min_distance:
                    min_distance = allowed_neighbors[i]['distance_to_other']
                    nearest_customer = allowed_neighbors[i]['id']
                    nearest_customer_idx = i
            
            exist = False
            for route in chromosome_routes:
                if nearest_customer in route:
                    exist = True

            if exist == False:
                curr_customer = nearest_customer
                curr_customer_idx = nearest_customer_idx
                curr_route.append(curr_customer)
                curr_distance += allowed_neighbors[curr_customer_idx]['distance_to_other']
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
        for i in range(random_chromosome_num):
            lock.acquire()
            futures.append(executor.submit(create_random_chromosome, customers_size, customers, distance_matrix, fleet_capacity, fleet_total_time, customers_index, ))
            lock.release()
        for j in range(greedy_chromosome_num):
            lock.acquire()
            futures.append(executor.submit(create_greedy_chromosome, customers_size, customers, distance_matrix, fleet_capacity, fleet_total_time, customers_index, route_radius, ))
            lock.release()

    for future in futures:
        with lock:
            results.append(future.result())
    
    population_routes = [-1 * population_size]
    population_distances = [-1 * population_size]
    population_num_routes =  [-1 * population_size]
    population_distances_sum = [-1 * population_size]

    
    population_routes[0::] = results[0::4]
    population_distances[0::] = results[1::4]
    population_num_routes[0::] = results[2::4]
    population_distances_sum[0::] = results[3::4]

    return population_routes, population_distances, population_num_routes, population_distances_sum

customers = []
for i in range(1, num_customers_with_depot):
    customers.append({'id': i, 'demand': df_customers.loc[i, 'DEMAND'],  
                      'service_time': df_customers.loc[i, 'DUE DATE'] -  df_customers.loc[i, 'READY TIME'],
                      'ready_time': df_customers.loc[i, 'READY TIME'],
                      'due_time': df_customers.loc[i, 'DUE DATE'],
                       'complete_time': df_customers.loc[i, 'DUE DATE'] +  (df_customers.loc[i, 'DUE DATE'] -  df_customers.loc[i, 'READY TIME']) ,})
    
    fleet_total_time = df_fleets.loc[0, 'fleet_max_working_time']
    fleet_capacity = df_fleets.loc[0, 'fleet_capacity']

    customers_index = list(range(1, num_customers_with_depot))

    for generation in range(generations):
        population_route_file = os.path.join("./solomon_benchmark/R1/generations/R101_population_route_" + str(generation) + ".csv")
        population_distance_matrix_file = os.path.join("./solomon_benchmark/R1/generations/R101_population_distance_matrix_" + str(generation) + ".csv")
        population_results_file = os.path.join("./solomon_benchmark/R1/generations/R101_population_results_" + str(generation) + ".csv")

        population_routes, population_distances, population_num_routes, population_distances_sum = initial_population(population_size, random_chromosome_num,    greedy_chromosome_num, len(customers_index), customers, distance_matrix,
                                                                                               fleet_capacity, fleet_total_time,   customers_index,  route_radius
                                                                                               )
        print("")
    


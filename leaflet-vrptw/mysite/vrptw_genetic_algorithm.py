import os
import math
import multiprocessing
import random
import pandas as pd
import sys
import concurrent.futures
import threading
import random
import copy
from .utils import argmin, argmax


class GA_VRPTW:
    def __init__(
        self,
        experiments=20,
        population_size=300,
        random_population=0.9,
        greedy_population=0.1,
        num_generations=400,
        tournament_size=4,
        tournament_threshold=0.8,
        crossover_prob=0.7,
        mutation_prob=0.1,
        alpha=100,
        beta=0.001,
    ):
        self.experiments = experiments
        self.population_size = population_size
        self.random_population = random_population
        self.greedy_population = greedy_population
        self.num_generations = num_generations
        self.tournament_size = tournament_size
        self.tournament_threshold = tournament_threshold
        self.population_routes = []
        self.population_distances = []
        self.population_results = []
        self.crossover_prob = crossover_prob
        self.mutation_prob = 1/population_size

        self.alpha = alpha
        self.beta = beta

    def create_population(self, customers_input, fleets_input, distance_matrix):
        """
        attrs:
        customers: list[dict] . dict= {'id': int, 'demand': int, 'service_time': int, 'due_time': int, 'ready_time': int, 'lat': float, 'lon': float}
        fleets: list[dict]. dict = {fleet_size: int, 'fleet_capacity': int, 'fleet_max_working_time': int, 'fleet_lat': float, 'fleet_lon': float}
        distance_matrix: dict[dict]
        """

        df_customers = pd.DataFrame(customers_input)
        df_fleets = pd.DataFrame(fleets_input)
        num_customers = df_customers.shape[0] - 1
        num_customers_with_depot = num_customers + 1

        random_chromosome_num = math.floor(
            self.population_size * self.random_population
        )
        greedy_chromosome_num = self.population_size - random_chromosome_num

        max_dist = -1
        min_dist = 999999

        for i in range(0, num_customers):
            for j in range(0, num_customers):
                if i != j:
                    max_dist = max(max_dist, distance_matrix[i][j])
                    min_dist = min(min_dist, distance_matrix[i][j])

        route_radius = (max_dist - min_dist) / 2
        customers = []

        fleet_total_time = df_fleets.loc[0, "fleet_max_working_time"]
        fleet_capacity = df_fleets.loc[0, "fleet_capacity"]
        for i in range(1, num_customers_with_depot):
            customers.append(
                {
                    "id": i,
                    "demand": df_customers.loc[i, "demand"],
                    "service_time": df_customers.loc[i, "due_time"]
                    - df_customers.loc[i, "ready_time"],
                    "ready_time": df_customers.loc[i, "ready_time"],
                    "due_time": df_customers.loc[i, "due_time"],
                    "complete_time": df_customers.loc[i, "due_time"]
                    + (
                        df_customers.loc[i, "due_time"]
                        - df_customers.loc[i, "ready_time"]
                    ),
                }
            )

        self.customers = customers
        self.distance_matrix = distance_matrix
        self.fleet = {
            "fleet_total_time": fleet_total_time,
            "fleet_capacity": fleet_capacity,
        }
        self.service_time = self.customers["service_time"]

        fleet_total_time = df_fleets.loc[0, "fleet_max_working_time"]
        fleet_capacity = df_fleets.loc[0, "fleet_capacity"]

        customers_index = list(range(1, num_customers_with_depot))

        for experiment in range(self.experiments):
            population_route_file = os.path.join(
                "./result/population/R101_population_route_" + str(experiment) + ".csv"
            )
            population_distance_matrix_file = os.path.join(
                "./result/population/R101_population_distance_matrix_"
                + str(experiment)
                + ".csv"
            )
            population_results_file = os.path.join(
                "./result/population/R101_population_results_"
                + str(experiment)
                + ".csv"
            )

            (
                population_routes,
                population_distances,
                population_num_routes,
                population_total_route_distances,
            ) = initial_population(
                self.population_size,
                random_chromosome_num,
                greedy_chromosome_num,
                len(customers_index),
                customers,
                distance_matrix,
                fleet_capacity,
                fleet_total_time,
                customers_index,
                route_radius,
            )
            (
                population_routes,
                population_distances,
                population_num_routes,
                population_total_route_distances,
                fitnesses,
            ) = routing_phase_two(
                self.population_size,
                self.alpha,
                self.beta,
                population_routes,
                population_distances,
                distance_matrix,
                customers,
                fleet_total_time,
                fleet_capacity,
            )

            df_population_routes = pd.DataFrame(
                {"routes": population_routes, "distances": population_distances}
            )

            df_population_routes.to_csv(population_route_file, index=False)

            df_initial_population_solution = pd.DataFrame(
                {
                    "num_vehicles": population_num_routes,
                    "total_distance": population_total_route_distances,
                    "fitness": fitnesses,
                }
            )

            self.population_routes.append(population_routes)

            self.population_results.append(
                {
                    "num_vehicles": population_num_routes,
                    "total_distance": population_total_route_distances,
                    "fitness": fitnesses,
                }
            )

            self.population_distances.append(population_distances)

            df_initial_population_solution.to_csv(population_results_file, index=False)

            print("membuat populasi experiment ke-", experiment)

    def elitism(self, pop_chromosome_routes, pop_chromosome_distances, pop_results):
        """
        pilih chromosome terbaik dari populasi (fitness paling kecil)

        dari papernya:
        An elite model is incorporated to ensure that the best
        individual is carried on into the next generation
        """
        fitnesses = []
        for i in range(len(pop_results)):
            fitnesses.append(pop_results[i]["fitness"])

        best_chromosome_idx = argmin(fitnesses)
        elite_chromosome_routes = pop_chromosome_routes[best_chromosome_idx]
        elite_chromosome_distances = pop_chromosome_distances[best_chromosome_idx]
        elite_chromosome_results = pop_results[best_chromosome_idx]
        return (
            elite_chromosome_routes,
            elite_chromosome_distances,
            elite_chromosome_results,
        )

    def solve(self):
        """
        solve vehicle routing problem with time windows menggunakan algoritma genetika

        lakukan experiment sebanyak self.experiments
        setiap experiment, lakukan iterasi sebanyak self.num_generations, simpan  chromosome terbaik di setiap populasi experiment
        setiap iterasi, lakukan recombination, mutation, elitism, dan update populasi

        solusi terbaik adalah solusi terbaik dari semua chromosome terbaik setiap eksperimen
        """
        best_solution_results = {}  # buat nyimpen solusi terbaik dari semua eksperiment
        best_solution_routes = []
        best_solution_distances = []

        experiment_results = (
            []
        )  # buat nyimpen chromosome yang terbaik di setiap eksperiment
        experiment_routes = []
        experiment_distances = []

        for experiment in range(self.experiments):
            for generation in range(self.num_generations):
                
                
                new_population_routes = []
                new_population_distances = []
                new_population_results = []

                # recombination (selection & crossover)
                (
                    new_population_results,
                    new_population_routes,
                    new_population_distances,
                ) = self.recombination_phase(
                    self.population_results,
                    new_population_results,
                    self.population_routes,
                    self.population_distances,
                    new_population_routes,
                    new_population_distances,
                )

                # mutasi
                (
                    new_population_results,
                    new_population_routes,
                    new_population_distances,
                ) = self.mutation(
                    new_population_results,
                    new_population_routes,
                    new_population_distances,
                )

                # elitism
                (
                    elite_chromosome_routes,
                    elite_chromosome_distances,
                    elite_chromosome_results,
                ) = self.elitism(
                    new_population_routes,
                    new_population_distances,
                    new_population_results,
                )

                # ganti chromosome terburuk dalam populasi dengan elite chromosome
                fitnesses = []
                for i in range(len(new_population_results)):
                    fitnesses.append(new_population_results[i]["fitness"])

                worst_chromosome_idx = argmax(fitnesses)
                new_population_routes[worst_chromosome_idx] = elite_chromosome_routes
                new_population_distances[worst_chromosome_idx] = (
                    elite_chromosome_distances
                )
                new_population_results[worst_chromosome_idx] = elite_chromosome_results

                # update populasi
                self.population_routes = copy.deepcopy(new_population_routes)
                self.population_distances = copy.deepcopy(new_population_distances)
                self.population_results = copy.deepcopy(new_population_results)

            # save chromosome terbaik di experiment ini
            bext_chromosome_in_experiment_idx = argmin(
                [chromosome["fitness"] for chromosome in self.population_results]
            )

            experiment_results.append(
                {
                    "num_vehicles": self.population_results[
                        bext_chromosome_in_experiment_idx
                    ]["num_vehicles"],
                    "total_distance": self.population_results[
                        bext_chromosome_in_experiment_idx
                    ]["total_distance"],
                    "fitness": self.population_results[
                        bext_chromosome_in_experiment_idx
                    ]["fitness"],
                }
            )

            experiment_routes.append(
                self.population_routes[bext_chromosome_in_experiment_idx]
            )
            experiment_distances.append(
                self.population_distances[bext_chromosome_in_experiment_idx]
            )

        # save best chromosome di semua experiments
        best_solution_idx = argmin(
            [chromosome["fitness"] for chromosome in experiment_results]
        )
        best_solution_results = experiment_results[best_solution_idx]
        best_solution_routes = experiment_routes[best_solution_idx]
        best_solution_distances = experiment_distances[best_solution_idx]

        return best_solution_results, best_solution_routes, best_solution_distances

    def tournament_selection(self, population_results):
        """

        dari papernya:
        . A set of Kindividuals are randomly selected from the population. This
        is known as the tournament set. In this paper, the set size is
        taken to be 4. We also select a random number r, between
        0 and 1. If r is less than 0.8 (0.8 is set empirically by trying
        values 0.6, 0.7, 0.8, 0.9 and 1.0), the fittest individual in
        the tournament set is then chosen as the one to be used
        for reproduction. Otherwise, any chromosome is chosen for
        reproduction from the tournament set.

        """
        random_chromosomes_idx = random.sample(
            range(population_results), self.tournament_size
        )  # pilih random chromosome sebanyak tournament_size
        random_chromosomes = [
            population_results[i] for i in random_chromosomes_idx
        ]  # ambil chromosome dari index yang sudah dipilih
        if random.uniform(0, 1) < self.tournament_threshold:
            # best_chromosome = min(random_chromosomes, key=lambda x: x['fitness'])
            rand_chrom_fitnesses = [
                chromosome["fitness"] for chromosome in random_chromosomes
            ]
            best_chromosome_idx = argmin(
                rand_chrom_fitnesses
            )  # pilih chromosome dengan fitness terkecil
            best_chromosome = random_chromosomes_idx[best_chromosome_idx]
        else:
            best_chromosome = random.choice(
                random_chromosomes_idx
            )  # pilih random chromosome dari tournament set
        return best_chromosome

    def calculate_new_eta(self, route):
        """
        new chromosome eta

        """
        eta = 0
        for i in range(1, len(route)):
            eta += self.distance_matrix[route[i - 1]][route[i]]
        return eta

    def mutation_chromosome(
        self,
        chromosome_routes,
        chromosome_distances,
    ):
        """
        inversion mutasi satu chromosome

        attrs:
        chromosome_routes: List[List[int]] -> [[0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0]] -> rute-rute setiap vehicle (satu chromosome routes)
        chromosome_distances: List[int] -> [100, 200, 300] -> eta total rute setiap vehicle

        return:
        is_this_route_valid: Bool -> True jika rute valid, False jika tidak valid
        chromosome_routes: List[List[int]] -> [[0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0]] -> rute-rute setiap vehicle setelah dimutasi
        chromosome_distances: List[int] -> [100, 200, 300] -> eta total rute setiap vehicle setelah dimutasi
        """
        if random.random() < self.mutation_prob:
            is_this_route_valid = False
            routes_len = []

            for i in range(len(chromosome_routes)):
                # untuk setiap rute vehicle di chromosome, kita hitung panjang rute nya
                routes_len.append(len(chromosome_routes[i]))

            feasible_idx = [] # index rute vehicle yang panjang rute nya >= 5
            for i in range(len(routes_len)):
                if routes_len[i] >= 5:
                    # jika panjang rute vehicle >= 5, maka kita bisa mutasi rute vehicle ini
                    feasible_idx.append(i)

            if len(feasible_idx) > 0:
                #In this paper mutation is carried out only in one randomly chosen route so as to minimize total route disruption
                rand_route = random.choice(feasible_idx)
                mutated_route = copy.deepcopy(chromosome_routes[rand_route]) # deep copy agar perubahan 

                mutation_len = random.choice([2, 3])  # 2 atau 3 customer (panjang gen dari chromosome yang akan dimutasi)
                len_mutated_route = len(mutated_route)
                start_index = 1

                if len_mutated_route - 2 > mutation_len:
                    # jika jumlah customer yang dikunjungi vehicle > mutation_len, maka start_idx = random integer dari 1 sampai len_mutated_route - 2 - mutation_len
                    # start cut point mutation
                    start_index = random.randint(
                        1, len_mutated_route - 2 - mutation_len
                    )  # set start cut point mutation exclude last customer

                temp_mutated_custs = copy.deepcopy(
                    mutated_route[start_index : start_index + mutation_len]
                )  # get customers yang mau dimutasi
                reversed_mutated_custs = temp_mutated_custs[::-1]  # inversion mutation

                mutated_route[start_index : start_index + mutation_len] = (
                    reversed_mutated_custs  # perbarui rute vehicle dengan inverted customers di antara 2 cut point
                )

                is_this_route_valid, new_route_distance = is_route_valid(
                    mutated_route,
                    self.distance_matrix,
                    self.customers,
                    self.fleet["fleet_total_time"],
                    self.fleet["fleet_capacity"],
                )  # cek apakah rute baru valid

                if (
                    is_this_route_valid == True
                    and new_route_distance <= chromosome_distances[rand_route]
                ):
                    # jika rute baru setelah dimutasi valid & jaraknya lebih kecil, maka kita update rute vehicle
                    chromosome_routes[rand_route] = mutated_route
                    chromosome_distances[rand_route] = new_route_distance

        return is_this_route_valid, chromosome_routes, chromosome_distances

    def mutation(
        self,
        pop_results,
        pop_chromosome_routes,
        pop_chromosome_distances,
    ):
        """
        inverse mutation setiap chromosome di dalam populasi

        attrs:
        pop_results: List[Dict] -> [{"num_vehicles": int, "total_distance": int, "fitness": float}, ...] -> hasil evaluasi setiap chromsome di  populasi
        pop_chromosome_routes: List[List[List[int]]] -> [[[0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0]], ...] -> rute-rute setiap vehicle untuk setiap chromosome di populasi
        pop_chromosome_distances: List[List[int][] -> [[100, 200, 300], ...] -> eta total rute setiap vehicle untuk setiap chromosome di populasi

        return:
        pop_results: List[Dict] -> [{"num_vehicles": int, "total_distance": int, "fitness": float}, ...] -> hasil evaluasi setiap chromsome di populasi setelah mutasi
        pop_chromosome_routes: List[List[int]] -> [[0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0]] -> rute-rute  setiap chromsome di populasi setelah mutasi
        pop_chromosome_distances: List[int] -> [100, 200, 300] ->eta total rute setiap vehicle untuk setiap chromosome di populasi setelah mutasi

        """

        mutation_res = []
        for i in range(0, self.population_size):
            # mutasi setiap chromosome di dalam populasi
            mutation_res.append(
                self.mutation_chromosome(
                    pop_chromosome_routes[i], pop_chromosome_distances[i]
                )
            )

        unzipped = zip(*mutation_res)
        results = list(unzipped)

        updated_routes = []  # menyimpan chrosomsome yang dimutasi
        for i in range(0, self.population_size):
            if mutation_res[i][0] == True:
                updated_routes.append(i)

        if len(updated_routes) > 0:
            # jika jumlah chromosome yang dimutasi > 0, maka kita update populasi dengan chromosome yang dimutasi
            for i in updated_routes:
                # results = [is_route_valid, chromosome_routes, chromosome_distances]
                pop_chromosome_routes[i] = copy.deepcopy(results[i][1])
                pop_chromosome_distances[i] = copy.deepcopy(results[i][2])
                pop_results[i] = {
                    "num_vehicles": len(results[i][1]),
                    "total_distance": sum(results[i][2]),
                    "fitness": fitness_function_weighted_sum(
                        self.alpha, self.beta, results[i][2], results[i][1]
                    ),
                }

        return pop_results, pop_chromosome_routes, pop_chromosome_distances

    def insert_customer(self, cust_to_insert, route, route_distance):
        """
        kita coba insert customer_to_insert ke dalam route
        kalau ternyata pas udah insert lebih baik, maka kita update route
        misal mau insert customer "1" ke [0,2,3,4,0] -> kita coba satu satu posisi dari index 1 sampai index 3 (exclude titik depot)
        kalau lebih bagus & insertnya bikin jarak rute baru lebih pendek maka kita update rute, kita pilih posisi yang bikin jarak rute baru paling pendek

        attrs:
        customer_to_insert: Int customer yang mau diinsert
        route: List[Int]-> [0,1,2,3,0] rute satu vehicle
        route_distance: Float jarak rute  satu vehicle
        arr_customers: np.array data customer
        """
        inserted_status = False
        # len = 5 -> [0, 1, 2, 3, 0] -> 1, 2, 3 adalah customer, kita cuma bisa insert customer baru ke  posisi setelah cust 1/2/3 atau sebelum cust 1
        arr_positions_to_insert = [
            i for i in range(1, len(route[1:-1]) + 1)
        ]  # posisi insert bukan di depot (bukan di titik awal dan akhir rute)
        update_route_distance_eval = 999999999
        updated_route = route.copy()  # rute baru yang udah diinsert cust_to_insert
        updated_distance = route_distance
        for positions_to_insert in arr_positions_to_insert:
            # coba satu satu posisi dari index 1 sampai sebelum customer akhir (exclude titik depot)
            # kita coba insert customer_to_insert ke dalam route di posisi positions_to_insert
            # kita pake rute yang bikin distance rute baru lebih pendek dan paling kecil

            new_route = route.copy()
            new_route.insert(positions_to_insert, cust_to_insert)
            route_validity_status, curr_dist = is_route_valid(
                new_route,
                self.distance_matrix,
                self.customers,
                self.fleet["fleet_total_time"],
                self.fleet["fleet_capacity"],
                self.service_time,
            )  # cek apakah setelah insert customer_to_insert, rute vehicle yang baru masih valid

            if route_validity_status == True:
                # jika rute vehicle yang baru masih valid, maka kita cek apakah jarak rute baru lebih pendek
                inserted_status = True
                distance_eval = (
                    curr_dist - route_distance
                )  # jarak rute baru - jarak rute lama
                if distance_eval < update_route_distance_eval:
                    # jika jarak rute baru lebih pendek, maka kita update rute vehicle
                    updated_route = new_route
                    update_route_distance_eval = distance_eval
                    updated_distance = curr_dist

        # return status insert, rute vehicle yang baru, jarak rute vehicle yang baru, jarak rute vehicle yang baru - jarak rute vehicle lama, setelah diinsert customer_to_insert
        return (
            inserted_status,
            updated_route,
            updated_distance,
            update_route_distance_eval,
        )

    def insertion(
        self, cust_other_parent_to_insert, chromosome_routes, chromosome_distances
    ):
        """
        insert customer yang ada di cust_other_parent_to_insert ke dalam chromosome_routes, insert ke best position
        - best position = posisi yang bikin eta rute baru lebih pendek & memenuhi semua constraint vrptw


        dari paper:
        the next step is to locate the best possible locations for the removed customers in the corresponding children.

        attrs:
        cust_other_parent_to_insert: List[int] -> [1, 2, 3] -> customer yang akan diinsert ke dalam chromosome_routes
        chromosome_routes: List[List[int]] -> [[0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0]] -> routes kromsom =  rute-rute setiap vehicle
        chromosome_distances: List[int] -> [100, 200, 300] -> eta total rute setiap vehicle

        """
        for cust_to_insert in cust_other_parent_to_insert:
            # coba insert setiap customer di cust_other_parent_to_insert ke dalam chromosome_routes
            curr_chrom_routes = chromosome_routes.copy() # shallow copy biar tidak keupdate chromosome_routes
            curr_chrom_distances = chromosome_distances.copy()
            metadata = {
                "routes": chromosome_routes,
                "distances": chromosome_distances,
                "is_inserted": [False] * len(chromosome_routes),
                "updated_distance": [0] * len(chromosome_routes),
            }

            is_inserted = False

            for i in range(len(chromosome_routes)):
                # coba insert custtoinsert ke dalam rute vehicle ke-i
                route = chromosome_routes[i]  # rute vehicle ke-i
                route_distance = chromosome_distances[i]

                is_inserted, new_route, new_distance = self.insert_customer(
                    cust_to_insert, route, route_distance
                )  # coba insert customer_to_insert ke dalam rute vehicle ke-i (insert ke best position)

                if is_inserted:
                    # jika bisa diinsert ke rute vehicle ke-i, maka kita update metadata rute vehicle ke-i
                    metadata["is_inserted"][i] = True
                    metadata["updated_distance"][i] = new_distance
                    metadata["updated_routes"][i] = new_route

            metadata["distance_diff"] = [
                metadata["updated_distance"][i] - metadata["distances"][i]
                for i in range(len(metadata["updated_distance"]))
            ]  # jarak rute baru - jarak rute lama untuk setiap rute vehicle-vehicle di chromosome

            updated_chrom = (
                []
            )  # berisi rute vehicle yang bisa diinsert customer_to_insert
            for i in range(len(metadata["routes"])):
                if metadata["distance_diff"][i] > 0:
                    updated_chrom.append(metadata["distance_diff"][i])

            if len(updated_chrom) > 0:
                # jika  rute vehicle yang bisa diinsert customer_to_insert lebih dari 0, maka kita pilih rute vehicle yang jarak rute baru - jarak rute lama nya paling kecil

                min_route_idx = argmin(
                    updated_chrom
                )  # pilih rute vehicle yang jarak rute baru - jarak rute lama nya paling kecil
                curr_chrom_routes[min_route_idx] = copy.deepcopy(
                    metadata["updated_routes"][min_route_idx]
                )  # update rute vehicle yang baru

                curr_chrom_distances[min_route_idx] = metadata["updated_distance"][
                    min_route_idx
                ]  # update jarak rute vehicle yang baru

                curr_chrom_total_distance = sum(
                    curr_chrom_distances
                )  # update jarak total rute vehicle yang baru

            else:
                # create new vehicle route jika memang customer tidak bisa diinsert ke salah satu rute vehicle
                new_vehicle_route = [
                    0,
                    cust_to_insert,
                    0,
                ]  # vehicle baru hanya mengunjungi customer_to_insert
                curr_chrom_routes.append(
                    new_vehicle_route
                )  # tambahkan vehicle baru ke dalam chromosome_routes
                curr_chrom_distances.append(self.calculate_new_eta(new_vehicle_route))
                curr_chrom_total_distance = sum(curr_chrom_distances)

            chromosome_routes = copy.deepcopy(
                curr_chrom_routes
            )  # update chromosome_routes dengan chromosome yang baru diinsert customer_to_insert
            chromosome_distances = copy.deepcopy(curr_chrom_distances)
            curr_chrom_total_distance = sum(chromosome_distances)
        return chromosome_routes, chromosome_distances, curr_chrom_total_distance

    def removal_crossover_customer(
        self, remove_customers, chromosome_opposite_routes, chromosome_distances
    ):
        """
        hapus customer di remove_customers dari chromosome_opposite_routes

        attrs:
        remove_customers: List[int] -> [1, 2, 3] -> customer yang akan dihapus di chromosome_opposite_routes
        chromosome_opposite_routes: List[List[int]] -> [[0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0]] -> rute-rute setiap vehicle
        chromosome_distances: List[int] -> [100, 200, 300] -> eta total rute setiap vehicle
        """
        for i in remove_customers:
            for j in range(len(chromosome_opposite_routes)):
                chromosome_opposite_vehicle_route = chromosome_opposite_routes[j]
                if i in chromosome_opposite_vehicle_route:
                    if len(chromosome_opposite_vehicle_route) > 3:
                        # jika vehicle ke j di rute nya mengunjungi lebih dari satu customer, maka hapus customer i
                        chromosome_opposite_vehicle_route.remove(i)
                        chromosome_distances[j] = self.calculate_new_eta(
                            chromosome_opposite_vehicle_route
                        )
                        break
                    elif len(chromosome_opposite_vehicle_route) == 3:
                        # jika vehicle ke j di rute nya hanya mengunjungi satu customer, maka hapus rute vehicle ke j
                        chromosome_opposite_routes.pop(j)
                        chromosome_distances.pop(j)
                        j -= 1
                        break

        return chromosome_opposite_routes, chromosome_distances

    def recombination(
        self,
        population_results,
        population_routes,
        population_distances,
    ):
        """
        tournament selection dan crossover ( Best cost route crossover (BCRC))


        dari papernya:
        , from each parent, a route is chosen randomly. In this case, for P1,
        route R2 with customers 5 and 6 is chosen, while for P2,
        route R3 with customers 7 and 3 is picked. Next, for a given
        parent, the customers in the chosen route from the opposite
        parent are removed. For example in Step a, for parent P1,
        customers 7 and 3 (which belonged to the randomly selected
        route in P2) are removed from P1 resulting in the upcoming
        child C1. Likewise customers 5 and 6 which belonged to a
        route in P1 are removed from the routes in P2, resulting in
        the upcoming C2., the next step is to locate the best possible locations for the removed
        customers in the corresponding children.

        attrs:
        tournament_size: Int -> 4 -> ukuran tournament selection
        tournament_threshold: Float -> 0.7 -> threshold tournament selection
        population_results: List[Dict] -> [{"num_vehicles": int, "total_distance": int, "fitness": float}, ...] -> hasil evaluasi setiap chromsome di populasi
        population_size: Int -> 300 -> ukuran populasi
        population_routes: List[List[List[int]]] -> [[[0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0]], ...] -> rute-rute setiap vehicle untuk setiap chromosome di populasi
        population_distances: List[List[int]] -> [[100, 200, 300], ...] -> eta total rute setiap vehicle untuk setiap chromosome di populasi
        customers: List[Dict] -> [{'id': int, 'demand': int, 'service_time': int, 'due_time': int, 'ready_time': int, 'lat': float, 'lon': float}, ...] -> data customer
        distance_matrix: Dict[Dict] -> {0: {0: 0, 1: 100, 2: 200, ...}, 1: {0: 100, 1: 0, 2: 100, ...}, ...} -> eta/jarak antar customer
        fleet_capacity: Int -> 100 -> kapasitas vehicle
        fleet_total_time: Int -> 1000 -> waktu total kerja vehicle
        service_time: Int -> 100 -> waktu service setiap customer (anggap semua service time semua customer sama)

        return:
        chromosome1_routes: List[List[int]] -> [[0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0]] -> rute-rute vehicle chromosome 1
        chromosome2_routes: List[List[int]] -> [[0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0]] -> rute-rute vehicle chromosome 2
        chromosome1_distances: List[int] -> [100, 200, 300] -> eta total rute setiap vehicle chromosome 1
        chromosome2_distances: List[int] -> [100, 200, 300] -> eta total rute setiap vehicle chromosome 2



        """
        parent_one = self.tournament_selection(population_results)
        parent_two = self.tournament_selection(population_results)
        while parent_one == parent_two:
            # kalau parent_one == parent_two, kita pilih parent_two yang lain
            parent_two = self.tournament_selection(population_results)

        # rute-rute vehicles parent one
        parent_one_routes = population_routes[parent_one]
        parent_two_routes = population_routes[parent_two]
        parent_one_distances = population_distances[parent_one]
        parent_two_distances = population_distances[parent_two]

        remove_route_one = parent_one_routes[
            random.randint(0, len(parent_one_routes) - 1)
        ]  # pilih satu rute vehicle dari parent one, customer-customer di rute ini akan dihapus di parent two dan diinsert ulang di parent two

        remove_route_two = parent_two_routes[
            random.randint(0, len(parent_two_routes) - 1)
        ]  # pilih satu rute vehicle dari parent two, customer-customer di rute ini akan dihapus di parent one dan diinsert ulang di parent one

        # . Next, for a given parechromosome2_distancesnt, the customers in the chosen route from the opposite parent are removed.


        chromosome1_routes = parent_one_routes.copy()
        chromosome2_routes = parent_two_routes.copy()
        chromosome1_distances = parent_one_distances.copy()
        chromosome2_distances = parent_two_distances.copy()
        chromosome1_total_distance = sum(chromosome1_distances)
        chromosome2_total_distance = sum(chromosome2_distances)

        if random.random() < self.crossover_prob:
            # jika random < crossover_prob, maka kita lakukan crossover
            parent_one_routes, parent_one_distances = self.removal_crossover_customer(
                remove_route_two, parent_one_routes, parent_one_distances
            )  # hapus customer di remove_route_two dari routes parent_one

            parent_two_routes, parent_two_distances = self.removal_crossover_customer(
                remove_route_one, parent_two_routes, parent_two_distances
            )  # hapus customer di remove_route_one dari routes parent_two

            chromosome1_routes, chromosome1_distances, chromosome1_total_distance = (
                self.insertion(remove_route_two, parent_one_routes, parent_one_distances)
            )  # insert customer yang ada di remove_route_two tadi ke parent_one, tapi sekarang insert ke best position

            chromosome2_routes, chromosome2_distances, chromosome2_total_distance = (
                self.insertion(remove_route_one, parent_two_routes, parent_two_distances)
            )  # insert customer yang ada di remove_route_one tadi ke parent_two, tapi sekarang insert ke best position
        
            
        
        return (
            chromosome1_routes,
            chromosome2_routes,
            chromosome1_distances,
            chromosome2_distances,
            chromosome1_total_distance,
            chromosome2_total_distance,
        )

    def recombination_phase(
        self,
        population_results,
        new_population_results,
        population_routes,
        population_distances,
        new_population_routes,
        new_population_distances,
    ):
        """
        selection dan crossover

        attrs:
        population_results: List[Dict] -> [{"num_vehicles": int, "total_distance": int, "fitness": float}, ...] -> hasil evaluasi setiap chromsome di populasi
        new_population_results: List[Dict] -> [{"num_vehicles": int, "total_distance": int, "fitness": float}, ...] -> hasil evaluasi setiap chromsome di populasi setelah recombination
        population_routes: List[List[List[int]]] -> [[[0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0]], ...] -> rute-rute setiap vehicle untuk setiap chromosome di populasi
        population_distances: List[List[int]] -> [[100, 200, 300], ...] -> eta total rute setiap vehicle untuk setiap chromosome di populasi
        new_population_routes: List[List[List[int]]] -> [[[0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0]], ...] -> rute-rute setiap vehicle untuk setiap chromosome di populasi setelah recombination
        new_population_distances: List[List[int]] -> [[100, 200, 300], ...] -> eta total rute setiap vehicle untuk setiap chromosome di populasi setelah recombination

        
        return:
        new_population_results: List[Dict] -> [{"num_vehicles": int, "total_distance": int, "fitness": float}, ...] -> hasil evaluasi setiap chromsome di populasi setelah recombination
        new_population_routes: List[List[List[int]]] -> [[[0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0]], ...] -> rute-rute setiap vehicle untuk setiap chromosome di populasi setelah recombination
        new_population_distances: List[List[int]] -> [[100, 200, 300], ...] -> eta total rute setiap vehicle untuk setiap chromosome di populasi setelah recombination


        """
        recombination_size = int(self.population_size / 2)
        childrens_after_recombination = []

        for i in range(recombination_size):
            (
                chrom_one_routes,
                chrom_two_routes,
                chrom_one_distances,
                chrom_two_distances,
                chrom_one_tot_distance,
                chrom_two_tot_distance,
            ) = self.recombination(
                population_results,
                population_routes,
                population_distances,
            )
            childrens_after_recombination.append(
                (
                    chrom_one_routes, # 150
                    chrom_two_routes, # 150
                    chrom_one_distances,
                    chrom_two_distances,
                    chrom_one_tot_distance,
                    chrom_two_tot_distance,
                )
            )

        for i in range(len(childrens_after_recombination)):
            res = childrens_after_recombination[i]
            new_population_routes.append(res[0]) 
            new_population_routes.append(res[1])
            new_population_distances.append(res[2])
            new_population_distances.append(res[3])
            new_population_results.append(
                {
                    "num_vehicles": len(res[0]),
                    "total_distance": res[4],
                    "fitness": fitness_function_weighted_sum(
                        self.alpha, self.beta, res[2], res[0]
                    ),
                }
            )
            new_population_results.append(
                {
                    "num_vehicles": len(res[1]),
                    "total_distance": res[5],
                    "fitness": fitness_function_weighted_sum(
                        self.alpha, self.beta, res[3], res[1]
                    ),
                }
            )

        return new_population_results, new_population_routes, new_population_distances


def euclidean_distance(p1, p2):
    return math.sqrt((p1["lon"] - p2["lon"]) ** 2 + (p1["lat"] - p2["lat"]) ** 2)


def allowed_neigbors_search(
    customer_size,
    customers,
    distance_matrix,
    available_customers,
    fleet_total_time,
    fleet_capacity,
    curr_time,
    curr_capacity,
    curr_customer,
):
    # customers -> [0, 1, 2, ... 24]
    allowed_neigbors = [None] * len(
        available_customers
    )  # -> 0, 1, 2, ... , 24 (customer)
    # available_customers -> 1, 2, 3, ..., 25

    available_customers_index = [i - 1 for i in available_customers]
    for i in range(len(available_customers_index)):
        cust_index = available_customers_index[i]
        customer = customers[cust_index]
        allowed_neigbors[i] = {
            "id": customer["id"],
            "demand": customer["demand"],
            "service_time": customer["service_time"],
            "ready_time": customer["ready_time"],
            "due_time": customer["due_time"],
            "complete_time": customer["complete_time"],
        }

        allowed_neigbors[i]["demand"] += curr_capacity

        allowed_neigbors[i]["distance_to_curr_customer"] = distance_matrix[
            curr_customer
        ][cust_index + 1]

        allowed_neigbors[i]["arrival_time"] = (
            allowed_neigbors[i]["distance_to_curr_customer"] + curr_time
        )

        allowed_neigbors[i]["waiting_time"] = (
            allowed_neigbors[i]["ready_time"] - allowed_neigbors[i]["arrival_time"]
        )
        allowed_neigbors[i]["waiting_time"] = max(
            0, allowed_neigbors[i]["waiting_time"]
        )

        allowed_neigbors[i]["start_time"] = (
            allowed_neigbors[i]["arrival_time"] + allowed_neigbors[i]["waiting_time"]
        )

        allowed_neigbors[i]["finish_time"] = (
            allowed_neigbors[i]["start_time"] + allowed_neigbors[i]["service_time"]
        )

        allowed_neigbors[i]["return_time"] = (
            distance_matrix[cust_index + 1][0] + allowed_neigbors[i]["finish_time"]
        )

    for i in range(len(allowed_neigbors) - 1, -1, -1):
        neighbor = allowed_neigbors[i]
        if (
            neighbor["demand"] > fleet_capacity
            or neighbor["return_time"] > fleet_total_time
            or allowed_neigbors[i]["start_time"] < allowed_neigbors[i]["ready_time"]
            or allowed_neigbors[i]["finish_time"] > allowed_neigbors[i]["complete_time"]
        ):
            allowed_neigbors.pop(i)

    return allowed_neigbors


def create_random_chromosome(
    customers_size,
    customers,
    distance_matrix,
    fleet_capacity,
    fleet_total_time,
    customers_index,
):

    available_customers = customers_index.copy()

    chromosome_routes = []
    chromosome_distances = []
    curr_route = [0]  # start dari city-0
    curr_capacity = 0
    curr_time = 0
    curr_customer = 0
    curr_distance = 0

    while len(available_customers) != 0:
        allowed_neighbors = allowed_neigbors_search(
            customers_size,
            customers,
            distance_matrix,
            available_customers,
            fleet_total_time,
            fleet_capacity,
            curr_time,
            curr_capacity,
            curr_customer,
        )
        if len(allowed_neighbors) != 0:
            allowed_neighbors_id = []
            for i in range(len(allowed_neighbors)):
                allowed_neighbors_id.append(allowed_neighbors[i]["id"])
            curr_customer = int(random.choice(allowed_neighbors_id))
            curr_customer_idx = allowed_neighbors_id.index(curr_customer)

            curr_route.append(curr_customer)
            curr_distance += allowed_neighbors[curr_customer_idx][
                "distance_to_curr_customer"
            ]
            curr_capacity = allowed_neighbors[curr_customer_idx]["demand"]
            curr_time = allowed_neighbors[curr_customer_idx]["finish_time"]
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

    return (
        chromosome_routes,
        chromosome_distances,
        len(chromosome_routes),
        total_distance,
    )


def create_greedy_chromosome(
    customers_size,
    customers,
    distance_matrix,
    fleet_capacity,
    fleet_total_time,
    customers_index,
    route_radius,
):

    available_customers = customers_index.copy()  # 1,2,3,4, ..., 25

    chromosome_routes = []
    chromosome_distances = []
    curr_route = [0]  # start dari city-0
    curr_capacity = 0
    curr_time = 0
    curr_customer = 0
    curr_distance = 0

    prev_random_choice = True

    while len(available_customers) != 0:
        allowed_neighbors = allowed_neigbors_search(
            customers_size,
            customers,
            distance_matrix,
            available_customers,
            fleet_total_time,
            fleet_capacity,
            curr_time,
            curr_capacity,
            curr_customer,
        )

        allowed_neighbors_id = []
        for i in range(len(allowed_neighbors)):
            allowed_neighbors_id.append(allowed_neighbors[i]["id"])

        if len(allowed_neighbors) != 0 and prev_random_choice == True:

            curr_customer = int(random.choice(allowed_neighbors_id))
            curr_customer_idx = allowed_neighbors_id.index(curr_customer)

            curr_route.append(curr_customer)  # add ci to l
            curr_distance += allowed_neighbors[curr_customer_idx][
                "distance_to_curr_customer"
            ]
            curr_capacity = allowed_neighbors[curr_customer_idx]["demand"]
            curr_time = allowed_neighbors[curr_customer_idx]["finish_time"]
            available_customers.remove(curr_customer)  # remove ci
            min_distance = 999999999

            allowed_neighbors = allowed_neigbors_search(
                customers_size,
                customers,
                distance_matrix,
                available_customers,
                fleet_total_time,
                fleet_capacity,
                curr_time,
                curr_capacity,
                curr_customer,
            )

            allowed_neighbors_id = []
            for i in range(len(allowed_neighbors)):
                allowed_neighbors_id.append(allowed_neighbors[i]["id"])

            nearest_customer = None
            nearest_customer_idx = None
            for index in range(len(allowed_neighbors)):  # 0, 1, 2, 3, 4, ..., 24
                if allowed_neighbors[index]["distance_to_curr_customer"] < min_distance:
                    min_distance = allowed_neighbors[index]["distance_to_curr_customer"]
                    nearest_customer = allowed_neighbors[index]["id"]
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
                curr_distance += allowed_neighbors[curr_customer_idx][
                    "distance_to_curr_customer"
                ]
                curr_capacity = allowed_neighbors[curr_customer_idx]["demand"]
                curr_time = allowed_neighbors[curr_customer_idx]["finish_time"]
                available_customers.remove(curr_customer)
                prev_random_choice = False
            else:
                prev_random_choice = True
                continue

        elif len(allowed_neighbors) != 0 and prev_random_choice == False:

            nearest_customer = None
            nearest_customer_idx = None
            min_distance = 999999999

            for index in range(len(allowed_neighbors)):  # 0, 1, 2, 3, 4, ..., 24
                if allowed_neighbors[index]["distance_to_curr_customer"] < min_distance:
                    min_distance = allowed_neighbors[index]["distance_to_curr_customer"]
                    nearest_customer = allowed_neighbors[index]["id"]
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
                curr_distance += allowed_neighbors[curr_customer_idx][
                    "distance_to_curr_customer"
                ]
                curr_capacity = allowed_neighbors[curr_customer_idx]["demand"]
                curr_time = allowed_neighbors[curr_customer_idx]["finish_time"]
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

    return (
        chromosome_routes,
        chromosome_distances,
        len(chromosome_routes),
        total_distance,
    )


def initial_population(
    population_size,
    random_chromosome_num,
    greedy_chromosome_num,
    customers_size,
    customers,
    distance_matrix,
    fleet_capacity,
    fleet_total_time,
    customers_index,
    route_radius,
):
    lock = threading.Lock()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for _ in range(random_chromosome_num):
            futures.append(
                executor.submit(
                    create_random_chromosome,
                    customers_size,
                    customers,
                    distance_matrix,
                    fleet_capacity,
                    fleet_total_time,
                    customers_index,
                )
            )
        for _ in range(greedy_chromosome_num):
            futures.append(
                executor.submit(
                    create_greedy_chromosome,
                    customers_size,
                    customers,
                    distance_matrix,
                    fleet_capacity,
                    fleet_total_time,
                    customers_index,
                    route_radius,
                )
            )

    for future in futures:
        with lock:
            results.append(future.result())

    population_routes = [None] * population_size
    population_distances = [None] * population_size
    population_num_routes = [None] * population_size
    population_total_route_distances = [None] * population_size

    for i in range(len(results)):
        res = results[i]
        population_routes[i] = res[0]
        population_distances[i] = res[1]
        population_num_routes[i] = res[2]
        population_total_route_distances[i] = res[3]

    return (
        population_routes,
        population_distances,
        population_num_routes,
        population_total_route_distances,
    )


def is_route_valid(route, distance_matrix, customers, fleet_total_time, fleet_capacity):
    """
    cek apakah rute valid atau gak

    attrs:
    route: List[int] -> [0, 1, 2, 3, 0] -> rute vehicle
    distance_matrix: dict[dict] -> {0: {0: 0, 1: 100, 2: 200, ...}, 1: {0: 100, 1: 0, 2: 100, ...}, ...}
    customers: List[dict] -> [{'id': 1, 'demand': 10, 'service_time': 10, 'due_time': 100, 'ready_time': 0, 'lat': 0.0, 'lon': 0.0}, ...]
    fleet_total_time: int -> 1000
    fleet_capacity: int -> 100

    """
    route_valid = True

    city_one_idx = route[1] - 1  # index customer pertama yang dikunjungi
    # demang, ready_time, due_time, curr_time, curr_time_dist setelah kunjungi cusotmer pertama
    curr_demand = customers[city_one_idx]["demand"]
    ready_time = customers[city_one_idx]["ready_time"]
    due_time = customers[city_one_idx]["due_time"]
    curr_time = distance_matrix[route[0]][route[1]]
    curr_time_dist = distance_matrix[route[0]][route[1]]

    # kunjungi customer selanjutnya
    # skip depot, buat kunjungi next customers setelah customer ke-1
    for i in range(
        1, len(route) - 2
    ):  # next_customers exclude titik depot(0) & exclude last city sebelum depot
        # 1-2, 2-3, ....
        curr_time += distance_matrix[route[i]][
            route[i + 1]
        ]  # curr_time setelah kunjungi customer ke i+1
        curr_time_dist += distance_matrix[route[i]][
            route[i + 1]
        ]  # curr_time_dist setelah kunjungi customer ke i+1
        next_city_idx = route[i + 1] - 1  # next city index di list customers
        ready_time = customers[next_city_idx][
            "ready_time"
        ]  # ready time customer ke i+1
        due_time = customers[next_city_idx]["due_time"]  # due time customer ke i+1
        service_time = customers[next_city_idx][
            "service_time"
        ]  # service time customer ke i+1
        curr_demand += customers[next_city_idx]["demand"]  # demand customer ke i+1
        if curr_time > due_time or curr_demand > fleet_capacity:
            # cek apakah waktu saat kurir kunjungi customer ke-i+1 lebih dari due time customer ke-i+1
            # atau demand customer ke-i+1 lebih dari kapasitas fleet
            # kalau iya berarti rute tidak valid
            route_valid = False
            break
        else:
            # else rute valid
            wait_time = max(
                0, ready_time - curr_time
            )  # waktu tunggu kurir =  waktu ready customer ke-i+1 - waktu kurir sampai di customer ke-i+1
            # kalau kurir sampai lebih terlambat dari ready_time berarti gak ada wait_time (wait_time = 0)
            if (
                curr_time + wait_time + service_time
                > fleet_total_time  # jika waktu kurir selesai melayani customer ke-i+1 melebihi jam pulang kurir -> rute tidak valid
                or curr_time + wait_time < ready_time
            ):
                route_valid = False
                break
            else:
                curr_time += (
                    wait_time + service_time
                )  # rute valid & curr_time setelah melayani customer ke-i+1 diupdate

    if route_valid == True:
        # jika rute valid eta ditambah dengan eta antara last city ke depot
        curr_time += distance_matrix[route[-2]][route[-1]]  # from last city ke depot
        curr_time_dist += distance_matrix[route[-2]][route[-1]]
        if curr_time > fleet_total_time:
            # jika waktu kurir saat sampai ke depot melebihi jam pulang -> rute tidak valid
            route_valid = False

    return route_valid, curr_time_dist


def fitness_function_weighted_sum(alpha, beta, chromosome_distances, chromosome_routes):
    total_distance = sum(chromosome_distances)
    num_vehicles = len(chromosome_routes)
    fitness = (alpha * num_vehicles) + (
        beta * total_distance
    )  #  fitness untuk chromosome saat ini
    return fitness


def phase_two(
    alpha,
    beta,
    chromosome_routes,
    chromosome_distances,
    distance_matrix,
    customers,
    fleet_total_time,
    fleet_capacity,
):
    """
    In Phase 2, the last customer of each route ri , is relocated
    to become the first customer to route ri+1 . If this removal
    and insertion maintains feasibility for route ri+1, and the
    sum of costs of r1 and ri+1 at Phase 2 is less than sum of
    costs of ri + ri+1 at Phase 1, the routing configuration at
    Phase 2 is accepted, otherwise the network topology before
    Phase 2 (i.e., at Phase 1) is maintained.
    """
    indexes = [i for i in range(len(chromosome_routes) + 1)]
    indexes[len(chromosome_routes)] = (
        0  # buat masangin route last vehicle dg route first vehicle
    )

    for i in range(len(indexes) - 1):
        route_one_idx = indexes[i]
        route_two_idx = indexes[i + 1]
        route_one = chromosome_routes[route_one_idx].copy()
        route_two = chromosome_routes[route_two_idx].copy()
        distance_route_one = chromosome_distances[route_one_idx]
        distance_route_two = chromosome_distances[route_two_idx]
        last_customer_route_one = route_one[-2]
        route_two.insert(1, last_customer_route_one)
        is_route_two_valid, new_route_two_dist = is_route_valid(
            route_two, distance_matrix, customers, fleet_total_time, fleet_capacity
        )

        if is_route_two_valid == True:
            if len(route_one) > 3:
                new_route_one_dist = (
                    distance_route_one
                    - distance_matrix[route_one[-2]][0]
                    + distance_matrix[route_one[-3]][0]
                )
                if (new_route_one_dist + new_route_two_dist) < (
                    distance_route_one + distance_route_two
                ):
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
    fitness = fitness_function_weighted_sum(
        alpha, beta, chromosome_distances, chromosome_routes
    )
    return (
        chromosome_routes,
        chromosome_distances,
        num_vehicles,
        total_distance,
        fitness,
    )


def fast_non_dominated_sort_fitness(values1, values2):
    S = [[] for _ in range(len(values1))]
    front = [[]]
    n = [0 for _ in range(len(values1))]
    rank = [0 for _ in range(len(values1))]

    for p in range(len(values1)):
        S[p] = []  # S[p] isinya semua solusi yang didominasi/lebih buruk oleh solusi p
        n[p] = 0  # jumllah solusi lain yang mendominasi / lebih baik dari solusi p
        for q in range(len(values1)):
            if (
                (values1[p] < values1[q] and values2[p] < values2[q])
                or (values1[p] <= values1[q] and values2[p] < values2[q])
                or (values1[p] < values1[q] and values2[p] <= values2[q])
            ):
                S[p].append(q)  # p dominates q
            elif (
                (values1[q] < values1[p] and values2[q] < values2[p])
                or (values1[q] <= values1[p] and values2[q] < values2[p])
                or (values1[q] < values1[p] and values2[q] <= values2[p])
            ):
                n[
                    p
                ] += 1  #  ada solusi lain yang mendominasi solusi p, solusi lain yang mendominasi p ada n[p] buah
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
                n[q] -= 1  # n[q] solusi yang lebih baik dari q
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


def routing_phase_two(
    population_size,
    alpha,
    beta,
    population_routes,
    population_distances,
    distance_matrix,
    customers,
    fleet_total_time,
    fleet_capacity,
):

    results = []
    for i in range(population_size):
        results.append(
            phase_two(
                alpha,
                beta,
                population_routes[i],
                population_distances[i],
                distance_matrix,
                customers,
                fleet_total_time,
                fleet_capacity,
            )
        )

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

    return (
        new_population_routes,
        new_population_distances,
        new_population_num_routes,
        new_population_total_route_distances,
        fitnesses,
    )

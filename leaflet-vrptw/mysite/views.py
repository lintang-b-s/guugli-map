from django.shortcuts import render, redirect
from .vrptw_genetic_algorithm import GA_VRPTW
from .utils import IdMap
from datetime import datetime
import time
import requests
from datetime import timedelta
from django.http.response import JsonResponse


def home(request):
    context = {}
    date_format = "%Y-%m-%dT%H:%M"

    if request.method == "POST":
        depot_name = request.POST.get("depot_name")
        depot_lat_lng = request.POST.get("depot_lat_lng")
        depot_lat, depot_lon = depot_lat_lng.split(",")
        depot_lat = float(depot_lat)
        depot_lon = float(depot_lon)
        depot_ready_time = datetime.strptime(
            request.POST.get("depot_ready_time"), date_format
        )
        depot_due_time = datetime.strptime(
            request.POST.get("depot_due_time"), date_format
        )
        depot_capacity = float(request.POST.get("depot_capacity"))

        depot = {
            "fleet_lat": depot_lat,
            "fleet_lon": depot_lon,
            "ready_time": depot_ready_time,
            "fleet_max_working_time": (
                depot_due_time - depot_ready_time
            ).total_seconds()
            / 60,
            "fleet_capacity": depot_capacity,
            "fleet_size": 25,
        }

        number_of_customers = int(request.POST.get("number_of_customers", 0))
        customers = []
        id_map = IdMap()
        depot_id = id_map["depot"]
        for i in range(number_of_customers):
            customer_name = request.POST.get(f"customer_{i}_name")
            customer_lat_lng = request.POST.get(f"customer_{i}_lat_lng")
            customer_latitude, customer_longitude = customer_lat_lng.split(",")
            customer_latitude = float(customer_latitude)
            customer_longitude = float(customer_longitude)

            customer_demand = float(request.POST.get(f"customer_{i}_demand"))
            customer_ready_time = datetime.strptime(
                request.POST.get(f"customer_{i}_ready_time"), date_format
            )
            customer_due_time = datetime.strptime(
                request.POST.get(f"customer_{i}_due_time"), date_format
            )
            customer_service_time = float(
                request.POST.get(f"customer_{i}_service_time")
            )
            customers.append(
                {
                    "id": id_map[customer_name],
                    "lat": customer_latitude,
                    "lon": customer_longitude,
                    "demand": customer_demand,
                    "ready_time": (
                        customer_ready_time - depot_ready_time
                    ).total_seconds()
                    / 60,
                    "due_time": (customer_due_time - depot_ready_time).total_seconds()
                    / 60,
                    "service_time": customer_service_time,
                    "name": customer_name,
                }
            )

        sp_url = "http://localhost:5000/api/navigations/shortest-path"
        distance_matrix = {}
        navigations_matrix = {}

        distance_matrix[depot_id] = {}
        distance_matrix[depot_id][depot_id] = 0
        navigations_matrix[depot_id] = {}
        navigations_matrix[depot_id][depot_id] = []

        for customer in customers:
            for customer_pair in customers:
                if customer_pair == customer:
                    if customer["id"] not in distance_matrix:
                        distance_matrix[customer["id"]] = {}
                        navigations_matrix[customer["id"]] = {}
                    distance_matrix[customer["id"]][customer_pair["id"]] = 0
                    navigations_matrix[(customer["id"], customer_pair["id"])] = []
                    continue
                data = {
                    "src_lat": customer["lat"],
                    "src_lon": customer["lon"],
                    "dst_lat": customer_pair["lat"],
                    "dst_lon": customer_pair["lon"],
                }
                try:

                    response = requests.post(sp_url, json=data)
                    if response.status_code == 200:
                        eta = response.json()["ETA"]
                        navigations = response.json()["path"]
                        if customer["id"] not in distance_matrix:
                            distance_matrix[customer["id"]] = {}
                            navigations_matrix[customer["id"]] = {}
                        distance_matrix[customer["id"]][customer_pair["id"]] = eta
                        navigations_matrix[customer["id"]][
                            customer_pair["id"]
                        ] = navigations
                except requests.exceptions.RequestException as e:
                    print("Error request ke shortest-path API:", e)
        for customer in customers:
            data = {
                "src_lat": depot_lat,
                "src_lon": depot_lon,
                "dst_lat": customer["lat"],
                "dst_lon": customer["lon"],
            }
            data_reversed = {
                "src_lat": customer["lat"],
                "src_lon": customer["lon"],
                "dst_lat": depot_lat,
                "dst_lon": depot_lon,
            }
            try:
                response = requests.post(sp_url, json=data)
                if response.status_code == 200:
                    eta = response.json()["ETA"]
                    navigations = response.json()["path"]
                    distance_matrix[depot_id][customer["id"]] = eta
                    navigations_matrix[depot_id][customer["id"]] = navigations
                response = requests.post(sp_url, json=data_reversed)

                if response.status_code == 200:
                    eta = response.json()["ETA"]
                    navigations = response.json()["path"]
                    distance_matrix[customer["id"]][depot_id] = eta
                    navigations_matrix[customer["id"]][depot_id] = navigations
            except requests.exceptions.RequestException as e:
                print("Error request ke shortest-path API:", e)

        ga_population = GA_VRPTW()
        ga_population.create_population(
            customers_input=customers,
            fleets_input=[depot],
            distance_matrix=distance_matrix,
        )
        (
            best_solution_results,
            best_solution_routes,
            best_solution_distances,
            customers_service_time,
        ) = ga_population.solve()

        # best_solution_results =  {"num_vehicles": int, "total_distance": int, "fitness": float}
        vehicles_routes = []  # polyline untuk setiap vehicles
        vehicles_route_orders = []  # urutan customer yang dilayani oleh setiap vehicles

        for i in range(len(best_solution_routes)):
            curr_vehicle_route = best_solution_routes[i]
            vehicles_route_orders.append(curr_vehicle_route)

            curr_navigation = []  # polyline untuk vehicle ke-i
            for j in range(len(curr_vehicle_route)):
                if j == 0:
                    continue
                # navigasi dari customer ke-j-1 ke customer ke-j
                polyline = navigations_matrix[curr_vehicle_route[j - 1]][
                    curr_vehicle_route[j]
                ]
                curr_navigation.append(polyline)

            vehicles_routes.append(curr_navigation)

        context = {
            "depot_name": depot_name,
            "depot_lat_lng": depot_lat_lng,
            "depot_ready_time": depot_ready_time,
            "depot_due_time": depot_due_time,
            "depot_capacity": depot_capacity,
            "number_of_customers": number_of_customers,
            "customers": customers,
            "best_solution_results": best_solution_results,
            "vehicles_routes": vehicles_routes,
            "vehicles_route_orders": vehicles_route_orders,
            "best_solution_distances": best_solution_distances,
            "customers_service_time": customers_service_time,
        }
        return render(request, "index.html", context)
    return render(request, "index.html", context)


def vrptw_from_csv(request):
    context = {}
    date_format = "%Y-%m-%d %H:%M:%S"

    if request.method == "POST":
        id_map = IdMap()
        customers = []
        depot_id = id_map["depot"]
        fleets = {}
        fleet_due_time = 0
        try:
            csv_file_fleets = request.FILES["fleets-file"]
            csv_file_customers = request.FILES["customers-file"]
            if not csv_file_fleets.name.endswith(".csv"):
                print("File is not CSV type")

            if csv_file_fleets.multiple_chunks():
                print(
                    "Uploaded file is too big (%.2f MB)."
                    % (csv_file_fleets.size / (1000 * 1000),)
                )

            file_fleets_data = csv_file_fleets.read().decode("utf-8")
            file_customers_data = csv_file_customers.read().decode("utf-8")

            customer_lines = file_customers_data.split("\n")
            lines = file_fleets_data.split("\n")

            fleet_ready_time = 0
            for line in lines:
                fields = line.split(",")
                for i in range(len(fields)):
                    fields[i] = fields[i].lstrip()

                if fields[1] == "fleet_capacity":
                    continue
                fleet_ready_time = datetime.strptime(fields[-2], date_format)
                fleet_due_time = datetime.strptime(fields[-1], date_format)
                fleet_max_working_time = (
                    fleet_due_time - fleet_ready_time
                ).total_seconds() / 60

                fleets["fleet_capacity"] = float(fields[1])
                fleets["fleet_lon"] = float(fields[2])
                fleets["fleet_lat"] = float(fields[3])
                fleets["fleet_max_working_time"] = float(fleet_max_working_time)
                fleets["fleet_size"] = float(fields[0])
                fleets["ready_time"] = fleet_ready_time

            i = 0
            for line in customer_lines:
                fields = line.split(",")
                if fields[1] == "XC":
                    continue

                for j in range(len(fields)):
                    fields[j] = fields[j].lstrip()

                customer_ready_time_data = datetime.strptime(fields[4], date_format)
                customer_ready_time_from_depot = (
                    customer_ready_time_data - fleet_ready_time
                ).total_seconds() / 60
                customer_due_time = datetime.strptime(fields[5], date_format)
                fields[6] = fields[6].split(":")[1]
                customers.append(
                    {
                        "id": int(id_map[str(i)]),
                        "lat": float(fields[2]),
                        "lon": float(fields[1]),
                        "demand": float(fields[3]),
                        "ready_time": customer_ready_time_from_depot,
                        "due_time": (
                            customer_due_time - fleet_ready_time
                        ).total_seconds()
                        / 60,
                        "service_time": float(fields[6]),
                        "name": str(i),
                    }
                )
                i += 1
        except Exception as e:
            print("Unable to upload file. ", e)

        sp_url = "http://localhost:5000/api/navigations/shortest-path"
        distance_matrix = {}
        navigations_matrix = {}

        distance_matrix[depot_id] = {}
        distance_matrix[depot_id][depot_id] = 0
        navigations_matrix[depot_id] = {}
        navigations_matrix[depot_id][depot_id] = []

        for customer in customers:
            for customer_pair in customers:
                if customer_pair == customer:
                    if customer["id"] not in distance_matrix:
                        distance_matrix[customer["id"]] = {}
                        navigations_matrix[customer["id"]] = {}
                    distance_matrix[customer["id"]][customer_pair["id"]] = 0
                    navigations_matrix[(customer["id"], customer_pair["id"])] = []
                    continue
                data = {
                    "src_lat": customer["lat"],
                    "src_lon": customer["lon"],
                    "dst_lat": customer_pair["lat"],
                    "dst_lon": customer_pair["lon"],
                }
                try:

                    response = requests.post(sp_url, json=data)
                    if response.status_code == 200:
                        eta = response.json()["ETA"]
                        navigations = response.json()["path"]
                        if customer["id"] not in distance_matrix:
                            distance_matrix[customer["id"]] = {}
                            navigations_matrix[customer["id"]] = {}
                        distance_matrix[customer["id"]][customer_pair["id"]] = eta
                        navigations_matrix[customer["id"]][
                            customer_pair["id"]
                        ] = navigations
                except requests.exceptions.RequestException as e:
                    print("Error request ke shortest-path API:", e)

        for customer in customers:
            data = {
                "src_lat": fleets["fleet_lat"],
                "src_lon": fleets["fleet_lon"],
                "dst_lat": customer["lat"],
                "dst_lon": customer["lon"],
            }
            data_reversed = {
                "src_lat": customer["lat"],
                "src_lon": customer["lon"],
                "dst_lat": fleets["fleet_lat"],
                "dst_lon": fleets["fleet_lon"],
            }
            try:
                response = requests.post(sp_url, json=data)
                if response.status_code == 200:
                    eta = response.json()["ETA"]
                    navigations = response.json()["path"]
                    distance_matrix[depot_id][customer["id"]] = eta
                    navigations_matrix[depot_id][customer["id"]] = navigations
                response = requests.post(sp_url, json=data_reversed)

                if response.status_code == 200:
                    eta = response.json()["ETA"]
                    navigations = response.json()["path"]
                    distance_matrix[customer["id"]][depot_id] = eta
                    navigations_matrix[customer["id"]][depot_id] = navigations
            except requests.exceptions.RequestException as e:
                print("Error request ke shortest-path API:", e)

        start_time = time.time()
        ga_population = GA_VRPTW()
        ga_population.create_population(
            customers_input=customers,
            fleets_input=[fleets],
            distance_matrix=distance_matrix,
        )

        (
            best_solution_results,
            best_solution_routes,
            best_solution_distances,
            customers_service_time,
        ) = ga_population.solve()
        end_time = time.time()
        runtime = end_time - start_time
        print(f"lama waktu solve: {runtime/ 60}")
        # aneh rute cuma 1 vehicle tapi distances nya lebih dari satu vehicle

        # best_solution_results =  {"num_vehicles": int, "total_distance": int, "fitness": float}
        vehicles_routes = []  # polyline untuk setiap vehicles
        vehicles_route_orders = []  # urutan customer yang dilayani oleh setiap vehicles

        for i in range(len(best_solution_routes)):
            curr_vehicle_route = best_solution_routes[i]
            vehicles_route_orders.append(curr_vehicle_route)

            curr_navigation = []  # polyline untuk vehicle ke-i
            for j in range(1, len(curr_vehicle_route)):

                # navigasi dari customer ke-j-1 ke customer ke-j
                polyline = navigations_matrix[curr_vehicle_route[j - 1]][
                    curr_vehicle_route[j]
                ]
                curr_navigation.append(polyline)

            vehicles_routes.append(curr_navigation)
        depot_lat_lng = f"{fleets['fleet_lat']}, {fleets['fleet_lon']}"

        # ubah date customer lagi...
        for i in range(len(customers)):
            customers[i]["ready_time"] = fleet_ready_time + timedelta(
                seconds=customers[i]["ready_time"] * 60
            )
            customers[i]["due_time"] = fleet_ready_time + timedelta(
                seconds=customers[i]["due_time"] * 60
            )

        context = {
            "depot_name": "depot",
            "depot_lat_lng": depot_lat_lng,
            "depot_ready_time": fleets["ready_time"],
            "depot_due_time": fleet_due_time,
            "depot_capacity": fleets["fleet_capacity"],
            "number_of_customers": len(customers),
            "customers": customers,
            "best_solution_results": best_solution_results,
            "vehicles_routes": vehicles_routes,
            "vehicles_route_orders": vehicles_route_orders,
            "best_solution_distances": best_solution_distances,
            "customers_service_time": customers_service_time,
        }
        # return render(request, "index.html", context)
        return JsonResponse(context)

    return "ok"

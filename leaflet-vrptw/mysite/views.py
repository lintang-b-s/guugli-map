from django.shortcuts import render, redirect
from .vrptw_genetic_algorithm import GA_VRPTW
from .utils import IdMap
from datetime import datetime
import requests


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
            "ready_time": 0,
            "fleet_max_working_time": (
                depot_due_time - depot_ready_time
            ).total_seconds(),
            "fleet_capacity": depot_capacity,
            "fleet_size": 25,
        }

        number_of_customers = int(request.POST.get("number_of_customers", 0))
        customers = []
        id_map = IdMap()
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
                    ).total_seconds(),
                    "due_time": (customer_due_time - depot_ready_time).total_seconds(),
                    "service_time": customer_service_time,
                }
            )

        sp_url = "http://localhost:5000/api/navigations/shortest-path"
        distance_matrix = {}
        for customer in customers:
            for customer_pair in customers:
                data = {
                    "src_lat": customer["lat"],
                    "src_lon": customer["lon"],
                    "dst_lat": customer_pair["lat"],
                    "dst_lon": customer_pair["lon"],
                }
                response = requests.post(sp_url, json=data)
                if response.status_code == 200:
                    eta = response.json()["ETA"]
                    if customer["id"] not in distance_matrix:
                        distance_matrix[customer["id"]] = {}
                    distance_matrix[customer["id"]][customer_pair["id"]] = eta

        ga_population = GA_VRPTW()
        ga_population.create_population(
            customers_input=customers,
            fleets_input=[depot],
            distance_matrix=distance_matrix,
        )
        best_solution_results, best_solution_routes, best_solution_distances = (
            ga_population.solve()
        )
        

        return redirect("home")
    return render(request, "index.html", context)

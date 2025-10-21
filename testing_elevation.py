import asyncio
from mcpuniverse.evaluator.google_maps.functions import (
    google_maps_validate_elevation_meters_of_scenic_viewpoints,
)

payload = {
    "routes": [
        {
            "scenic_viewpoints": [
                {
                    "name": "Pontian",
                    "city": "Segamat",
                    "address": "Jalan Kota, Bandar Hilir, 75000 Melaka, Malaysia",
                    "elevation_meters": 195.0,  # replace with the model’s reported elevation
                }
            ]
        }
    ]
}

async def main():
    passed, reason = await google_maps_validate_elevation_meters_of_scenic_viewpoints(payload)
    print(f"passed={passed}")
    if reason:
        print(f"reason: {reason}")

asyncio.run(main())

# Reason: the returned elevation of Bukit Soga Perdana in Batu Pahat at 83000 Batu Pahat, Johor, Malaysia is 76.00, while gt = 9.06


# {
#       "name": "Bukit Soga Perdana",
#       "formatted_address": "83000 Batu Pahat, Johor, Malaysia",
#       "location": {
#         "lat": 1.8488881,
#         "lng": 102.960787
#       },
#       "place_id": "ChIJG2b0EfNX0DERgDpO03BVjwQ",
#       "rating": 4.7,
#       "types": [
#         "point_of_interest",
#         "establishment"
#       ]
#     },
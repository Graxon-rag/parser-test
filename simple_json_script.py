import json
import random
from datetime import datetime, timedelta

NUM_RECORDS = 1000

categories = ["Electronics", "Clothing", "Furniture", "Books", "Sports"]

brands = {
    "Electronics": ["Apple", "Samsung", "Sony"],
    "Clothing": ["Nike", "Adidas", "Zara"],
    "Furniture": ["IKEA", "Pepperfry", "Urban Ladder"],
    "Books": ["Penguin", "Oxford", "Harper"],
    "Sports": ["Yonex", "Decathlon", "Adidas"],
}

cities = ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad"]

data = []

for i in range(NUM_RECORDS):
    category = random.choice(categories)

    item = {
        "id": f"ORD-{i:06d}",
        "created_at": (
            datetime.now() - timedelta(days=random.randint(0, 365))
        ).isoformat(),

        "customer": {
            "customer_id": f"CUST-{random.randint(1000, 9999)}",
            "name": f"Customer {random.randint(1, 500)}",
            "email": f"user{i}@example.com",

            "address": {
                "street": f"{random.randint(1, 500)} Main Road",
                "city": random.choice(cities),
                "country": "India",
                "postal_code": str(random.randint(100000, 999999))
            }
        },

        "order": {
            "category": category,
            "brand": random.choice(brands[category]),
            "price": random.randint(1000, 100000),
            "quantity": random.randint(1, 5),

            "items": [
                {
                    "sku": f"SKU-{random.randint(10000,99999)}",
                    "name": f"Product {random.randint(1,1000)}",
                    "price": random.randint(100, 5000)
                }
                for _ in range(random.randint(1, 4))
            ]
        },

        "payment": {
            "method": random.choice([
                "card",
                "upi",
                "netbanking",
                "wallet"
            ]),
            "status": random.choice([
                "success",
                "pending",
                "failed"
            ]),
            "transaction_id": f"TXN-{random.randint(1000000,9999999)}"
        },

        "metadata": {
            "source": "test_generator",
            "version": "1.0",

            "tags": random.sample(
                ["premium", "discount", "sale", "featured", "bulk"],
                k=random.randint(1, 3)
            ),

            "analytics": {
                "view_count": random.randint(0, 10000),
                "click_count": random.randint(0, 1000),
                "conversion_rate": round(random.random(), 4)
            }
        }
    }

    data.append(item)

with open("nested_test.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Generated nested_test.json with {NUM_RECORDS} records")

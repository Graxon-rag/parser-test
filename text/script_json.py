import json
import random

categories = {
    "Electronics": {
        "brands": ["Apple", "Samsung", "Sony", "LG", "Dell"],
        "products": ["Phone", "Laptop", "TV", "Tablet", "Monitor"],
    },
    "Clothing": {
        "brands": ["Nike", "Adidas", "Puma", "Zara", "H&M"],
        "products": ["T-Shirt", "Jeans", "Jacket", "Shorts", "Shoes"],
    },
    "Furniture": {
        "brands": ["IKEA", "Urban Ladder", "Pepperfry", "Godrej"],
        "products": ["Chair", "Table", "Sofa", "Bed", "Desk"],
    },
    "Books": {
        "brands": ["Penguin", "Harper", "Oxford", "Pearson"],
        "products": ["Novel", "Textbook", "Biography", "Dictionary"],
    },
    "Sports": {
        "brands": ["Yonex", "Decathlon", "Adidas", "Nike"],
        "products": ["Bat", "Racket", "Football", "Basketball", "Helmet"],
    },
}

data = []

for category_name, category_data in categories.items():
    category = {
        "category": category_name,
        "brands": []
    }

    for brand in category_data["brands"]:
        brand_obj = {
            "brand": brand,
            "products": []
        }

        for product_type in category_data["products"]:
            product = {
                "name": f"{brand} {product_type}",
                "variants": []
            }

            for variant_id in range(1, 4):
                variant = {
                    "variant_id": variant_id,
                    "price": random.randint(500, 100000),
                    "rating": round(random.uniform(3.0, 5.0), 1),
                    "reviews": []
                }

                for review_id in range(1, 4):
                    variant["reviews"].append({
                        "review_id": review_id,
                        "user": f"user_{review_id}",
                        "rating": round(random.uniform(3.0, 5.0), 1),
                        "comment": (
                            f"This {product_type.lower()} from "
                            f"{brand} is very good."
                        )
                    })

                product["variants"].append(variant)

            brand_obj["products"].append(product)

        category["brands"].append(brand_obj)

    data.append(category)

with open("nested_products.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Generated nested_products.json")

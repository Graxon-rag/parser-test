
import random
import yaml

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

SHEETS = {
    "Products_US": 500,
    "Products_EU": 500,
    "Products_India": 500,
    "Products_APAC": 500,
}

catalog = {}

global_id = 0

for region, count in SHEETS.items():
    records = []

    for _ in range(count):
        category = random.choice(list(categories.keys()))
        brand = random.choice(categories[category]["brands"])
        product = random.choice(categories[category]["products"])

        price = random.randint(500, 100000)
        rating = round(random.uniform(3.0, 5.0), 1)
        stock = random.randint(0, 500)

        records.append({
            "ProductID": f"P{global_id:06d}",
            "Name": f"{brand} {product}",
            "Brand": brand,
            "Category": category,
            "Price": price,
            "Rating": rating,
            "Stock": stock,
            "Description": (
                f"High quality {product.lower()} from {brand} "
                f"in the {category.lower()} category with rating {rating}"
            ),
        })

        global_id += 1

    catalog[region] = records

with open("test_products.yaml", "w", encoding="utf-8") as f:
    yaml.dump(catalog, f, sort_keys=False, allow_unicode=True)

print("Generated test_products.yaml")
print(f"Total records: {global_id}")

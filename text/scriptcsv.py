import csv
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

with open("test.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "ProductID",
        "Name",
        "Brand",
        "Category",
        "Price",
        "Rating",
        "Stock",
        "Description"
    ])

    for i in range(1250):
        category = random.choice(list(categories.keys()))

        brand = random.choice(categories[category]["brands"])
        product_type = random.choice(categories[category]["products"])

        name = f"{brand} {product_type}"

        price = random.randint(500, 100000)
        rating = round(random.uniform(3.0, 5.0), 1)
        stock = random.randint(0, 500)

        description = (
            f"High quality {product_type.lower()} from {brand} "
            f"in the {category.lower()} category with rating {rating}"
        )

        writer.writerow([
            f"P{i:05d}",
            name,
            brand,
            category,
            price,
            rating,
            stock,
            description
        ])

print("Generated test.csv with 1250 rows")
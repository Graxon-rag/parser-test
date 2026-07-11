from openpyxl import Workbook
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

SHEETS = {
    "Products_US": 500,
    "Products_EU": 500,
    "Products_India": 500,
    "Products_APAC": 500,
}

wb = Workbook()

# Remove default sheet
wb.remove(wb.active)

headers = [
    "ProductID",
    "Name",
    "Brand",
    "Category",
    "Price",
    "Rating",
    "Stock",
    "Description",
]

global_id = 0

for sheet_name, row_count in SHEETS.items():
    ws = wb.create_sheet(title=sheet_name)

    ws.append(headers)

    for _ in range(row_count):
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

        ws.append([
            f"P{global_id:06d}",
            name,
            brand,
            category,
            price,
            rating,
            stock,
            description,
        ])

        global_id += 1

output_file = "test_multisheet.xlsx"
wb.save(output_file)

print(f"Generated {output_file}")
print(f"Sheets: {list(SHEETS.keys())}")
print(f"Total rows: {sum(SHEETS.values())}")

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

global_id = 0

html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Product Catalog</title>
<style>
body{
    font-family:Arial,sans-serif;
    margin:40px;
}
table{
    border-collapse:collapse;
    width:100%;
    margin-bottom:50px;
}
th,td{
    border:1px solid #ccc;
    padding:8px;
}
th{
    background:#f4f4f4;
}
h2{
    margin-top:40px;
}
</style>
</head>
<body>

<h1>Generated Product Catalog</h1>
"""

for sheet_name, rows in SHEETS.items():

    html += f"<h2>{sheet_name}</h2>\n"
    html += "<table>\n"
    html += """
<tr>
<th>ProductID</th>
<th>Name</th>
<th>Brand</th>
<th>Category</th>
<th>Price</th>
<th>Rating</th>
<th>Stock</th>
<th>Description</th>
</tr>
"""

    for _ in range(rows):
        category = random.choice(list(categories.keys()))
        brand = random.choice(categories[category]["brands"])
        product = random.choice(categories[category]["products"])

        price = random.randint(500, 100000)
        rating = round(random.uniform(3.0, 5.0), 1)
        stock = random.randint(0, 500)

        html += f"""
<tr>
<td>P{global_id:06d}</td>
<td>{brand} {product}</td>
<td>{brand}</td>
<td>{category}</td>
<td>{price}</td>
<td>{rating}</td>
<td>{stock}</td>
<td>High quality {product.lower()} from {brand} in the {category.lower()} category with rating {rating}</td>
</tr>
"""

        global_id += 1

    html += "</table>\n"

html += """
</body>
</html>
"""

with open("test_products.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Generated test_products.html")

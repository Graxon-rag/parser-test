import random
import xml.etree.ElementTree as ET
from xml.dom import minidom

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

root = ET.Element("ProductCatalog")

global_id = 0

for sheet_name, rows in SHEETS.items():

    sheet = ET.SubElement(root, "Sheet")
    sheet.set("name", sheet_name)

    for _ in range(rows):

        category = random.choice(list(categories.keys()))
        brand = random.choice(categories[category]["brands"])
        product = random.choice(categories[category]["products"])

        price = random.randint(500, 100000)
        rating = round(random.uniform(3.0, 5.0), 1)
        stock = random.randint(0, 500)

        item = ET.SubElement(sheet, "Product")

        ET.SubElement(item, "ProductID").text = f"P{global_id:06d}"
        ET.SubElement(item, "Name").text = f"{brand} {product}"
        ET.SubElement(item, "Brand").text = brand
        ET.SubElement(item, "Category").text = category
        ET.SubElement(item, "Price").text = str(price)
        ET.SubElement(item, "Rating").text = str(rating)
        ET.SubElement(item, "Stock").text = str(stock)
        ET.SubElement(item, "Description").text = (
            f"High quality {product.lower()} from {brand} "
            f"in the {category.lower()} category with rating {rating}"
        )

        global_id += 1

xml_bytes = ET.tostring(root, encoding="utf-8")
pretty_xml = minidom.parseString(xml_bytes).toprettyxml(indent="    ")

with open("test_products.xml", "w", encoding="utf-8") as f:
    f.write(pretty_xml)

print("Generated test_products.xml")

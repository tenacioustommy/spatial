"""
Get valid object candidates
"""

from tdw.librarian import ModelLibrarian

librarian = ModelLibrarian("models_core.json")
chair_records = []
for record in librarian.records:
    if "chair" in record.name.lower():
        chair_records.append(record.name)


from tdw.librarian import ModelLibrarian

librarian = ModelLibrarian("models_core.json")
for record in librarian.records:
    print(record.name)
import json
import os
os.makedirs("data/json_files",exist_ok=True)
json_data = {
    "company": "TechCorp",
    "employees": [
        {
            "id": 1,
            "name": "John Doe",
            "role": "Software Engineer",
            "skills": ["Python", "JavaScript", "React"],
            "projects": [
                {"name": "RAG System", "status": "In Progress"},
                {"name": "Data Pipeline", "status": "Completed"}
            ]
        },
        {
            "id": 2,
            "name": "Jane Smith",
            "role": "Data Scientist",
            "skills": ["Python", "Machine Learning", "SQL"],
            "projects": [
                {"name": "ML Model", "status": "In Progress"},
                {"name": "Analytics Dashboard", "status": "Planning"}
            ]
        }
    ],
    "departments": {
        "engineering": {
            "head": "Mike Johnson",
            "budget": 1000000,
            "team_size": 25
        },
        "data_science": {
            "head": "Sarah Williams",
            "budget": 750000,
            "team_size": 15
        }
    }
}
# print(json_data)
with open('data/json_files/company_data.json','w') as f:
    json.dump(json_data, f, indent=2)

jsonl_data = [
    {"timestamp": "2024-01-01", "event": "user_login", "user_id": 123},
    {"timestamp": "2024-01-01", "event": "page_view", "user_id": 123, "page": "/home"},
    {"timestamp": "2024-01-01", "event": "purchase", "user_id": 123, "amount": 99.99}
]

with open('data/json_files/events.jsonl', 'w') as f:
    for item in jsonl_data:
        f.write(json.dumps(item) + '\n')

## JSON PROCESSING STARTERGY
from langchain_community.document_loaders import JSONLoader
import json

## MEthod1 : JsonLoader With jq_schema
print("1️⃣ JSONLoader - Extract specific fields")

# Extract employee information
employee_loader = JSONLoader(
    file_path='data/json_files/company_data.json',
    jq_schema='.employees[]',  # jq query to extract each employee
    text_content=False  # Get full JSON objects
)

employee_docs = employee_loader.load()
print(f"Loaded {len(employee_docs)} employee documents")
print(f"First employee: {employee_docs[0].page_content[:200]}...")
print(employee_docs)



## SQL DATABASES
import sqlite3
import os

os.makedirs("data/databases",exist_ok=True)

## create a sample database
conn=sqlite3.connect("data/databases/company.db")
cursor=conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS employees
                 (id INTEGER PRIMARY KEY, name TEXT, role TEXT, department TEXT, salary REAL)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS projects
                 (id INTEGER PRIMARY KEY, name TEXT, status TEXT, budget REAL, lead_id INTEGER)''')

employees = [
    (1, 'John Doe', 'Senior Developer', 'Engineering', 95000),
    (2, 'Jane Smith', 'Data Scientist', 'Analytics', 105000),
    (3, 'Mike Johnson', 'Product Manager', 'Product', 110000),
    (4, 'Sarah Williams', 'DevOps Engineer', 'Engineering', 98000)
]

projects = [
    (1, 'RAG Implementation', 'Active', 150000, 1),
    (2, 'Data Pipeline', 'Completed', 80000, 2),
    (3, 'Customer Portal', 'Planning', 200000, 3),
    (4, 'ML Platform', 'Active', 250000, 2)
]


cursor.executemany('INSERT OR REPLACE INTO employees VALUES(?,?,?,?,?)',employees)
cursor.executemany('INSERT OR REPLACE INTO employees VALUES(?,?,?,?,?)',projects)

cursor.execute("Select * from employees")

conn.commit()
conn.close()

# database content extraction
from langchain_community.utilities import SQLDatabase
from langchain_community.document_loaders import SQLDatabaseLoader

#METHOD 1 SQLDATABASE utility
db=SQLDatabase.from_uri("sqlite:///data/databases/company.db")

 # get database info
print(f"Tables: {db.get_usable_table_names()}")
print(f"\nTable DDl:")
print(db.get_table_info())
from typing import List
from langchain_core.documents import Document

# Method 2: Custom SQL to Document conversion
print("\n2️⃣ Custom SQL Processing")


def sql_to_documents(db_path: str) -> List[Document]:
    """Convert SQL Database To documents with context"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    documents = []
    # Strategy 1: Create documents for each table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]

        # Get table schema
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]

        # Get table data
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        # Create table overview document
        table_content = f"Table: {table_name}\n"
        table_content += f"Columns: {', '.join(column_names)}\n"
        table_content += f"Total Records: {len(rows)}\n\n"

        # Add sample records
        table_content += "Sample Records:\n"
        for row in rows[:5]:  # First 5 records
            record = dict(zip(column_names, row))
            table_content += f"{record}\n"

        doc = Document(
            page_content=table_content,
            metadata={
                'source': db_path,
                'table_name': table_name,
                'num_records': len(rows),
                'data_type': 'sql_table'
            }
        )
        documents.append(doc)

    # Strategy 2: Create relationship documents
    # Example: Join employees and projects
    cursor.execute("""
        SELECT e.name, e.role, p.name as project_name, p.status
        FROM employees e
        JOIN projects p ON e.id = p.lead_id
    """)

    relationships = cursor.fetchall()
    rel_content = "Employee-Project Relationships:\n\n"
    for rel in relationships:
        rel_content += f"{rel[0]} ({rel[1]}) leads {rel[2]} - Status: {rel[3]}\n"

    rel_doc = Document(
        page_content=rel_content,
        metadata={
            'source': db_path,
            'data_type': 'sql_relationships',
            'query': 'employee_project_join'
        }
    )
    documents.append(rel_doc)

    conn.close()
    return documents



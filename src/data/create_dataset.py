import openai
import os
import json
from dotenv import load_dotenv
from src.sql.sql_executor import SQLExecutor

load_dotenv()

SQL_CONNECTION_STRING = os.getenv("SQL_CONNECTION_STRING")

class SyntheticDatasetGenerator:
    def __init__(self):
       self.llm = "gpt-3.5-turbo"

        
    def generate_examples(self, query_type: str, num_examples: int = 15):
        # Create the prompt with the specific query type
        prompt_template = """
        You are an expert SQL developer helping to create training data for a text-to-SQL system. 
        
        DATABASE SCHEMA:
        The database contains the following tables:
        
        1. Customers
        - CustomerID (text, primary key)
        - CompanyName (text)
        - ContactName (text)
        - ContactTitle (text)
        - Address (text)
        - City (text)
        - Region (text)
        - PostalCode (text)
        - Country (text)
        - Phone (text)
        - Fax (text)
        
        2. Orders
        - OrderID (integer, primary key)
        - CustomerID (text, foreign key to Customers.CustomerID)
        - EmployeeID (integer)
        - OrderDate (date)
        - RequiredDate (date)
        - ShippedDate (date)
        - ShipVia (integer)
        - Freight (numeric)
        - ShipName (text)
        - ShipAddress (text)
        - ShipCity (text)
        - ShipRegion (text)
        - ShipPostalCode (text)
        - ShipCountry (text)
        
        3. OrderDetails
        - OrderID (integer, foreign key to Orders.OrderID, part of composite primary key)
        - ProductID (integer, foreign key to Products.ProductID, part of composite primary key)
        - UnitPrice (numeric)
        - Quantity (integer)
        - Discount (real)
        
        4. Products
        - ProductID (integer, primary key)
        - ProductName (text)
        - SupplierID (integer)
        - CategoryID (integer, foreign key to Categories.CategoryID)
        - QuantityPerUnit (text)
        - UnitPrice (numeric)
        - UnitsInStock (integer)
        - UnitsOnOrder (integer)
        - ReorderLevel (integer)
        - Discontinued (integer)
        
        5. Categories
        - CategoryID (integer, primary key)
        - CategoryName (text)
        - Description (text)
        - Picture (binary)
        
        RELATIONSHIPS:
        - Customers have many Orders (CustomerID)
        - Orders have many OrderDetails (OrderID)
        - Products have many OrderDetails (ProductID)
        - Categories have many Products (CategoryID)
        
        TASK:
        Generate {num_examples} unique examples of natural language questions about this database and the corresponding MS SQL query that correctly answers those questions.
        The natural language questions should be something a business user might ask, and the SQL query should follow MS SQL syntax.

        The questions should cover diverse business scenarios such as:
        - Customer Analysis:
            * Customer demographics and locations
            * Customer order history and preferences
            * Top customers by order volume or revenue
            * Customer contact information lookups
        
        - Sales Analysis:
            * Sales trends over time
            * Revenue by product/category
            * Order volumes and frequencies
            * Shipping patterns and delivery times
        
        - Product Analysis:
            * Product inventory levels
            * Popular products and categories
            * Product pricing analysis
            * Discontinued products
        
        - Geographic Analysis:
            * Sales by region/country
            * Customer distribution
            * Shipping destinations
            * Regional performance
        
        - Operational Metrics:
            * Order processing times
            * Shipping efficiency
            * Employee performance
            * Inventory management
        
        Format your response as a list of JSON objects with "input" and "output" fields:
        
        [
        {{
            "input": "Your first natural language question here",
            "output": "Your first MS SQL query here"
        }},
        {{
            "input": "Your second natural language question here",
            "output": "Your second MS SQL query here"
        }}
        ]
        
        Make sure your query:
        1. Uses the correct MS SQL syntax. 
        2. Uses the exact table and column names from the schema
        3. Does not include any newlines or backslashes, just spaces
        4. Includes JOINs where necessary
        5. Uses aliases for readability when appropriate
        6. Is optimized for better performance
        7. Ensure each query is unique and not similar to the others
        8. Ensure you use TOP instead of LIMIT as this is MS SQL syntax
        
        Generate {num_examples} questions that require {query_type} in the query. Ensure final output is a list with no ```json 
        """
        prompt = prompt_template.format(query_type=query_type, num_examples=num_examples)
        
        # Call the API
        try:
            response = openai.ChatCompletion.create(
                model=self.llm,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates SQL examples. Your responses must be valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,
                max_tokens=2000
            )
            
            # Extract the content from the response
            content = response.choices[0].message.content.strip()
            
            # Try to parse the JSON
            return self._parse_json_response(content)
            
        except Exception as e:
            print(f"Error generating examples for {query_type}: {str(e)}")
            return []
    
    def _clean_sql_response(self, content: str) -> str:
        """Clean up the SQL response to make it valid JSON."""
        # Remove backslash line continuations and convert to proper newlines
        content = content.replace('\\\n', ' ')
        content = content.replace('\\', '')
        return content
    
    def _parse_json_response(self, content: str):
        try:
            cleaned_content = self._clean_sql_response(content)
            result = json.loads(cleaned_content)
            return result
        except json.JSONDecodeError as e:
            print("JSON Error:", str(e))
            return {"question": "", "query": ""}


if __name__ == "__main__":
    # initialize the generator
    generator = SyntheticDatasetGenerator()
    
    # initialize the SQL executor
    sql_executor = SQLExecutor(SQL_CONNECTION_STRING)
    
    # define the query types
    query_types = [
        "a simple SELECT",
        "filtering with WHERE conditions",
        "aggregation (COUNT, SUM, AVG, etc.)",
        "multiple table JOINs",
        "sorting with ORDER BY",
        "grouping with GROUP BY",
        "filtering aggregated results with HAVING",
    ]
    
    # Create or open the output file in write mode
    with open("src/data/synthetic_examples.jsonl", "w") as f:
        for query_type in query_types:
            examples = generator.generate_examples(query_type, 15)
            
            # Validate and write each example individually
            for example in examples:
                if sql_executor.validate_query(example["output"]):
                    # Write each example as a single line
                    f.write(json.dumps(example) + "\n")
                else:
                    print(f"Invalid query: {example['input']}/n{example['output']}/n")
    
    # close the SQL executor
    sql_executor.disconnect()
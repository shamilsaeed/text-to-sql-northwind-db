# text-to-sql-northwind-db
A text to SQL app to help users runs natural language queries on the Microsoft's Northwind database


Setup: 

1. Downloaded the SQL Express and SSMS: https://www.microsoft.com/en-us/sql-server/sql-server-downloads

2. Created a new database called "Northwind"

3. Downloaded the Northwind database from: https://github.com/microsoft/sql-server-samples/blob/master/samples/databases/northwind-pubs/instnwnd.sql

4. Ran the instnwnd.sql file to get the Northwind database

5. Created a subset of the Northwind database using the Customer/Orders/Products/Categories/Order Details tables using the create_northwind_db.sql file
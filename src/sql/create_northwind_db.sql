-- Create the new Northwind database
CREATE DATABASE Northwind;
GO

-- Switch to the new database
USE Northwind;
GO

-- Copy the five selected tables with their data
-- 1. Categories table
SELECT * INTO Northwind.dbo.Categories FROM master.dbo.Categories;

-- 2. Products table
SELECT * INTO Northwind.dbo.Products FROM master.dbo.Products;

-- 3. Customers table
SELECT * INTO Northwind.dbo.Customers FROM master.dbo.Customers;

-- 4. Orders table
SELECT * INTO Northwind.dbo.Orders FROM master.dbo.Orders;

-- 5. Order Details table
SELECT * INTO Northwind.dbo.[Order Details] FROM master.dbo.[Order Details];

-- Now recreate the primary keys
ALTER TABLE Categories ADD CONSTRAINT PK_Categories PRIMARY KEY (CategoryID);
ALTER TABLE Products ADD CONSTRAINT PK_Products PRIMARY KEY (ProductID);
ALTER TABLE Customers ADD CONSTRAINT PK_Customers PRIMARY KEY (CustomerID);
ALTER TABLE Orders ADD CONSTRAINT PK_Orders PRIMARY KEY (OrderID);
ALTER TABLE [Order Details] ADD CONSTRAINT PK_OrderDetails PRIMARY KEY (OrderID, ProductID);

-- Recreate the foreign key relationships
ALTER TABLE Products ADD CONSTRAINT FK_Products_Categories 
    FOREIGN KEY (CategoryID) REFERENCES Categories (CategoryID);

ALTER TABLE Orders ADD CONSTRAINT FK_Orders_Customers 
    FOREIGN KEY (CustomerID) REFERENCES Customers (CustomerID);

ALTER TABLE [Order Details] ADD CONSTRAINT FK_OrderDetails_Orders 
    FOREIGN KEY (OrderID) REFERENCES Orders (OrderID);

ALTER TABLE [Order Details] ADD CONSTRAINT FK_OrderDetails_Products 
    FOREIGN KEY (ProductID) REFERENCES Products (ProductID);

-- Creating  indexes for better performance
CREATE INDEX IX_Products_CategoryID ON Products(CategoryID);
CREATE INDEX IX_Orders_CustomerID ON Orders(CustomerID);
CREATE INDEX IX_OrderDetails_ProductID ON [Order Details](ProductID);

-- Update statistics for query optimization
EXEC sp_updatestats;
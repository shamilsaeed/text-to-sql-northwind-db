import pyodbc
from typing import Tuple, Optional


class SQLExecutor:
    def __init__(self, conn_str: str):
        """Initialize SQL executor with connection string.
        
        Args:
            conn_str: SQL Server connection string
        """
        self.conn_str = conn_str
        self.conn = None
        self.cursor = None
        self._connect()

    def _connect(self) -> bool:
        """Establish database connection.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.conn = pyodbc.connect(self.conn_str)
            self.cursor = self.conn.cursor()
            print("SQL db connection successful!")
            return True
        except Exception as e:
            print(f"Connection failed: {str(e)}. Please check your connection string.")
            return False
        
    def disconnect(self):
        """Close database connection and cursor."""
        try:
            if self.cursor:
                self.cursor.close()
                self.cursor = None
                
            if self.conn:
                self.conn.close()
                self.conn = None
                
            print("Database connection closed successfully")
        except Exception as e:
            print(f"Error disconnecting from database: {str(e)}")

    def validate_query(self, sql_query: str) -> bool:
        """Validate SQL query.
        
        Args:
            sql_query: SQL query to validate
        """
        try:
            self.cursor.execute(sql_query)
            return True
        except Exception as e:
            return False
    
    def execute_query(self, sql_query: str) -> Tuple[bool, Optional[list], Optional[str]]:
        """Execute SQL query and return results.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            Tuple[bool, Optional[list], Optional[str]]: (success, results, error_message)
        """

        try:
            # Execute the query
            self.cursor.execute(sql_query)
            
            # Fetch results if it's a SELECT query
            if sql_query.strip().upper().startswith('SELECT'):
                results = self.cursor.fetchall()
                # Convert results to list of dictionaries
                columns = [column[0] for column in self.cursor.description]
                results = [dict(zip(columns, row)) for row in results]
                return True, results, None
            
            return True, None, None

        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            if self.conn:
                self.conn.rollback()
            return False, None, error_msg

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        
        
import sqlite3

def view_users():
    """Display all registered users in the database"""
    try:
        # Connect to the database
        conn = sqlite3.connect('database.db')
        conn.row_factory = sqlite3.Row
        
        # Check if the users table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cursor.fetchone():
            print("The users table doesn't exist yet. No users have registered.")
            return
        
        # Query all users
        users = conn.execute('SELECT id, username, email, created_at FROM users').fetchall()
        
        if not users:
            print("No users are registered in the database yet.")
            return
        
        # Display users
        print("\n===== REGISTERED USERS =====")
        print(f"{'ID':<5} {'Username':<20} {'Email':<30} {'Created At':<20}")
        print("-" * 75)
        
        for user in users:
            print(f"{user['id']:<5} {user['username']:<20} {user['email']:<30} {user['created_at']:<20}")
        
        print("\nTotal users:", len(users))
        
    except Exception as e:
        print(f"Error accessing database: {str(e)}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    view_users()

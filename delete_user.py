import sqlite3
import sys

def delete_user(user_id=None, username=None, email=None):
    """
    Delete a user from the database by ID, username, or email.
    At least one parameter must be provided.
    """
    if not any([user_id, username, email]):
        print("Error: You must provide at least one of: user_id, username, or email")
        return False
    
    try:
        # Connect to the database
        conn = sqlite3.connect('database.db')
        conn.row_factory = sqlite3.Row
        
        # Check if the users table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cursor.fetchone():
            print("The users table doesn't exist yet. No users to delete.")
            return False
        
        # Find the user to delete
        query = "SELECT id, username, email FROM users WHERE "
        params = []
        
        if user_id:
            query += "id = ?"
            params.append(user_id)
        elif username:
            query += "username = ?"
            params.append(username)
        elif email:
            query += "email = ?"
            params.append(email)
        
        user = conn.execute(query, params).fetchone()
        
        if not user:
            print(f"No user found with the provided information.")
            return False
        
        # Confirm deletion
        print(f"Found user: ID: {user['id']}, Username: {user['username']}, Email: {user['email']}")
        confirm = input("Are you sure you want to delete this user? (y/n): ")
        
        if confirm.lower() != 'y':
            print("User deletion cancelled.")
            return False
        
        # Delete the user
        conn.execute("DELETE FROM users WHERE id = ?", (user['id'],))
        conn.commit()
        
        print(f"User {user['username']} (ID: {user['id']}) has been deleted successfully.")
        return True
        
    except Exception as e:
        print(f"Error accessing database: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def display_help():
    """Display help information for the script."""
    print("\nUsage: python delete_user.py [options]")
    print("\nOptions:")
    print("  --id ID        Delete user by ID")
    print("  --username USERNAME  Delete user by username")
    print("  --email EMAIL      Delete user by email")
    print("  --help        Show this help message")
    print("\nExample:")
    print("  python delete_user.py --id 5")
    print("  python delete_user.py --username john")
    print("  python delete_user.py --email john@example.com")

if __name__ == "__main__":
    # Parse command line arguments
    args = sys.argv[1:]
    
    if not args or "--help" in args:
        display_help()
        sys.exit(0)
    
    user_id = None
    username = None
    email = None
    
    i = 0
    while i < len(args):
        if args[i] == "--id" and i + 1 < len(args):
            try:
                user_id = int(args[i + 1])
            except ValueError:
                print("Error: ID must be a number")
                sys.exit(1)
            i += 2
        elif args[i] == "--username" and i + 1 < len(args):
            username = args[i + 1]
            i += 2
        elif args[i] == "--email" and i + 1 < len(args):
            email = args[i + 1]
            i += 2
        else:
            print(f"Unknown option: {args[i]}")
            display_help()
            sys.exit(1)
    
    # Delete the user
    success = delete_user(user_id, username, email)
    
    if not success:
        sys.exit(1)

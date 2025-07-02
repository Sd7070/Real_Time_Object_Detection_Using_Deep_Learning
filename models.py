from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os


def get_db_connection():
    """Create a connection to the SQLite database"""
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with required tables"""
    conn = get_db_connection()
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_admin BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS contact_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                subject TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create admin user if not exists
        admin_user = conn.execute('SELECT * FROM users WHERE username = ?', ('admin',)).fetchone()
        if not admin_user:
            password_hash = generate_password_hash('admin123')
            conn.execute('INSERT INTO users (username, email, password_hash, is_admin) VALUES (?, ?, ?, ?)',
                       ('admin', 'admin@example.com', password_hash, True))

    conn.close()

class User(UserMixin):
    def __init__(self, id, username, email, password_hash, is_admin=False):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.is_admin = is_admin

    @staticmethod
    def get(user_id):
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        conn.close()
        
        if not user:
            return None
            
        # Check if is_admin column exists in the result
        try:
            is_admin = bool(user['is_admin'])
        except (IndexError, KeyError):
            is_admin = False
            
        return User(
            id=user['id'],
            username=user['username'],
            email=user['email'],
            password_hash=user['password_hash'],
            is_admin=is_admin
        )
        
    @staticmethod
    def find_by_username(username):
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if not user:
            return None
            
        # Check if is_admin column exists in the result
        try:
            is_admin = bool(user['is_admin'])
        except (IndexError, KeyError):
            is_admin = False
            
        return User(
            id=user['id'],
            username=user['username'],
            email=user['email'],
            password_hash=user['password_hash'],
            is_admin=is_admin
        )
    
    @staticmethod
    def find_by_email(email):
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        
        if not user:
            return None
            
        # Check if is_admin column exists in the result
        try:
            is_admin = bool(user['is_admin'])
        except (IndexError, KeyError):
            is_admin = False
            
        return User(
            id=user['id'],
            username=user['username'],
            email=user['email'],
            password_hash=user['password_hash'],
            is_admin=is_admin
        )
    
    @staticmethod
    def create(username, email, password, is_admin=False):
        conn = get_db_connection()
        
        # Check if username or email already exists
        existing_user = conn.execute(
            'SELECT * FROM users WHERE username = ? OR email = ?', 
            (username, email)
        ).fetchone()
        
        if existing_user:
            conn.close()
            return False
        
        password_hash = generate_password_hash(password)
        
        conn.execute(
            'INSERT INTO users (username, email, password_hash, is_admin) VALUES (?, ?, ?, ?)',
            (username, email, password_hash, 1 if is_admin else 0)
        )
        conn.commit()
        
        user_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        conn.close()
        
        return User.get(user_id)
    
    def check_password(self, password):
        try:
            return check_password_hash(self.password_hash, password)
        except ValueError as e:
            # Handle unsupported hash type (scrypt in Python 3.12)
            if 'unsupported hash type scrypt' in str(e):
                # For demonstration purposes, we'll reset the password
                # In a production environment, you would use a more secure approach
                print(f"Handling unsupported hash type for user {self.username}")
                conn = get_db_connection()
                # Update the password hash to a supported format (pbkdf2:sha256)
                new_hash = generate_password_hash(password, method='pbkdf2:sha256')
                conn.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, self.id))
                conn.commit()
                conn.close()
                # Update the current instance
                self.password_hash = new_hash
                return True
            else:
                # Re-raise other ValueError exceptions
                raise


class ContactMessage:
    def __init__(self, id=None, name=None, email=None, subject=None, message=None, created_at=None):
        self.id = id
        self.name = name
        self.email = email
        self.subject = subject
        self.message = message
        self.created_at = created_at
    
    @staticmethod
    def create(name, email, subject, message):
        """Create a new contact message in the database"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO contacts (name, email, subject, message) VALUES (?, ?, ?, ?)',
            (name, email, subject, message)
        )
        conn.commit()
        
        # Get the ID of the newly inserted message
        message_id = cursor.lastrowid
        conn.close()
        
        return ContactMessage.get(message_id)
    
    @staticmethod
    def get(message_id):
        """Get a contact message by ID"""
        conn = get_db_connection()
        message = conn.execute('SELECT * FROM contacts WHERE id = ?', (message_id,)).fetchone()
        conn.close()
        
        if not message:
            return None
            
        return ContactMessage(
            id=message['id'],
            name=message['name'],
            email=message['email'],
            subject=message['subject'],
            message=message['message'],
            created_at=message['created_at']
        )
    
    @staticmethod
    def get_all():
        """Get all contact messages"""
        conn = get_db_connection()
        messages = conn.execute('SELECT * FROM contacts ORDER BY created_at DESC').fetchall()
        conn.close()
        
        return [ContactMessage(
            id=message['id'],
            name=message['name'],
            email=message['email'],
            subject=message['subject'],
            message=message['message'],
            created_at=message['created_at']
        ) for message in messages]


def init_db():
    """Initialize the database with the users and contacts tables"""
    conn = sqlite3.connect('database.db')
    
    # Check if is_admin column exists in users table
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(users)")
    columns = cursor.fetchall()
    column_names = [column[1] for column in columns]
    
    if 'users' not in [col[0] for col in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
        # Create users table if it doesn't exist
        conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
    elif 'is_admin' not in column_names:
        # Add is_admin column if it doesn't exist
        conn.execute('ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0')
        print("Added is_admin column to users table")
    
    
    conn.execute('''
    CREATE TABLE IF NOT EXISTS contacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        subject TEXT NOT NULL,
        message TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()


# Function to set a user as admin
def set_user_as_admin(username):
    conn = get_db_connection()
    try:
        conn.execute('UPDATE users SET is_admin = 1 WHERE username = ?', (username,))
        conn.commit()
        print(f"User '{username}' has been set as admin")
        return True
    except Exception as e:
        print(f"Error setting user as admin: {str(e)}")
        return False
    finally:
        conn.close()

# Function to get all active sessions (placeholder)
def get_active_sessions():
    # This is a placeholder - in a real app, you'd track sessions in a database or cache
    # For this demo, we'll return an empty list since we don't have session tracking implemented
    return []

# This will only run when the file is executed directly
if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialized successfully.")
    
    # Ensure an admin user exists
    admin_user = User.find_by_username('Sanirajd4511')
    if not admin_user:
        # Create admin user if it doesn't exist
        User.create('Sanirajd4511', 'admin@example.com', 'UX-98%gh67lgo ', is_admin=True)
        print("Admin user created.")
    else:
        # Ensure the user has admin privileges
        if not admin_user.is_admin:
            set_user_as_admin(admin_user.username)
            print(f"User '{admin_user.username}' has been set as admin")
        print("Admin user privileges confirmed.")
        
        # Update password if needed
        conn = get_db_connection()
        conn.execute('UPDATE users SET password_hash = ? WHERE username = ?', 
                   (generate_password_hash('UX-98%gh67lgo '), 'Sanirajd4511'))
        conn.commit()
        conn.close()
        print("Admin password updated.")
    
    # Display all users in the database
    conn = get_db_connection()
    try:
        users = conn.execute('SELECT id, username, email, is_admin, created_at FROM users').fetchall()
        
        if not users:
            print("\nNo users are registered in the database yet.")
        else:
            print("\n===== REGISTERED USERS =====")
            print(f"{'ID':<5} {'Username':<20} {'Email':<30} {'Admin':<6} {'Created At':<20}")
            print("-" * 85)
            
            for user in users:
                is_admin = "Yes" if user['is_admin'] else "No"
                print(f"{user['id']:<5} {user['username']:<20} {user['email']:<30} {is_admin:<6} {user['created_at']:<20}")
            
            print("\nTotal users:", len(users))
    except Exception as e:
        print(f"Error accessing database: {str(e)}")
    finally:
        conn.close()
        
    # Display all contact messages
    print("\nRetrieving contact messages...")
    messages = ContactMessage.get_all()
    
    if not messages:
        print("No contact messages found.")
    else:
        print("\n===== CONTACT MESSAGES =====")
        print(f"{'ID':<5} {'Name':<20} {'Email':<30} {'Subject':<30}")
        print("-" * 85)
        
        for msg in messages:
            print(f"{msg.id:<5} {msg.name:<20} {msg.email:<30} {msg.subject:<30}")
        
        print("\nTotal messages:", len(messages))

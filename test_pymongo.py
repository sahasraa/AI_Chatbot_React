from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

# Connection string using your replica set and encoded password
uri = (
  
)

try:
    # Connect to MongoDB with TLS and custom CA
    client = MongoClient(
        uri,
        tls=True,
        tlsCAFile="mongo_root_ca.pem"  # Make sure this file is present in your working directory
    )

    # Force connection test
    print("✅ Attempting to connect to MongoDB Atlas...")
    databases = client.list_database_names()
    print("🎉 Connection successful!")
    print("📦 Databases on the cluster:")
    for db in databases:
        print("  •", db)

except ServerSelectionTimeoutError as e:
    print("❌ Failed to connect to MongoDB Atlas.")
    print("Reason:", e)
except Exception as ex:
    print("❌ Unexpected error occurred.")
    print("Details:", ex)

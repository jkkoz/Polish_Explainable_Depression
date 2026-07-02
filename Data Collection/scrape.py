from apify_client import ApifyClient
import pandas as pd

# Initialize the ApifyClient with API token
client = ApifyClient("<API>")

# Get the list of users (depressed or non-depressed)
f1 = open("<FILENAME>.txt", "r") 
users = f1.readlines()

for i in range(len(users)):
    users[i] = users[i].replace("\n", "")

print(users)
f1.close()

database = []

# Extract more tweets from each user
for user in users: 
    run_input = {
        "username": user,
        "query": f"(from: {user}) lang:pl -filter:retweets -filter:quotes", # Filter for retweets and quotes. 
        "search_type": "Top",
        "max_posts": 20,
    }

    # Run the Apify Actor
    run = client.actor("<ACTOR_NB>").call(run_input=run_input)

    # Fetch and print Actor results
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        item["queried_user"] = user
        database.append(item)

df = pd.DataFrame(database)
df.to_csv("<FILENAME>.csv", index=False, encoding="utf-8")

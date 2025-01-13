from flask import Flask, jsonify, request
from atproto import Client, models
import json
import os
import pickle
import time
import json

app = Flask(__name__)
feed_dict = {}
last_updated = {}  # Timestamp of the last update
update_interval = 300  # 5 minutes in seconds

def serve_feed(did = "did:plc:y7crv2yh74s7qhmtx3mvbgv5", feed = "art-new"):
    global last_updated
    global feed_dict

    current_time = time.time()

    # Check if the feed file exists and if it was updated recently
    if current_time - last_updated.get(feed, 0) < update_interval:
        app.logger.info("Returning existing feed")
        return json.dumps(feed_dict[feed], indent=2)

    app.logger.info(f"Updating feed at {current_time}")
    # Initialize the client
    client = Client()
    client.login(os.getenv("BLUESKY_HANDLE"), os.getenv("BLUESKY_PASSWORD"))

    # Parse the feed URI
    #algorithm = f"at://{did}/app.bsky.feed.generator/{feed}"
    feed_uri = f"at://{did}/app.bsky.feed.generator/{feed}"

    # Get feed with correct parameter format
    data = client.app.bsky.feed.get_feed(
        {
            "feed": feed_uri,
            "limit": 50,
        },
    )

    # Save feed as JSON
    feed_dict[feed] = json.loads(data.model_dump_json())

    # Update the last_updated timestamp
    last_updated[feed] = current_time

    return json.dumps(feed_dict[feed], indent=2)

@app.route("/", methods=["GET"])
def serve_art_feed():
    return serve_feed()


#https://bsky.app/profile/feristalj.bsky.social/feed/aaakapfwax3jm
@app.route("/video", methods=["GET"])
def serve_video_feed():
    return serve_feed(did="did:plc:if2a7jn3az2j3k2tyr4ycfg6", feed="aaakapfwax3jm")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

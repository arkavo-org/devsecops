from flask import Flask, jsonify
from atproto import Client
import os
import json
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Initialize Bluesky client
client = Client()

def fetch_user_fyp_with_media():
    timeline = client.get_timeline(limit=100)

    # Parse the timeline JSON dump
    timeline_json = json.loads(timeline.model_dump_json())

    # Debugging: Log each post's raw JSON
    logging.debug(f"Post: {json.dumps(timeline_json, indent=2)}")

    posts_with_media = []

    for post in timeline_json.get("feed", []):
        if "post" in post:
            post_data = post["post"]
            logging.debug("POST FOUND")
            if "embed" in post_data:
                logging.debug("EMBED FOUND")
                embed_data = post_data["embed"]
                if "playlist" in embed_data:
                    logging.debug("PLAYLIST FOUND")    
                    posts_with_media.append(post)
                    
    logging.debug(f"Fetched {len(timeline_json.get('feed', []))} posts from timeline.")
    logging.debug(f"Posts with media: {len(posts_with_media)}")

    return posts_with_media

# Filter posts with vertical videos
def filter_vertical_videos(posts):
    vertical_video_posts = []
    for post in posts:
        logging.debug(json.dumps(post))
        embed = getattr(post.post, 'embed', None)
        if embed and embed.get("py_type") == "app.bsky.embed.video#view":
            aspect_ratio = embed.get("aspect_ratio", {})
            height = aspect_ratio.get("height", 0)
            width = aspect_ratio.get("width", 0)
            if height > width:  # Check if the video is vertical
                vertical_video_posts.append(post)
    return vertical_video_posts

# Flask route to serve the filtered feed
@app.route("/", methods=["GET"])
def get_filtered_feed():
    try:
        # Log in to Bluesky
        client.login(
            os.getenv("BLUESKY_HANDLE"),
            os.getenv("BLUESKY_PASSWORD")
        )
        # Fetch the user's extended FYP with media
        fyp_posts_with_media = fetch_user_fyp_with_media()
        # Filter for vertical videos
        vertical_video_posts = filter_vertical_videos(fyp_posts_with_media)
        # Return the filtered feed as JSON
        return jsonify({"status": "success", "feed": [post.dict() for post in vertical_video_posts]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

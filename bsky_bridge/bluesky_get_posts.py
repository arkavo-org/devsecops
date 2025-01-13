import os
import requests
from datetime import datetime
import time
import env

def get_repos(pds_host, cursor=None):
    try:
        params = {
            'limit': 50
        }
        if cursor:
            params['cursor'] = cursor

        response = requests.get(
            f"{pds_host}/xrpc/com.atproto.sync.listRepos",
            params=params
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching repos: {e}")
        return None

def get_posts_for_repo(pds_host, repo_did, cursor=None):
    try:
        params = {
            'collection': 'app.bsky.feed.post',
            'repo': repo_did,
            'limit': 100
        }
        if cursor:
            params['cursor'] = cursor

        response = requests.get(
            f"{pds_host}/xrpc/com.atproto.repo.listRecords",
            params=params
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching posts for {repo_did}: {e}")
        return None

def transform_post(record):
    try:
        return {
            'uri': record['uri'],
            'cid': record['cid'],
            'text': record['value'].get('text', ''),
            'created_at': record['value'].get('createdAt', ''),
            'author_did': record['uri'].split('/')[2]
        }
    except (KeyError, IndexError) as e:
        print(f"Error transforming post: {e}")
        return None

def main():
    pds_host = env.VITE_BLUESKY_HOST
    print(f"Fetching posts from {pds_host}...")

    try:
        # First, get list of repositories
        repo_cursor = None
        processed_dids = set()

        while True:
            repos_data = get_repos(pds_host, repo_cursor)
            if not repos_data or 'repos' not in repos_data:
                print("No repositories found or invalid response")
                break

            # Process each repository
            for repo in repos_data['repos']:
                did = repo.get('did')
                if not did or did in processed_dids:
                    continue

                processed_dids.add(did)
                print(f"\nFetching posts for repository: {did}")
                
                # Get posts for this repository
                posts_cursor = None
                while True:
                    posts_data = get_posts_for_repo(pds_host, did, posts_cursor)
                    if not posts_data or 'records' not in posts_data:
                        break

                    # Process posts
                    for record in posts_data['records']:
                        post = transform_post(record)
                        if post:
                            created_at = datetime.fromisoformat(post['created_at'].replace('Z', '+00:00'))
                            print("\n---")
                            print(f"Time: {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                            print(f"Author DID: {post['author_did']}")
                            print(f"URI: {post['uri']}")
                            print(f"Text: {post['text']}")

                    # Check for more posts in this repo
                    if 'cursor' in posts_data:
                        posts_cursor = posts_data['cursor']
                    else:
                        break

                    time.sleep(0.5)  # Small delay between post pages

            # Check for more repositories
            if 'cursor' in repos_data:
                repo_cursor = repos_data['cursor']
                print(f"\nFetching next page of repositories with cursor: {repo_cursor}")
            else:
                print("\nNo more repositories to process")
                break

            time.sleep(1)  # Small delay between repository pages

    except KeyboardInterrupt:
        print("\nStopping post fetch...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    main()
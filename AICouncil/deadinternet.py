#!/usr/bin/env python3
import os
import json
import secrets
import string
import requests
from typing import Optional, Dict, List
from datetime import datetime, UTC

class BlueskyClient:
    def __init__(self, base_url: str, admin_password: str = "changeme"):
        """
        Initialize the Bluesky client.
        
        Args:
            base_url: The base URL of the PDS (Personal Data Server)
            admin_password: Admin password for the PDS (default: "changeme")
        """
        self.base_url = base_url.rstrip('/')
        self.admin_password = admin_password
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
        self.auth_token = None
        self.did = None

    def generate_password(self, length: int = 24) -> str:
        """Generate a secure random password."""
        alphabet = string.ascii_letters + string.digits
        alphabet = alphabet.translate(str.maketrans('', '', '=+/'))
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    def create_session(self, identifier: str, password: str) -> Optional[Dict]:
        """Create a session and get the access token."""
        try:
            login_url = f"{self.base_url}/xrpc/com.atproto.server.createSession"
            login_data = {
                "identifier": identifier,
                "password": password
            }
            response = self.session.post(login_url, json=login_data)
            response.raise_for_status()
            session_data = response.json()
            self.auth_token = session_data.get('accessJwt')
            self.did = session_data.get('did')
            return session_data
        except requests.exceptions.RequestException as e:
            print(f"Error creating session: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return None

    def create_admin_session(self) -> Optional[Dict]:
        """Create an admin session using admin credentials."""
        return self.create_session("admin", self.admin_password)

    def create_invite_code(self, use_count: int = 1) -> Optional[str]:
        """Create new invite codes."""
        try:
            url = f"{self.base_url}/xrpc/com.atproto.server.createInviteCodes"
            headers = {'Authorization': f'Bearer {self.auth_token}'} if self.auth_token else None
            data = {
                "useCount": use_count
            }
            
            # Try with bearer token first if available
            if headers:
                response = self.session.post(url, json=data, headers=headers)
            else:
                # Fall back to basic auth for admin
                response = self.session.post(
                    url,
                    json=data,
                    auth=('admin', self.admin_password)
                )
                
            response.raise_for_status()
            return response.json().get('codes', [None])[0]
        except requests.exceptions.RequestException as e:
            print(f"Error creating invite code: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return None

    def list_invite_codes(self) -> Optional[List[Dict]]:
        """List available invite codes."""
        try:
            url = f"{self.base_url}/xrpc/com.atproto.server.listInviteCodes"
            headers = {'Authorization': f'Bearer {self.auth_token}'} if self.auth_token else None
            
            if headers:
                response = self.session.get(url, headers=headers)
            else:
                response = self.session.get(url, auth=('admin', self.admin_password))
                
            response.raise_for_status()
            return response.json().get('codes', [])
        except requests.exceptions.RequestException as e:
            print(f"Error listing invite codes: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return None

    def check_handle_availability(self, handle: str) -> bool:
        """Check if a handle is available."""
        url = f"{self.base_url}/xrpc/com.atproto.identity.resolveHandle"
        try:
            response = self.session.get(url, params={"handle": handle})
            return False  # Handle exists
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return True  # Handle is available
            raise

    def create_account(self, email: str, handle: str, password: Optional[str] = None, 
                      invite_code: Optional[str] = None) -> Optional[Dict]:
        """Create a new Bluesky account."""
        try:
            if not password:
                password = self.generate_password()
            
            if not invite_code:
                invite_code = self.create_invite_code()
                if not invite_code:
                    raise ValueError("Failed to create invite code")

            # Check handle availability
            if not self.check_handle_availability(handle):
                raise ValueError(f"Handle '{handle}' is already taken")
                
            create_url = f"{self.base_url}/xrpc/com.atproto.server.createAccount"
            create_data = {
                "email": email,
                "handle": handle,
                "password": password,
                "inviteCode": invite_code
            }
            
            response = self.session.post(create_url, json=create_data)
            response.raise_for_status()
            
            account_data = response.json()
            return {
                "handle": handle,
                "did": account_data['did'],
                "password": password,
                "accessJwt": account_data.get('accessJwt')
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Error creating account: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return None

    def create_post(self, text: str, reply_to: Optional[str] = None) -> Optional[Dict]:
        """Create a new post."""
        if not self.auth_token or not self.did:
            raise ValueError("Must be logged in to create posts")

        try:
            url = f"{self.base_url}/xrpc/com.atproto.repo.createRecord"
            
            post_data = {
                "text": text,
                "createdAt": datetime.now(UTC).isoformat(),
                "$type": "app.bsky.feed.post"
            }
            
            if reply_to:
                post_data["reply"] = {"root": reply_to}

            data = {
                "repo": self.did,
                "collection": "app.bsky.feed.post",
                "record": post_data
            }

            headers = {"Authorization": f"Bearer {self.auth_token}"}
            response = self.session.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"Error creating post: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return None

    def get_profile(self, handle: str) -> Optional[Dict]:
        """Get profile information for a handle."""
        try:
            url = f"{self.base_url}/xrpc/app.bsky.actor.getProfile"
            params = {"actor": handle}
            headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else None
            
            response = self.session.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching profile: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return None

def main():
    # Configuration
    BLUESKY_HOST = "https://app.codecollective.us"
    ADMIN_PASSWORD = "changeme"
    TEST_EMAIL = "alice@example.com"
    TEST_HANDLE = "alice.test"
    
    # Initialize client
    client = BlueskyClient(BLUESKY_HOST, ADMIN_PASSWORD)
    
    # Create admin session
    print("Creating admin session...")
    admin_session = client.create_admin_session()
    if not admin_session:
        print("Failed to create admin session")
        return

    print("Admin session created successfully!")
    
    # Check handle availability
    print(f"\nChecking handle availability for {TEST_HANDLE}...")
    if client.check_handle_availability(TEST_HANDLE):
        print(f"Handle '{TEST_HANDLE}' is available!")
        
        # Create account
        print("\nCreating new account...")
        account_data = client.create_account(TEST_EMAIL, TEST_HANDLE)
        
        if account_data:
            print("\nAccount created successfully!")
            print("-----------------------------")
            print(f"Handle   : {account_data['handle']}")
            print(f"DID      : {account_data['did']}")
            print(f"Password : {account_data['password']}")
            print("-----------------------------")
            
            # Create a test post
            if client.create_session(TEST_HANDLE, account_data['password']):
                print("\nCreating test post...")
                post_result = client.create_post("Hello, Bluesky! This is my first post.")
                if post_result:
                    print("Post created successfully!")
                else:
                    print("Failed to create post!")
            else:
                print("Failed to create session for new account!")
        else:
            print("Failed to create account!")
    else:
        print(f"Handle '{TEST_HANDLE}' is already taken!")

if __name__ == "__main__":
    main()
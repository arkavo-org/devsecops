#!/usr/bin/env python3
import os
import json
import secrets
import string
import requests
import env
from typing import Optional
from datetime import datetime, UTC
import re


class ATProtoClient:
    def __init__(self, PDS_HOSTNAME=env.USER_WEBSITE, PDS_ADMIN_PASSWORD="changeme"):
        self.hostname = PDS_HOSTNAME
        self.admin_password = PDS_ADMIN_PASSWORD

        self.session = requests.Session()
        self.auth_token = None
        self.did = None  # Store DID after login for post creation

    def validate_handle(self, handle: str) -> bool:
        """
        Validate handle format according to AT Protocol specifications.
        
        Args:
            handle (str): Handle to validate
            
        Returns:
            bool: True if handle is valid, False otherwise
        """
        # Handle must be lowercase and can only contain a-z, 0-9, -, .
        pattern = r'^[a-z0-9][a-z0-9-]{0,63}(\.[a-z0-9][a-z0-9-]{0,63})*$'
        
        if not re.match(pattern, handle):
            print(f"Invalid handle format: {handle}")
            return False
            
        return True

    def generate_password(self, length: int = 24) -> str:
        """Generate a secure random password."""
        alphabet = string.ascii_letters + string.digits
        alphabet = alphabet.translate(str.maketrans("", "", "=+/"))
        return "".join(secrets.choice(alphabet) for _ in range(length))

    def create_invite_code(self) -> str:
        """Create a single-use invite code."""
        url = f"https://{self.hostname}/xrpc/com.atproto.server.createInviteCode"
        response = self.session.post(
            url, auth=("admin", self.admin_password), json={"useCount": 1}
        )
        response.raise_for_status()
        return response.json()["code"]

    def check_handle_availability(self, handle: str) -> bool:
        """Check if a handle is available."""
        url = f"https://{self.hostname}/xrpc/com.atproto.identity.resolveHandle"
        try:
            response = self.session.get(url, params={"handle": handle})
            return False  # Handle exists
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return True  # Handle is available
            raise

    def login(self, identifier: str, password: str) -> bool:
        """Log in to get an authentication token."""
        url = f"https://{self.hostname}/xrpc/com.atproto.server.createSession"
        try:
            response = self.session.post(
                url, json={"identifier": identifier, "password": password}
            )
            response.raise_for_status()
            session_data = response.json()
            self.auth_token = session_data.get("accessJwt")
            self.did = session_data.get("did")  # Store the DID
            print(f"Login success: {identifier}")
            return True
        except requests.exceptions.HTTPError:
            print(f"Login failed: {identifier}")
            return False

    def create_account(self, email: str, handle: str) -> dict:
        """Create a new account with the given email and handle."""
        # Check if handle is available
        print(f"Checking account availability for handle {handle}")
        if not self.check_handle_availability(handle):
            print(f"Handle '{handle}' is already taken")

        password = self.generate_password()
        invite_code = self.create_invite_code()

        url = f"https://{self.hostname}/xrpc/com.atproto.server.createAccount"
        data = {
            "email": email,
            "handle": handle,
            "password": password,
            "inviteCode": invite_code,
        }

        response = self.session.post(url, json=data)

        if response.status_code != 200:
            error_msg = response.json().get("message", "Unknown error occurred")
            print(f"Account creation failed: {error_msg}")
            return

        account_data = response.json()
        result = {"handle": handle, "password": password}
        # Display results
        print("\nAccount created successfully!")
        print("-----------------------------")
        print(f"Handle   : {result['handle']}")
        print(f"Password : {result['password']}")
        print("-----------------------------")
        print("Save this password, it will not be displayed again.")

        return result

    def handle_to_did(self, handle: str) -> str | None:
        """
        Resolve a handle to its corresponding DID.

        Args:
            handle (str): The handle to resolve (e.g., "alice.bsky.social")

        Returns:
            str | None: The DID if successful, None if resolution fails
        """
        resolve_url = f"https://{self.hostname}/xrpc/com.atproto.identity.resolveHandle"
        params = {"handle": handle}

        try:
            resolve_response = self.session.get(resolve_url, params=params)

            if resolve_response.status_code != 200:
                error_msg = resolve_response.json().get(
                    "message", "Unknown error occurred"
                )
                print(f"Failed to resolve handle: {error_msg}")
                return None

            did = resolve_response.json().get("did")
            if not did:
                print("Could not retrieve DID for handle")
                return None

            return did

        except Exception as e:
            print(f"Error resolving handle: {str(e)}")
            return None
        
    def request_account_delete(self, did: str, password: str) -> str | None:
        """Request account deletion and get token.

        Args:
            did (str): The DID of the account to delete
            password (str): The password for the account
            
        Returns:
            str | None: The deletion token if successful, None if failed
        """
        print(f"Requesting deletion token for DID {did}")

        # First create a session to get auth token
        session_url = f"https://{self.hostname}/xrpc/com.atproto.server.createSession"
        session_data = {
            "identifier": 'admin',
            "password": "changeme"
        }
        
        session_response = self.session.post(session_url, json=session_data)
        if session_response.status_code != 200:
            print(f"Failed to create session: {session_response.json().get('message')}")
            return None
            
        access_jwt = session_response.json().get('accessJwt')

        # Now request the deletion token using the session
        url = f"https://{self.hostname}/xrpc/com.atproto.server.requestAccountDelete"
        headers = {"Authorization": f"Bearer {access_jwt}"}
        
        response = self.session.post(url, headers=headers)

        if response.status_code != 200:
            error_msg = response.json().get("message", "Unknown error occurred")
            print(f"Failed to get deletion token: {error_msg}")
            return None

        return response.json().get("token")

    def delete_account(self, handle: str, password: str, did: str) -> bool:
        """Delete an existing account with given credentials.

        Args:
            handle (str): The handle of the account to delete
            password (str): The password for the account
            did (str): The DID of the account

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        print(f"Attempting to delete account for handle {handle}")

        # First get the deletion token
        token = self.request_account_delete(did, password)
        if not token:
            return False

        url = f"https://{self.hostname}/xrpc/com.atproto.server.deleteAccount"
        data = {
            "did": did,
            "password": password,
            "token": token
        }

        response = self.session.post(url, json=data)

        if response.status_code != 200:
            error_msg = response.json().get("message", "Unknown error occurred")
            print(f"Account deletion failed: {error_msg}")
            return False

        # Display results
        print("\nAccount deleted successfully!")
        print("-----------------------------")
        print(f"Handle : {handle}")
        print(f"DID    : {did}")
        print("-----------------------------")

        return True

    def create_post(self, text: str, reply_to: Optional[str] = None) -> dict:
        """Create a new post."""
        if not self.auth_token or not self.did:
            raise ValueError("Must be logged in to create posts")

        url = f"https://{self.hostname}/xrpc/com.atproto.repo.createRecord"

        post_data = {
            "text": text,
            "createdAt": datetime.now(UTC).isoformat(),
            "$type": "app.bsky.feed.post",
        }

        if reply_to:
            post_data["reply"] = {"root": reply_to}

        data = {
            "repo": self.did,
            "collection": "app.bsky.feed.post",
            "record": post_data,
        }

        headers = {"Authorization": f"Bearer {self.auth_token}"}
        response = self.session.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()

    def get_profile(self, handle: str) -> dict:
        """Get profile information for a handle."""
        url = f"https://{self.hostname}/xrpc/app.bsky.actor.getProfile"
        params = {"actor": handle}
        if self.auth_token:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            response = self.session.get(url, params=params, headers=headers)
        else:
            response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()


def main(email="alice@" + env.USER_WEBSITE, handle="alice." + env.USER_WEBSITE):
    client = ATProtoClient()
    result = client.create_account(email=email, handle=handle)

    # Demo post creations
    if client.login(handle, result["password"]):
        print("\nCreating demo post...")
        post_result = client.create_post("Hello, Bluesky! This is my first post.")
        print("Post created successfully!")
    else:
        print("Failed to login!")


if __name__ == "__main__":
    main()

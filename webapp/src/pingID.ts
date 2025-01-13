// src/authConfig.ts
import { UserManager, WebStorageStateStore } from 'oidc-client';

const authConfig = {
    authority: 'https://auth.pingone.com/YOUR_PING_IDENTITY_DOMAIN', // Replace with your Ping Identity domain
    client_id: 'YOUR_CLIENT_ID', // Replace with your Ping Identity client ID
    redirect_uri: 'http://localhost:5173/callback', // Replace with your redirect URI
    post_logout_redirect_uri: 'http://localhost:5173',
    response_type: 'code',
    scope: 'openid profile email',
    userStore: new WebStorageStateStore({ store: window.localStorage })
};

export const userManager = new UserManager(authConfig);

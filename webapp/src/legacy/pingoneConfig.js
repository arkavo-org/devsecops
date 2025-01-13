// pingoneConfig.js
export const pingoneConfig = {
    clientId: '31a5c075-4577-4c94-86f6-0623f8efafff', // Replace with your PingOne Client ID
    environmentId: '9c898001-7e8c-48d2-b3f6-c5e89d0ca2e1', // Replace with your PingOne Environment ID
    redirectUri: 'http://localhost:5173', // Your app's redirect URI
    scopes: ['openid', 'profile', 'email'],
    responseType: 'code',
    pkce: true,
    };

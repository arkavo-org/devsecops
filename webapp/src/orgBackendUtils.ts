export async function fetchKeycloakUsers(token: string): Promise<{ users: { username: string }[] }> {
    const backendUrl = import.meta.env.VITE_ORG_BACKEND_URL;

    if (!backendUrl) {
        throw new Error('VITE_ORG_BACKEND_URL is not set in the environment variables.');
    }

    const response = await fetch(`${backendUrl}/users`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
        },
    });

    if (!response.ok) {
        const errorText = await response.text();
        console.error(`Failed to fetch users: ${response.status} ${response.statusText} - ${errorText}`);
        throw new Error(`Failed to fetch users: ${response.status} ${response.statusText} - ${errorText}`);
    }

    const responseBody = await response.json();

    if (!responseBody?.users) {
        throw new Error('Response format invalid: Missing "users" property.');
    }

    return responseBody; // Expecting { users: [...] }
}

// Utils.tsx

import { json } from "stream/consumers";

// Function to validate if a Matrix token is still valid
export const validateToken = async (token: string, synapseBaseUrl: string): Promise<boolean> => {
    try {
        // Make a simple API call to check if the token is valid
        // Using /account/whoami is a lightweight way to validate the token
        const response = await fetch(`https://${synapseBaseUrl}/_matrix/client/v3/account/whoami`, {
            headers: { 
                Authorization: `Bearer ${token}`,
                'Content-Type': 'application/json' 
            },
        });
        
        return response.ok;
    } catch (error) {
        console.error('Token validation error:', error);
        return false;
    }
};

// Function to attempt token refresh
export const refreshToken = async (token: string, synapseBaseUrl: string): Promise<string | null> => {
    try {
        // Matrix doesn't have a standard refresh token mechanism
        // This is a placeholder for a custom refresh implementation
        // You might need to implement this based on your specific Matrix server setup
        
        // For example, if your server supports token refresh:
        const response = await fetch(`https://${synapseBaseUrl}/_matrix/client/v3/refresh`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                refresh_token: token,
            }),
        });
        
        if (response.ok) {
            const data = await response.json();
            return data.access_token;
        }
        
        return null;
    } catch (error) {
        console.error('Token refresh error:', error);
        return null;
    }
};

export const fetchUserProfile = async (userId: string, token: string, synapseBaseUrl: string) => {
    const response = await fetch(`https://${synapseBaseUrl}/_matrix/client/v3/profile/${userId}`, {
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    });

    if (!response.ok) throw new Error('Failed to fetch user profile');
    const data = await response.json();
    return {
        displayName: data.displayname || userId,
        avatarUrl: data.avatar_url || null,
    };
};

export const fetchRooms = async (token: string, synapseBaseUrl: string) => {
    // Fetch the list of joined rooms
    const response = await fetch(`https://${synapseBaseUrl}/_matrix/client/r0/joined_rooms`, {
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    });

    if (!response.ok) throw new Error('Failed to fetch rooms');
    const data = await response.json();
    console.log(JSON.stringify(data));

    // Fetch room details (name, alias, and avatar) for each room
    const roomDetails = await Promise.all(
        data.joined_rooms.map(async (roomId: string) => {
            let alias: string | null = null;
            let name: string | null = null;
            let avatarUrl: string | null = null;

            // Fetch room aliases
            try {
                const aliasResponse = await fetch(`https://${synapseBaseUrl}/_matrix/client/v3/rooms/${roomId}/aliases`, {
                    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
                });

                if (aliasResponse.ok) {
                    const aliasData = await aliasResponse.json();
                    alias = aliasData.aliases?.[0] || null;
                }
            } catch (error) {
                console.warn(`Failed to fetch alias for room ${roomId}:`, error);
            }

            // Fetch room name
            try {
                const nameResponse = await fetch(`https://${synapseBaseUrl}/_matrix/client/v3/rooms/${roomId}/state/m.room.name`, {
                    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
                });

                if (nameResponse.ok) {
                    const nameData = await nameResponse.json();
                    name = nameData.name || null;
                }
            } catch (error) {
                console.warn(`Failed to fetch name for room ${roomId}:`, error);
            }

            // Fetch room avatar
            try {
                const avatarResponse = await fetch(`https://${synapseBaseUrl}/_matrix/client/v3/rooms/${roomId}/state/m.room.avatar`, {
                    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
                });

                if (avatarResponse.ok) {
                    const avatarData = await avatarResponse.json();
                    avatarUrl = avatarData.url || null;
                }
            } catch (error) {
                console.warn(`Failed to fetch avatar for room ${roomId}:`, error);
            }

            return { roomId, alias, name, avatarUrl };
        })
    );
    
    console.log("Room Details:", roomDetails);
    return roomDetails;
};

export const fetchPeople = async (token: string, synapseBaseUrl: string, searchTerm: string = "", limit = 10) => {
    const requestBody = {
        limit,
        search_term: searchTerm, // Empty string defaults to searching all users
    };

    const response = await fetch(`https://${synapseBaseUrl}/_matrix/client/v3/user_directory/search`, {
        method: 'POST',
        headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
        throw new Error(`Failed to fetch people: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    console.log(`Response received from User Directory API. ${JSON.stringify(data)}`);
    return data.results || []; // Assumes the response contains a `results` array of users.
};

// Utils.tsx

export interface Message {
    sender: string;
    body: string;
    avatarUrl?: string;
    displayName?: string;
    timestamp?: number;
    eventId?: string;
}

const sinceTokens = new Map<string, string>(); // Map to store `since` tokens per room

export const fetchMessages = async (roomId: string, synapseBaseUrl: string, partial: boolean) => {
    let accessToken = localStorage.getItem('matrixAccessToken');
    if (!accessToken) {
        throw new Error('Access token not found.');
    }

    // Validate the token
    const isValid = await validateToken(accessToken, synapseBaseUrl);
    
    if (!isValid) {
        // Token is invalid, try to refresh it
        const refreshedToken = await refreshToken(accessToken, synapseBaseUrl);
        
        if (refreshedToken) {
            // Successfully refreshed the token
            localStorage.setItem('matrixAccessToken', refreshedToken);
            accessToken = refreshedToken;
        } else {
            // Refresh failed, throw error to trigger login redirect
            localStorage.removeItem('matrixAccessToken');
            throw new Error('Token invalid and refresh failed. Please login again.');
        }
    }

    const syncUrl = new URL(`https://${synapseBaseUrl}/_matrix/client/v3/sync`);

    // Handle `since` token for the specific room
    if (partial) {
        const since = sinceTokens.get(roomId); // Get the `since` token for this room
        if (since) {
            syncUrl.searchParams.append('since', since); // Append the `since` token to the request
        }
    } else {
        sinceTokens.delete(roomId); // Reset the `since` token for a full sync
    }

    const response = await fetch(syncUrl.toString(), {
        headers: {
            Authorization: `Bearer ${accessToken}`,
            'Content-Type': 'application/json',
        },
    });

    if (!response.ok) {
        // If fetching fails due to auth issues, throw a specific error
        if (response.status === 401 || response.status === 403) {
            localStorage.removeItem('matrixAccessToken');
            throw new Error('Authentication failed. Please login again.');
        }
        throw new Error(`Failed to fetch messages: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();

    // Update the `since` token for this room
    if (data.next_batch) {
        sinceTokens.set(roomId, data.next_batch);
    }

    const roomData = data.rooms?.join?.[roomId];
    if (!roomData || !roomData.timeline?.events) {
        console.warn(`No timeline events found for room ${roomId}`);
        return [];
    }

    const messages = await Promise.all(
        roomData.timeline.events
            .filter((event: any) => event.type === 'm.room.message')
            .map(async (event: any) => {
                const profile = await fetchUserProfile(event.sender, accessToken, synapseBaseUrl);
                return {
                    sender: event.sender,
                    body: event.content?.body || '[Unknown Message]',
                    avatarUrl: profile.avatarUrl,
                    displayName: profile.displayName,
                    timestamp: event.origin_server_ts || 0,
                    eventId: event.event_id,
                };
            })
    );

    // Sorting is optional if the API returns events in order
    messages.sort((a: any, b: any) => (a.timestamp || 0) - (b.timestamp || 0));
    return messages;
};

export const fetchPublicRooms = async (
    token: string,
    synapseBaseUrl: string,
    federated: boolean = false
) => {
    try {
        // Set up query parameters
        const params = new URLSearchParams();

        // Add the `server` parameter only for local search
        if (!federated) {
            params.append('server', synapseBaseUrl);
        }

        // Add the `include_all_networks` parameter for federated search
        params.append('include_all_networks', federated ? 'true' : 'false');

        // Construct the request URL
        const requestUrl = `https://${synapseBaseUrl}/_matrix/client/v3/publicRooms?${params.toString()}`;
        console.log('Request URL:', requestUrl);

        // Fetch public rooms
        const response = await fetch(requestUrl, {
            headers: {
                Authorization: `Bearer ${token}`,
                'Content-Type': 'application/json',
            },
        });

        // Handle errors
        if (!response.ok) {
            const responseText = await response.text(); // Log the raw response
            console.error('Error response:', responseText);
            throw new Error(`Failed to fetch public rooms: ${response.status} ${response.statusText}`);
        }

        // Parse and return the room data
        const data = await response.json();
        return data.chunk || [];
    } catch (err) {
        console.error('Error fetching public rooms:', err);
        throw err;
    }
};

export const joinRoom = async (accessToken: string, synapseBaseUrl: string, roomId: string) => {
    const response = await fetch(`https://${synapseBaseUrl}/_matrix/client/r0/join/${encodeURIComponent(roomId)}`, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${accessToken}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
    });

    if (!response.ok) {
        throw new Error(`Failed to join room: ${response.statusText}`);
    }

    return response.json();
};

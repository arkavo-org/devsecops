import { KeycloakInstance } from 'keycloak-js';

export interface UserProfile {
    name: string;
    picture: string;
}

export const fetchStoredUserProfile = (): UserProfile | null => {
    const storedProfile = localStorage.getItem('userProfile');
    return storedProfile ? JSON.parse(storedProfile) : null;
};

export const fetchAndSetUserProfile = async (
    keycloak: KeycloakInstance,
    setUserProfile: (profile: UserProfile | null) => void
): Promise<void> => {
    try {
        const userProfile = await loginAndFetchProfile(keycloak);
        setUserProfile(userProfile);
    } catch (error) {
        console.error('Failed to fetch and set user profile:', error);
        setUserProfile(null);
    }
};

export const loginAndFetchProfile = async (
    keycloak: KeycloakInstance,
    redirectUri: string = window.location.origin
): Promise<UserProfile | null> => {
    // Refresh token if necessary
    if (keycloak.authenticated && keycloak.token) {
        await keycloak.updateToken(30); // Refresh token if it's about to expire
    }

    if (!keycloak.authenticated) {
        await keycloak.login({
            scope: 'openid profile email',
            redirectUri,
        });
    }

    if (keycloak.authenticated) {
        try {
            const response = await fetch(
                `${keycloak.authServerUrl}/realms/${keycloak.realm}/protocol/openid-connect/userinfo`,
                {
                    headers: {
                        Authorization: `Bearer ${keycloak.token}`,
                    },
                }
            );

            const userInfo = await response.json();
            const userProfile: UserProfile = {
                name: userInfo.name || userInfo.preferred_username || 'User',
                picture: userInfo.picture || '/default-profile.png', // Default if picture is missing
            };

            localStorage.setItem('userProfile', JSON.stringify(userProfile));
            return userProfile;
        } catch (error) {
            console.error('Failed to fetch user profile:', error);
        }
    }

    return null;
};

export const logoutAndClearProfile = (keycloak: KeycloakInstance): void => {
    keycloak.logout();
    localStorage.removeItem('userProfile');
};

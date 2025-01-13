import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

const MatrixAuthForward: React.FC = () => {
    const [error, setError] = useState<string | null>(null);
    const synapseBaseUrl = import.meta.env.VITE_SYNAPSE_BASE_URL;
    const navigate = useNavigate();

    const initiateSSOLogin = async () => {
        const redirectUrl = `${window.location.origin}/chatauth`;
        window.location.href = `${synapseBaseUrl}/_matrix/client/r0/login/sso/redirect?redirectUrl=${encodeURIComponent(redirectUrl)}`;
    };

    const exchangeLoginTokenForAccessToken = async (loginToken: string): Promise<string> => {
        const response = await fetch(`${synapseBaseUrl}/_matrix/client/r0/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                type: 'm.login.token',
                token: loginToken,
            }),
        });

        if (!response.ok) throw new Error('Failed to exchange login token');
        const data = await response.json();
        return data.access_token;
    };

    useEffect(() => {
        const handleLogin = async () => {
            const urlParams = new URLSearchParams(window.location.search);
            const loginToken = urlParams.get('loginToken');

            try {
                if (loginToken) {
                    const accessToken = await exchangeLoginTokenForAccessToken(loginToken);
                    localStorage.setItem('matrixAccessToken', accessToken); // Persist access token
                    navigate('/chat');
                } else {
                    await initiateSSOLogin();
                }
            } catch (error) {
                setError(error instanceof Error ? error.message : 'Unknown error occurred');
            }
        };

        handleLogin();
    }, [navigate]);

    return (
        <div>
            {error ? <p>Error: {error}</p> : <p>Redirecting...</p>}
        </div>
    );
};

export default MatrixAuthForward;

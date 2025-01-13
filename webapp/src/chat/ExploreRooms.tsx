import React, { useState, useEffect } from 'react';
import '../css/ExploreRooms.css';
import { useNavigate } from 'react-router-dom';
import { useKeycloak } from '@react-keycloak/web';

const ExploreRooms: React.FC = () => {
    const { keycloak, initialized } = useKeycloak();
    const navigate = useNavigate();
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [publicRooms, setPublicRooms] = useState<any[]>([]);

    const synapseBaseUrl = import.meta.env.VITE_SYNAPSE_BASE_URL;

    const fetchPublicRooms = async (token: string) => {
        try {
            const response = await fetch(`${synapseBaseUrl}/_matrix/client/r0/publicRooms`, {
                headers: {
                    Authorization: `Bearer ${token}`,
                    'Content-Type': 'application/json',
                },
            });

            if (!response.ok) throw new Error('Failed to fetch public rooms');
            const data = await response.json();
            setPublicRooms(data.chunk || []);
        } catch (err) {
            console.error('Error fetching public rooms:', err);
            setError(err instanceof Error ? err.message : 'Unknown error occurred');
        }
    };

    const joinRoom = async (roomId: string) => {
        const accessToken = localStorage.getItem('matrixAccessToken');
        if (!accessToken) {
            navigate('/chatauth');
            return;
        }

        try {
            const response = await fetch(`${synapseBaseUrl}/_matrix/client/v3/join/${encodeURIComponent(roomId)}`, {
                method: 'POST',
                headers: {
                    Authorization: `Bearer ${accessToken}`,
                    'Content-Type': 'application/json',
                },
            });

            if (!response.ok) {
                throw new Error(`Failed to join room: ${response.statusText}`);
            }

            // Save the room ID to localStorage
            localStorage.setItem('selectedRoom', roomId);

            // Navigate back to the chat page
            navigate('/chat');
        } catch (err) {
            console.error('Error joining room:', err);
            setError(err instanceof Error ? err.message : 'Unknown error occurred');
        }
    };

    useEffect(() => {
        const initialize = async () => {
            const storedAccessToken = localStorage.getItem('matrixAccessToken');

            if (!storedAccessToken) {
                navigate('/chatauth');
                return;
            }

            setLoading(true);

            try {
                await fetchPublicRooms(storedAccessToken);
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Unknown error occurred');
            } finally {
                setLoading(false);
            }
        };

        initialize();
    }, [navigate, synapseBaseUrl]);

    const handleCreateRoomClick = () => {
        navigate('/create-room'); // Navigate to the room creation page
    };

    if (loading) return <div>Loading public rooms...</div>;
    if (error) return <div>Error: {error}</div>;

    return (
        <div id="public-rooms-container">
            <h1>Public Rooms</h1>
            <div className="create-room-box" onClick={handleCreateRoomClick}>
                <div className="plus-sign">+</div>
                <p>Create Room</p>
            </div>
            <ul>
                {publicRooms.map((room) => (
                    <li
                        key={room.room_id}
                        className="room-item"
                        onClick={() => joinRoom(room.room_id)} // Join room on click
                    >
                        <div>
                            <h2>{room.name || 'Unnamed Room'}</h2>
                            <p>{room.topic || 'No topic provided'}</p>
                            <small>Members: {room.num_joined_members}</small>
                        </div>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default ExploreRooms;

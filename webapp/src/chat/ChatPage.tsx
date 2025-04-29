// ChatPage.tsx

import React, { useState, useEffect, useRef } from 'react';
import '../css/ChatPage.css';
import { useKeycloak } from '@react-keycloak/web';
import { useNavigate } from 'react-router-dom';
import { Chat } from './Chat';
import { handleLogin } from './ChatAuth';
import { fetchRooms, fetchPeople, joinRoom, validateToken, refreshToken } from './Utils'; // Import required functions

const ChatPage: React.FC = () => {
    const { keycloak, initialized } = useKeycloak();
    const navigate = useNavigate();
    const [selectedRoom, setSelectedRoom] = useState<string | null>(
        localStorage.getItem('selectedRoom') // Load the selected room from localStorage
    );
    const [conversations, setConversations] = useState<{ [key: string]: string[] }>({});
    const [rooms, setRooms] = useState<any[]>([]);
    const [people, setPeople] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const accessTokenRef = useRef<string | null>(null);
    const synapseBaseUrl = import.meta.env.VITE_SYNAPSE_BASE_URL;

    const handleTokenExpiry = () => {
        localStorage.removeItem('matrixAccessToken');
        handleLogin();
    };

    const handleRoomSelect = async (roomId: string) => {
        let storedAccessToken = localStorage.getItem('matrixAccessToken');
        if (!storedAccessToken) {
            handleLogin();
            return;
        }

        try {
            // Validate the token
            const isValid = await validateToken(storedAccessToken, synapseBaseUrl);
            
            if (!isValid) {
                // Token is invalid, try to refresh it
                const refreshedToken = await refreshToken(storedAccessToken, synapseBaseUrl);
                
                if (refreshedToken) {
                    // Successfully refreshed the token
                    localStorage.setItem('matrixAccessToken', refreshedToken);
                    storedAccessToken = refreshedToken;
                } else {
                    // Refresh failed, redirect to login
                    console.log("Token refresh failed, redirecting to login");
                    localStorage.removeItem('matrixAccessToken');
                    handleLogin();
                    return;
                }
            }

            // Join the room if not already a member
            await joinRoom(storedAccessToken, synapseBaseUrl, roomId);
            setSelectedRoom(roomId);
            localStorage.setItem('selectedRoom', roomId); // Save the selected room to localStorage
        } catch (error) {
            setError((error as Error).message);
            
            // Check if the error is related to authentication
            const errorMessage = (error as Error).message;
            if (
                errorMessage.includes('Authentication failed') || 
                errorMessage.includes('Unauthorized') ||
                errorMessage.includes('Forbidden')
            ) {
                localStorage.removeItem('matrixAccessToken');
                handleLogin();
            }
        }
    };

    useEffect(() => {
        const initialize = async () => {
            const storedAccessToken = localStorage.getItem('matrixAccessToken');
            if (storedAccessToken) {
                try {
                    // Validate the token
                    const isValid = await validateToken(storedAccessToken, synapseBaseUrl);
                    
                    if (isValid) {
                        // Token is valid, proceed normally
                        accessTokenRef.current = storedAccessToken;
                        const peopleData = await fetchPeople(storedAccessToken, synapseBaseUrl);
                        setPeople(peopleData);

                        const roomsData = await fetchRooms(storedAccessToken, synapseBaseUrl);
                        setRooms(roomsData);
                    } else {
                        // Token is invalid, try to refresh it
                        const refreshedToken = await refreshToken(storedAccessToken, synapseBaseUrl);
                        
                        if (refreshedToken) {
                            // Successfully refreshed the token
                            localStorage.setItem('matrixAccessToken', refreshedToken);
                            accessTokenRef.current = refreshedToken;
                            
                            // Fetch data with the new token
                            const peopleData = await fetchPeople(refreshedToken, synapseBaseUrl);
                            setPeople(peopleData);

                            const roomsData = await fetchRooms(refreshedToken, synapseBaseUrl);
                            setRooms(roomsData);
                        } else {
                            // Refresh failed, redirect to login
                            console.log("Token refresh failed, redirecting to login");
                            localStorage.removeItem('matrixAccessToken');
                            handleLogin();
                        }
                    }
                } catch (error) {
                    setError((error as Error).message);
                    // On any error, it's safer to redirect to login
                    localStorage.removeItem('matrixAccessToken');
                    handleLogin();
                }
            } else {
                // No token found, redirect to login
                handleLogin();
            }
            setLoading(false);
        };

        initialize();
    }, [navigate, synapseBaseUrl]);

    useEffect(() => {
        if (!selectedRoom) {
            const storedRoom = localStorage.getItem('selectedRoom');
            if (storedRoom) {
                setSelectedRoom(storedRoom);
            }
        }
    }, []);

    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error: {error}</div>;

    return (
        <div className="chat-page">
            <div className="sidebar">
                <div className="rooms-list">
                    <h4>Rooms</h4>
                    {rooms.map(({ roomId, alias, name }) => (
                        <div
                            key={roomId}
                            className={`room-item ${selectedRoom === roomId ? 'selected' : ''}`}
                            onClick={() => handleRoomSelect(roomId)}
                        >
                            <div className="room-avatar">
                                {name ? name[0].toUpperCase() : alias ? alias[1].toUpperCase() : roomId[0].toUpperCase()}
                            </div>
                            <span>{name || alias || roomId}</span>
                        </div>
                    ))}
                </div>

                <div
                    className="add-room"
                    onClick={() => navigate('/create-room')}
                >
                    Create Room
                </div>


                <div
                    className="add-room"
                    onClick={() => navigate('/explore')}
                >
                    Explore Public Rooms
                </div>

                <div className="people-list">
                    <h4>People</h4>
                    {people.map((person) => (
                        <div key={person.id} className="person-item">
                            <img src={person.attributes.picture[0]} alt={person.firstName} />
                            <span>{person.firstName} {person.lastName}</span>
                        </div>
                    ))}
                </div>
            </div>

            <div className="chat-area">
                {selectedRoom ? (
                    <Chat roomId={selectedRoom} />
                ) : (
                    <div className="select-room">Select a room to start chatting</div>
                )}
            </div>
        </div>
    );
};

export default ChatPage;

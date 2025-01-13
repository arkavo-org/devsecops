import React, { useState, useEffect, useRef } from 'react';
import '../css/ChatPage.css';
import { useKeycloak } from '@react-keycloak/web';
import { useNavigate } from 'react-router-dom';
import { Chat } from './Chat';

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

    const fetchRooms = async (token: string) => {
        const response = await fetch(`${synapseBaseUrl}/_matrix/client/r0/joined_rooms`, {
            headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        });
    
        if (!response.ok) throw new Error('Failed to fetch rooms');
        const data = await response.json();
    
        // Fetch room names and aliases for each room
        const roomDetails = await Promise.all(
            data.joined_rooms.map(async (roomId: string) => {
                let alias: string | null = null;
                let name: string | null = null;
    
                // Fetch room aliases
                try {
                    const aliasResponse = await fetch(`${synapseBaseUrl}/_matrix/client/v3/rooms/${roomId}/aliases`, {
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
                    const nameResponse = await fetch(`${synapseBaseUrl}/_matrix/client/v3/rooms/${roomId}/state/m.room.name`, {
                        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
                    });
    
                    if (nameResponse.ok) {
                        const nameData = await nameResponse.json();
                        name = nameData.name || null;
                    }
                } catch (error) {
                    console.warn(`Failed to fetch name for room ${roomId}:`, error);
                }
    
                return { roomId, alias, name };
            })
        );
    
        // Add the "+" synthetic room
        roomDetails.push({
            roomId: "create-room",
            alias: null,
            name: "+Create Room",
        });
    
        console.log("Room Details:", roomDetails);
        setRooms(roomDetails);
    };
    

    const fetchPeople = async (token: string, searchTerm: string = "", limit = 10) => {
        const requestBody = {
            limit,
            search_term: searchTerm, // Empty string defaults to searching all users
        };

        const response = await fetch(`${synapseBaseUrl}/_matrix/client/v3/user_directory/search`, {
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

        console.log("Response received from User Directory API.");
        const data = await response.json();
        setPeople(data.results || []); // Assumes the response contains a `results` array of users.
    };

    const handleTokenExpiry = () => {
        localStorage.removeItem('matrixAccessToken');
        navigate('/chatauth');
    };

    useEffect(() => {
        const initialize = async () => {
            const storedAccessToken = localStorage.getItem('matrixAccessToken');
            if (storedAccessToken) {
                accessTokenRef.current = storedAccessToken;
                await fetchPeople(storedAccessToken);
                await fetchRooms(storedAccessToken);
            } else {
                navigate('/chatauth');
            }
            setLoading(false);
        };

        initialize();
    }, [navigate]);

    const handleRoomSelect = (roomId: string) => {
        setSelectedRoom(roomId);
        localStorage.setItem('selectedRoom', roomId); // Save the selected room to localStorage
    };

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

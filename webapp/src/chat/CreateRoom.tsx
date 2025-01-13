import React, { useState } from 'react';
import '../css/CreateRoom.css';
import { useNavigate } from 'react-router-dom';

const CreateRoom: React.FC = () => {
    const navigate = useNavigate();
    const [roomName, setRoomName] = useState('');
    const [roomTopic, setRoomTopic] = useState('');
    const [roomVisibility, setRoomVisibility] = useState('private_chat');
    const [error, setError] = useState<string | null>(null);

    const handleCreateRoom = async (e: React.FormEvent) => {
        e.preventDefault(); // Prevent default form submission behavior

        const synapseBaseUrl = import.meta.env.VITE_SYNAPSE_BASE_URL;
        console.log("Creating Room 1");

        const accessToken = localStorage.getItem('matrixAccessToken');
        if (!accessToken) {
            setError("Authentication token not found. Please log in again.");
            return;
        }

        console.log("Creating Room 2");
        const requestBody = {
            name: roomName,
            topic: roomTopic,
            preset: roomVisibility, // "private_chat" or "public_chat"
        };

        try {
            const response = await fetch(`${synapseBaseUrl}/_matrix/client/v3/createRoom`, {
                method: 'POST',
                headers: {
                    Authorization: `Bearer ${accessToken}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
            });

            if (!response.ok) {
                throw new Error(`Failed to create room: ${response.statusText}`);
            }

            const data = await response.json();
            console.log("Room created:", data);

            // Redirect to the chat page
            navigate('/chat');
        } catch (error) {
            console.error("Error creating room:", error);
            setError("Failed to create room. Please try again.");
        }
    };

    return (
        <div className="create-room-container">
            <h2>Create a Room</h2>
            {error && <div className="error-message">{error}</div>}
            <form onSubmit={handleCreateRoom}>
                <label>
                    Room Name:
                    <input
                        type="text"
                        value={roomName}
                        onChange={(e) => setRoomName(e.target.value)}
                        required
                    />
                </label>
                <label>
                    Room Topic:
                    <input
                        type="text"
                        value={roomTopic}
                        onChange={(e) => setRoomTopic(e.target.value)}
                    />
                </label>
                <label>
                    Room Visibility:
                    <select
                        value={roomVisibility}
                        onChange={(e) => setRoomVisibility(e.target.value)}
                    >
                        <option value="private_chat">Private</option>
                        <option value="public_chat">Public</option>
                    </select>
                </label>
                <button type="submit">Create Room</button>
            </form>
        </div>
    );
};

export default CreateRoom;

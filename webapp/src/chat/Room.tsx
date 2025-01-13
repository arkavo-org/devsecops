import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import '../css/Room.css';

interface Room {
    id: string;
    name: string;
}

interface Channel {
    id: string;
    name: string;
}

interface Message {
    id: string;
    content: string;
    sender: string;
    timestamp: string;
}

const Room: React.FC = () => {
    const { roomId } = useParams<{ roomId: string }>();
    const [rooms, setRooms] = useState<Room[]>([]);
    const [channels, setChannels] = useState<Channel[]>([]);
    const [messages, setMessages] = useState<Message[]>([]);
    const [selectedChannel, setSelectedChannel] = useState<string | null>(null);
    const [newMessage, setNewMessage] = useState('');

    useEffect(() => {
        // Fetch user rooms
        fetch('/api/user/rooms') // Replace with actual API endpoint
            .then((res) => res.json())
            .then((data) => setRooms(data))
            .catch((err) => console.error('Failed to fetch rooms:', err));
    }, []);

    useEffect(() => {
        if (roomId) {
            // Fetch channels for the selected room
            fetch(`/api/rooms/${roomId}/channels`) // Replace with actual API endpoint
                .then((res) => res.json())
                .then((data) => setChannels(data))
                .catch((err) => console.error('Failed to fetch channels:', err));
        }
    }, [roomId]);

    useEffect(() => {
        if (selectedChannel) {
            // Fetch messages for the selected channel
            fetch(`/api/channels/${selectedChannel}/messages`) // Replace with actual API endpoint
                .then((res) => res.json())
                .then((data) => setMessages(data))
                .catch((err) => console.error('Failed to fetch messages:', err));
        }
    }, [selectedChannel]);

    const handleSendMessage = () => {
        if (newMessage.trim() && selectedChannel) {
            // Send message to the selected channel
            fetch(`/api/channels/${selectedChannel}/messages`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: newMessage }),
            })
                .then(() => {
                    setMessages((prev) => [
                        ...prev,
                        {
                            id: String(Date.now()),
                            content: newMessage,
                            sender: 'You',
                            timestamp: new Date().toISOString(),
                        },
                    ]);
                    setNewMessage('');
                })
                .catch((err) => console.error('Failed to send message:', err));
        }
    };

    return (
        <div className="room-container">
            {/* Sidebar for rooms */}
            <aside className="sidebar">
                <h2>Rooms</h2>
                <ul>
                    {rooms.map((room) => (
                        <li
                            key={room.id}
                            className={room.id === roomId ? 'active' : ''}
                            onClick={() => window.location.href = `/room/${room.id}`} // Navigate to the room
                        >
                            {room.name}
                        </li>
                    ))}
                </ul>
            </aside>

            {/* Sidebar for channels */}
            <aside className="channel-bar">
                <h2>Channels</h2>
                <ul>
                    {channels.map((channel) => (
                        <li
                            key={channel.id}
                            className={channel.id === selectedChannel ? 'active' : ''}
                            onClick={() => setSelectedChannel(channel.id)}
                        >
                            {channel.name}
                        </li>
                    ))}
                </ul>
            </aside>

            {/* Main content for messages */}
            <main className="message-content">
                <h2>Messages</h2>
                {selectedChannel ? (
                    <>
                        <div className="messages">
                            {messages.map((message) => (
                                <div key={message.id} className="message">
                                    <span className="sender">{message.sender}:</span>
                                    <span className="content">{message.content}</span>
                                    <span className="timestamp">
                                        {new Date(message.timestamp).toLocaleTimeString()}
                                    </span>
                                </div>
                            ))}
                        </div>
                        <div className="message-input">
                            <input
                                type="text"
                                value={newMessage}
                                onChange={(e) => setNewMessage(e.target.value)}
                                placeholder="Type a message..."
                            />
                            <button onClick={handleSendMessage}>Send</button>
                        </div>
                    </>
                ) : (
                    <p>Please select a channel to view messages.</p>
                )}
            </main>
        </div>
    );
};

export default Room;

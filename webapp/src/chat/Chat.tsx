import React, { useState, useEffect, useRef, KeyboardEvent } from 'react';
import '../css/Chat.css';

interface ChatProps {
    roomId: string;
}

interface Message {
    sender: string;
    body: string;
    avatarUrl?: string;
    timestamp?: number;
    eventId?: string; // Include event ID for deduplication
}

export const Chat: React.FC<ChatProps> = ({ roomId }) => {
    const [conversations, setConversations] = useState<Message[]>([]);
    const [prompt, setPrompt] = useState('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const synapseBaseUrl = import.meta.env.VITE_SYNAPSE_BASE_URL;

    const messagesEndRef = useRef<HTMLDivElement | null>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const fetchMessages = async () => {
        const accessToken = localStorage.getItem('matrixAccessToken');
        if (!accessToken) {
            setError('Access token not found.');
            setLoading(false);
            return;
        }

        try {
            const response = await fetch(
                `${synapseBaseUrl}/_matrix/client/v3/sync?filter=${encodeURIComponent(
                    JSON.stringify({ room: { timeline: { limit: 50 } } })
                )}`,
                {
                    headers: {
                        Authorization: `Bearer ${accessToken}`,
                        'Content-Type': 'application/json',
                    },
                }
            );

            if (!response.ok) {
                setError(`Failed to fetch messages: ${response.status} ${response.statusText}`);
                setLoading(false);
                return;
            }

            const data = await response.json();
            const roomData = data.rooms?.join?.[roomId];

            if (!roomData || !roomData.timeline?.events) {
                console.log('No timeline events found for this room.');
                setLoading(false);
                return;
            }

            const messages = roomData.timeline.events
                .filter((event: any) => event.type === 'm.room.message')
                .map((event: any) => ({
                    sender: event.sender,
                    body: event.content?.body || '[Unknown Message]',
                    avatarUrl: event.content?.avatar_url || null,
                    timestamp: event.origin_server_ts || 0,
                    eventId: event.event_id, // Include event ID for deduplication
                }));

            messages.sort((a: Message, b: Message) => (a.timestamp || 0) - (b.timestamp || 0));

            setConversations((prev) => {
                const combinedMessages = [...prev, ...messages];
                const uniqueMessages = Array.from(
                    new Map(combinedMessages.map((msg) => [msg.eventId, msg])).values()
                );
                return uniqueMessages.sort((a, b) => a.timestamp - b.timestamp);
            });

            setLoading(false);
        } catch (err) {
            setError(`Error fetching messages: ${(err as Error).message}`);
            setLoading(false);
        }
    };

    useEffect(() => {
        // Reset state and fetch messages when room changes
        setConversations([]);
        setLoading(true);
        setError(null);
        fetchMessages();
        scrollToBottom();
    }, [roomId]);

    useEffect(() => {
        scrollToBottom(); // Scroll to the bottom on new messages
    }, [conversations]);

    const handleSubmit = async () => {
        if (!prompt.trim()) return;

        const accessToken = localStorage.getItem('matrixAccessToken');
        if (!accessToken) {
            setError('Access token not found.');
            return;
        }

        const txnId = `m${Date.now()}`;
        const newMessage: Message = {
            sender: 'You',
            body: prompt,
            timestamp: Date.now(),
        };

        setConversations((prev) => [...prev, newMessage]); // Optimistically update UI
        setPrompt('');

        try {
            await fetch(
                `${synapseBaseUrl}/_matrix/client/v3/rooms/${roomId}/send/m.room.message/${txnId}`,
                {
                    method: 'PUT',
                    headers: {
                        Authorization: `Bearer ${accessToken}`,
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        msgtype: 'm.text',
                        body: prompt,
                    }),
                }
            );
        } catch (err) {
            setError(`Failed to send message: ${(err as Error).message}`);
        }
    };

    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    if (loading) return <div>Loading messages...</div>;
    if (error) return <div className="error-message">{error}</div>;

    return (
        <main className="chat-container">
            <div className="chat-box">
                <div className="responses-container">
                    {conversations.map((message, index) => (
                        <div key={index} className="chat-message">
                            <div className="message-avatar">
                                {message.avatarUrl ? (
                                    <img src={`${message.avatarUrl}`} alt="Avatar" />
                                ) : (
                                    <div className="default-avatar">{message.sender[0]}</div>
                                )}
                            </div>
                            <div className="message-content">
                                <div className="message-sender">{message.sender}</div>
                                <div className="message-body">{message.body}</div>
                            </div>
                        </div>
                    ))}
                    <div ref={messagesEndRef} /> {/* Auto-scroll anchor */}
                </div>
                <div className="input-container">
                    <textarea
                        className="chat-input"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Enter your message"
                        rows={3}
                    />
                    <button className="send-button" onClick={handleSubmit}>
                        Send
                    </button>
                </div>
            </div>
        </main>
    );
};

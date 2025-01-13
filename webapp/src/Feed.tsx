// src/Feed.tsx
import React, { useEffect, useState, useRef } from 'react';
import axios from 'axios';
import './css/Feed.css';

import FeedItem from './FeedItem';

const Feed: React.FC = () => {
    const [feedItems, setFeedItems] = useState<any[]>([]); // Store raw JSON objects directly
    const [displayedItems, setDisplayedItems] = useState<any[]>([]);
    const [loading, setLoading] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const fetchFeedItems = async () => {
            try {
                const response = await axios.get('https://bsky_bridge.app.codecollective.us/');
                console.log(response.data);

                setFeedItems(response.data.feed); // Store raw feed data
                setDisplayedItems(response.data.feed.slice(0, 10)); // Display first 10 items
            } catch (error) {
                console.error("Error fetching feed data:", error);
            }
        };

        fetchFeedItems();
    }, []);

    const loadMoreItems = () => {
        if (loading || displayedItems.length >= feedItems.length) return;
        setLoading(true);
        setTimeout(() => {
            setDisplayedItems((prevItems) => [
                ...prevItems,
                ...feedItems.slice(prevItems.length, prevItems.length + 10),
            ]);
            setLoading(false);
        }, 1000);
    };

    useEffect(() => {
        const handleScroll = () => {
            if (
                containerRef.current &&
                containerRef.current.scrollTop + containerRef.current.clientHeight >=
                    containerRef.current.scrollHeight - 10
            ) {
                loadMoreItems();
            }
        };

        const currentContainer = containerRef.current;
        currentContainer?.addEventListener('scroll', handleScroll);

        return () => currentContainer?.removeEventListener('scroll', handleScroll);
    }, [feedItems, loading]);

    return (
        <div className="feed-container-outer" ref={containerRef}>
            <div className="feed-container">
                {displayedItems.map((item, index) => (
                    <FeedItem key={index} item={item} /> // Pass raw JSON object directly to FeedItem
                ))}
                {loading && <p>Loading more items...</p>}
            </div>
        </div>
    );
};

export default Feed;

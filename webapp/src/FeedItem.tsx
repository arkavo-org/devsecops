// src/FeedItem.tsx
import React from 'react';
import './css/FeedItem.css';

const FeedItem: React.FC<{ item: any }> = ({ item }) => {
    const { _author, avatar, display_name, handle } = item.post.author;
    const { text, created_at } = item.post.record;
    const image = item.post.embed?.images?.[0]?.fullsize;

    return (
        <div className="feed-item">
            <div className="feed-item-header">
                <img className="avatar" src={avatar} alt={`${display_name}'s avatar`} />
                <div className="author-info">
                    <span className="display-name">{display_name}</span>
                    <span className="author-handle">@{handle}</span>
                    <span className="separator">•</span>
                </div>
            </div>
            <div className="feed-item-header">
                <span className="timestamp">{new Date(created_at).toLocaleString()}</span>
            </div>
            <div className="feed-item-content">
                <p className="content-text">{text}</p>
                {image && <img className="content-image" src={image} alt="Post content" />}
            </div>
            <div className="feed-item-actions">
                <div className="action">
                    <span role="img" aria-label="Like">👍</span>
                    <span>{item.post.like_count || 0}</span>
                </div>
                <div className="action">
                    <span role="img" aria-label="Comment">💬</span>
                    <span>{item.post.reply_count || 0}</span>
                </div>
                <div className="action">
                    <span role="img" aria-label="Repost">🔄</span>
                    <span>{item.post.repost_count || 0}</span>
                </div>
            </div>
        </div>
    );
};

export default FeedItem;

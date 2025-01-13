import React, { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';
import axios from 'axios';
import './css/VideoFeed.css';
import { faL } from '@fortawesome/free-solid-svg-icons';

var soundOn = false;

const VideoFeedItem: React.FC<{ post: any; isActive: boolean}> = ({ post, isActive }) => {
    const { avatar, display_name, handle } = post.author;
    const { text, created_at } = post.record;
    const playlistUrl = post.embed?.playlist;
    const videoRef = useRef<HTMLVideoElement>(null);

    useEffect(() => {
        const video = videoRef.current;

        if (video && playlistUrl) {
            let hls: Hls | null = null;

            if (Hls.isSupported()) {
                hls = new Hls();
                hls.loadSource(playlistUrl);
                hls.attachMedia(video);
                hls.on(Hls.Events.MANIFEST_PARSED, () => {
                    if (isActive) {
                        video.play().catch((error) => {
                            console.error("Autoplay failed:", error);
                        });
                    }
                });
            } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
                video.src = playlistUrl;
                if (isActive) {
                    video.play().catch((error) => {
                        console.error("Autoplay failed:", error);
                    });
                }
            }

            return () => {
                if (hls) {
                    hls.destroy();
                }
            };
        }
    }, [playlistUrl, isActive]);

    useEffect(() => {
        const video = videoRef.current;
        if (video) {
            if (isActive) {
                video.play().catch((error) => {
                    console.error("Video play error:", error);
                });
                video.muted = !soundOn
            } else {
                video.pause();
            }
        }
    }, [isActive]);

    return (
        <div
            className={`video-feed-item ${isActive ? 'active' : ''}`}
            style={{ filter: isActive ? 'brightness(100%)' : 'brightness(50%)' }}
        >
            <div className="video-container">
                <video
                    ref={videoRef}
                    muted
                    playsInline
                    className="video-player"
                    preload="auto"
                />
                <div className="video-overlay">
                    <div className="avatar-container">
                        <img className="avatar" src={avatar} alt={`${display_name}'s avatar`} />
                    </div>
                    <div className="overlay-text">
                        <span className="display-name">{display_name}</span>
                        <span className="author-handle">@{handle}</span>
                        <p className="content-text">{text}</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

const VideoFeed: React.FC = () => {
    const [feedItems, setFeedItems] = useState<any[]>([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const fetchFeedItems = async () => {
            try {
                const response = await axios.get('https://bsky_bridge.app.codecollective.us/video');
                setFeedItems(response.data.feed);
            } catch (error) {
                console.error("Error fetching feed data:", error);
            }
        };

        fetchFeedItems();
    }, []);

    const handleScroll = () => {
        const container = containerRef.current;
        if (!container) return;

        const videos = Array.from(container.children) as HTMLElement[];
        let closestIndex = currentIndex;
        let closestOffset = Infinity;

        videos.forEach((video, index) => {
            const rect = video.getBoundingClientRect();
            const offset = Math.abs(rect.top + rect.height / 2 - window.innerHeight / 2);
            if (offset < closestOffset) {
                closestOffset = offset;
                closestIndex = index;
            }
        });

        if (closestIndex !== currentIndex) {
            setCurrentIndex(closestIndex);
        }
    };

    useEffect(() => {
        const container = containerRef.current;
        if (!container) return;

        container.addEventListener('scroll', handleScroll, { passive: true });
        return () => container.removeEventListener('scroll', handleScroll);
    }, [currentIndex]);

    const handleVideoClick = () => {
        const container = containerRef.current;
        if (!container) return;
        // change mute state
        soundOn = !soundOn;

        const videos = Array.from(container.querySelectorAll('video'));
        videos[currentIndex].muted = !soundOn;
    };

    return (
        <div className="feed-container-outer" ref={containerRef} onClick={handleVideoClick}>
            {feedItems.map((item, index) => (
                <VideoFeedItem key={index} post={item.post} isActive={index === currentIndex} // Pass soundOn state
                />
            ))}
        </div>
    );
};

export default VideoFeed;

// src/AppRoutes.tsx
import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Navbar from './Navbar';
import Feed from './Feed';
import VideoFeed from './VideoFeed';
//import SignIn from './SignIn';
import Privacy from './Privacy';
import ExploreRooms from './chat/ExploreRooms';
import CreateRoom from './chat/CreateRoom';
import Profile from './Profile';
import Room from './chat/Room';
import ChatPage from './chat/ChatPage';
import MatrixAuthForward from './chat/MatrixAuthForward';
import Settings from './Settings';
//import VideoRoom from './VideoRoom';
//import AIView from './AIView';
//import AICompute from './AICompute';
import Events from './Events';
import TDF from './TDF';
import './css/App.css';
import Bluesky from './Bluesky';
 
interface AppRoutesProps { 
    darkMode: boolean;
    onToggleDarkMode: () => void;
}

/*
<button onClick={onToggleDarkMode}>
{darkMode ? '☀️' : '🌒'}
</button>*/

const AppRoutes: React.FC<AppRoutesProps> = ({ darkMode, onToggleDarkMode }) => {
    return (
        <div id="fullpage" className={darkMode ? 'dark-mode' : ''}>
            <Navbar />
            <Routes>
                <Route path="/" element={<Feed />} /> {/* Home route for the feed */}
                <Route path="/privacy" element={<Privacy />} /> {/* Privacy page route */}
                <Route path="/profile" element={<Profile />} />
                <Route path="/chat" element={<ChatPage />} />
                <Route path="/create-room" element={<CreateRoom />} />
                <Route path="/events" element={<Events />} />
                <Route path="/navbar" element={<Navbar />} />
                <Route path="/room/:roomId" element={<Room />} />
                <Route path="/video" element={<VideoFeed />} />
                <Route path="/explore" element={<ExploreRooms />} />
                <Route path="/tdf" element={<TDF />} />
                <Route path="/chatauth" element={<MatrixAuthForward />} />
                {/*<Route path="/video" element={<VideoRoom />} />
                <Route path="/settings" element={<Settings />} />
                <Route path="/aiview" element={<AIView />} />
                <Route path="/aicompute" element={<AICompute />} /> */}
            </Routes>
        </div>
    );
};

export default AppRoutes;

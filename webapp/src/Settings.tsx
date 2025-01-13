import React, { useState, useEffect, ChangeEvent } from 'react';
import './Settings.css';

interface SettingsProps {
    userName: string;
    profilePicture: string;
    darkMode: boolean;
    onUpdateName: (newName: string) => void;
    onUpdateProfilePicture: (newPicture: string) => void;
}

const Settings: React.FC<SettingsProps> = ({ userName, profilePicture, darkMode: initialDarkMode, onUpdateName, onUpdateProfilePicture }) => {
    const [name, setName] = useState(userName);
    const [newProfilePicture, setNewProfilePicture] = useState(profilePicture);
    const [darkMode, setDarkMode] = useState(() => {
        const storedDarkMode = localStorage.getItem('darkMode');
        return storedDarkMode ? storedDarkMode === 'true' : initialDarkMode;
    });

    useEffect(() => {
        localStorage.setItem('darkMode', darkMode.toString());
    }, [darkMode]);

    const handleNameChange = (e: ChangeEvent<HTMLInputElement>) => {
        setName(e.target.value);
    };

    const handleProfilePictureChange = async (e: ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            const reader = new FileReader();
            reader.onloadend = () => {
                try {
                    const base64String = reader.result as string;
                    setNewProfilePicture(base64String);
                    onUpdateProfilePicture(base64String);
                } catch (error) {
                    console.error("Failed to process the file:", error);
                }
            };
            reader.onerror = () => {
                console.error("Error reading the file.");
            };
            reader.readAsDataURL(file);
        }
    };

    const handleSaveName = () => {
        onUpdateName(name);
    };

    const handleToggleDarkMode = () => {
        setDarkMode(!darkMode);
    };

    return (
        <div>
            <div className="settings-container">
                <h2>User Settings</h2>

                <div className="settings-section">
                    <label htmlFor="name">Change Name:</label>
                    <input
                        type="text"
                        id="name"
                        value={name}
                        onChange={handleNameChange}
                    />
                    <button onClick={handleSaveName}>Save</button>
                </div>

                <div className="settings-section">
                    <label htmlFor="profilePicture">Profile Picture:</label>
                    <div className="profile-picture-container">
                        <img src={newProfilePicture} alt="Profile" className="profile-picture" />
                        <input
                            id="profilePicture"
                            type="file"
                            accept="image/*"
                            onChange={handleProfilePictureChange}
                        />
                    </div>
                </div>

                <div className="settings-section">
                    <label>Dark Mode:</label>
                    <label className="toggle-switch">
                        <input
                            type="checkbox"
                            checked={darkMode}
                            onChange={handleToggleDarkMode}
                        />
                        <span className="slider" />
                    </label>
                </div>

            </div>
        </div>
    );
};

export default Settings;

import React, { useEffect, useState } from 'react';
import './css/global.css';
import './css/Profile.css';

const Profile: React.FC = () => {
  const [profileData, setProfileData] = useState<{ [key: string]: any } | null>(null);

  useEffect(() => {
    const storedProfile = localStorage.getItem('userProfile');
    if (storedProfile) {
      setProfileData(JSON.parse(storedProfile));
    }
  }, []);

  if (!profileData) {
    return <div>Loading profile...</div>;
  }

  const renderObject = (obj: { [key: string]: any }, depth: number = 0) => {
    if (depth > 5) return <span>...</span>; // Limit depth to 5 levels

    return (
      <ul>
        {Object.entries(obj).map(([key, value]) => (
          <li key={key}>
            <strong>{key}:</strong>{' '}
            {value && typeof value === 'object' ? (
              Array.isArray(value) ? (
                <ul>
                  {value.map((item, index) => (
                    <li key={index}>{typeof item === 'object' ? renderObject(item, depth + 1) : item || 'N/A'}</li>
                  ))}
                </ul>
              ) : (
                renderObject(value, depth + 1)
              )
            ) : (
              value || 'N/A'
            )}
          </li>
        ))}
      </ul>
    );
  };

  return (
    <div id="app-container">
      <div className="profile-container">
        <h2>User Profile</h2>
        {profileData.picture && (
          <img
            src={profileData.picture}
            alt="Profile"
            style={{ width: 100, height: 100, borderRadius: '50%', marginBottom: '1rem' }}
          />
        )}
        {renderObject(profileData)}
      </div>
    </div>
  );
};

export default Profile;

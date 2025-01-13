import React from 'react';

// Define types for our data structure
type Person = {
  id: string;
  firstName: string;
  lastName: string;
  attributes?: {
    picture?: string[];
  };
};

type SidebarProps = {
  people: Person[];
  selectedPerson: string;
  onPersonSelect: (personId: string) => void;
};

export const Sidebar: React.FC<SidebarProps> = ({ people, selectedPerson, onPersonSelect }) => {
  return (
    <aside className="sidebar">
      <ul>
        {people.map((person: Person) => {
          // Get profile picture or fall back to placeholder
          const profilePicture = 
            person.attributes?.picture?.[0] || '/api/placeholder/50/50';
          
          // Combine first and last names for display
          const displayName = `${person.firstName} ${person.lastName}`;
          
          return (
            <li
              key={person.id}
              className={selectedPerson === person.id ? 'selected' : ''}
              onClick={() => onPersonSelect(person.id)}
            >
              <img
                src={profilePicture}
                alt={displayName}
                className="person-image"
              />
              <span>{displayName}</span>
            </li>
          );
        })}
      </ul>
    </aside>
  );
};

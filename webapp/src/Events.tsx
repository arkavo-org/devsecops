// src/Events.tsx
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './css/Events.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSearch } from '@fortawesome/free-solid-svg-icons';
import FullCalendar from '@fullcalendar/react';
import dayGridPlugin from '@fullcalendar/daygrid';
import interactionPlugin from '@fullcalendar/interaction';
import {  EventApi } from '@fullcalendar/core';

interface EventData {
    id: string;
    title: string;
    description: string;
    category: string;
    date: string;
    location: string;
}

const Events: React.FC = () => {
    const [events, setEvents] = useState<EventData[]>([]);
    const [searchParams, setSearchParams] = useState({ category: '', keyword: '' });
    const [filteredEvents, setFilteredEvents] = useState<EventData[]>([]);

    useEffect(() => {
        const fetchEvents = async () => {
            try {
                // Send authenticated request to Go server
                const token = localStorage.getItem('authToken'); // Adjust based on your auth implementation
                const response = await axios.get('http://localhost:3000/events', {
                    headers: { Authorization: `Bearer ${token}` },
                });

                setEvents(response.data);
                setFilteredEvents(response.data); // Initialize filtered events
            } catch (error) {
                console.error("Error fetching events:", error);
            }
        };

        fetchEvents();
    }, []);

    const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setSearchParams(prevParams => ({ ...prevParams, [name]: value }));
    };

    useEffect(() => {
        // Filter events based on search parameters
        setFilteredEvents(
            events.filter(event =>
                event.category.includes(searchParams.category) &&
                event.title.toLowerCase().includes(searchParams.keyword.toLowerCase())
            )
        );
    }, [searchParams, events]);

    //<b>{eventInfo.timeText}</b>
    const renderEventContent = (eventInfo: EventApi) => (
        <div>
            <i>{eventInfo.title}</i>
        </div>
    );

    return (
        <div className="events-page">
            <div className="search-bar">
                <input
                    type="text"
                    name="category"
                    value={searchParams.category}
                    onChange={handleSearchChange}
                    placeholder="Filter by Category"
                />
                <input
                    type="text"
                    name="keyword"
                    value={searchParams.keyword}
                    onChange={handleSearchChange}
                    placeholder="Search by Keyword"
                />
                <button type="submit" onClick={() => setFilteredEvents(filteredEvents)}>
                    <FontAwesomeIcon icon={faSearch} />
                </button>
            </div>
            <div className="calendar-container">
                <FullCalendar
                    plugins={[dayGridPlugin, interactionPlugin]}
                    initialView="dayGridMonth"
                    events={filteredEvents.map(event => ({
                        id: event.id,
                        title: event.title,
                        start: event.date,
                        description: event.description,
                        location: event.location,
                    }))}
                    eventContent={renderEventContent}
                    selectable={true}
                />
            </div>
        </div>
    );
};

export default Events;

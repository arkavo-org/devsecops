package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/karalabe/go-bluesky"
)

type server struct {
	data       []byte
	lastUpdate time.Time
	mu         sync.Mutex
	did        string
	feed       string
}

const refreshInterval = 5 * time.Minute

func newServer(did, feed string) *server {
	return &server{
		did:  did,
		feed: feed,
	}
}

func (s *server) fetchFeedElements(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Check if the data is still fresh
	if time.Since(s.lastUpdate) < refreshInterval {
		fmt.Println("Using cached data")
		return nil
	}

	// Initialize the Bluesky client
	blueskyHandle := os.Getenv("BLUESKY_HANDLE")
	blueskyAppKey := os.Getenv("BLUESKY_PASSWORD")

	client, err := bluesky.Dial(ctx, bluesky.ServerBskySocial)
	if err != nil {
		return fmt.Errorf("failed to dial Bluesky server: %w", err)
	}
	defer client.Close()

	if err := client.Login(ctx, blueskyHandle, blueskyAppKey); err != nil {
		switch {
		case errors.Is(err, bluesky.ErrMasterCredentials):
			return fmt.Errorf("master credentials are not allowed: %w", err)
		case errors.Is(err, bluesky.ErrLoginUnauthorized):
			return fmt.Errorf("login unauthorized, check credentials: %w", err)
		default:
			return fmt.Errorf("login error: %w", err)
		}
	}

	// Fetch the feed data
	algorithm := fmt.Sprintf("at://%s/app.bsky.feed.generator/%s", s.did, s.feed)
	timeline, err := client.GetTimeline(ctx, algorithm)
	if err != nil {
		return fmt.Errorf("failed to get timeline: %w", err)
	}

	// Serialize the feed data into JSON
	data, err := json.Marshal(timeline)
	if err != nil {
		return fmt.Errorf("failed to marshal timeline data: %w", err)
	}

	// Update the server's data
	s.data = data
	s.lastUpdate = time.Now()
	fmt.Println("Feed data updated")
	return nil
}

func (s *server) handler(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	if err := s.fetchFeedElements(ctx); err != nil {
		http.Error(w, "Failed to fetch feed data", http.StatusInternalServerError)
		fmt.Println("Error fetching feed:", err)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(s.data)
}

func main() {
	// Define the DID and feed
	const did = "did:plc:y7crv2yh74s7qhmtx3mvbgv5"
	const feed = "art-new"

	// Create the server instance
	srv := newServer(did, feed)

	// Set up HTTP server
	http.HandleFunc("/", srv.handler)

	fmt.Println("Starting server at :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Printf("Server failed: %s\n", err)
	}
}

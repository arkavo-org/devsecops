package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
	irc "github.com/thoj/go-ircevent"
)

// OllamaMessage represents a message in the Ollama format.
type OllamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// OllamaConversation represents a conversation in the Ollama format.
type OllamaConversation struct {
	Messages []OllamaMessage `json:"messages"`
}

// Configurations
const (
	ircServer   = "irc:6667"
	ircChannel  = "#ollama"
	ircNickname = "ircbridge"
)

var ircConnection *irc.Connection

func irc_init(router *gin.Engine) {
	// Initialize IRC connection
	ircConnection := irc.IRC(ircNickname, ircNickname)
	ircConnection.VerboseCallbackHandler = true
	ircConnection.Debug = true

	err := ircConnection.Connect(ircServer)
	if err != nil {
		log.Fatalf("Failed to connect to IRC server: %v", err)
	}

	channelMessages := make(map[string][]string)
	// Map to store messages by channel

	// Callback for successful connection
	ircConnection.AddCallback("001", func(e *irc.Event) {
		ircConnection.Join(ircChannel)
		log.Printf("Joined IRC channel: %s", ircChannel)
	})

	// Callback for message events
	ircConnection.AddCallback("PRIVMSG", func(e *irc.Event) {
		channel := e.Arguments[0]
		message := e.Message()
		channelMessages[channel] = append(channelMessages[channel], message)
	})

	// Callback for channel joins
	ircConnection.AddCallback("JOIN", func(e *irc.Event) {
		channel := e.Arguments[0]
		if _, exists := channelMessages[channel]; !exists {
			channelMessages[channel] = []string{}
		}
		log.Printf("Tracking new channel: %s", channel)
	})

	// Run the IRC client in a goroutine
	go ircConnection.Loop()
	ircRoutes := router.Group("irc")
	{

		ircRoutes.GET("/", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{
				"status":  "ok",
				"message": "Org backend is running",
			})
		})

		// Add IRC-related routes to the router
		ircRoutes.POST("/conversation", func(c *gin.Context) {
			handleConversation(c.Writer, c.Request)
		})

		// Route to get all channels
		ircRoutes.GET("/channels", func(c *gin.Context) {
			channels := []string{}
			for channel := range channelMessages {
				channels = append(channels, channel)
			}
			if len(channels) == 0 {
				c.JSON(http.StatusNotFound, gin.H{"error": "No channels found"})
				return
			}
			c.JSON(http.StatusOK, gin.H{"channels": channels})
		})

		// Route to get messages from a specific channel
		ircRoutes.GET("/channels/:channel/messages", func(c *gin.Context) {
			channel := c.Param("channel")
			messages, exists := channelMessages[channel]
			if !exists {
				c.JSON(http.StatusNotFound, gin.H{"error": "Channel not found"})
				return
			}
			c.JSON(http.StatusOK, gin.H{"messages": messages})
		})
	}
}

// handleConversation processes Ollama conversations and sends them to IRC.
func handleConversation(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Invalid request method", http.StatusMethodNotAllowed)
		return
	}

	// Parse the JSON body
	var conversation OllamaConversation
	err := json.NewDecoder(r.Body).Decode(&conversation)
	if err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Send each message to the IRC channel
	for _, message := range conversation.Messages {
		if message.Role == "user" || message.Role == "assistant" {
			sendToIRC(fmt.Sprintf("[%s] %s", message.Role, message.Content))
		}
	}

	w.WriteHeader(http.StatusOK)
	w.Write([]byte("Conversation processed"))
}

// sendToIRC sends a message to the IRC channel.
func sendToIRC(msg string) {
	if ircConnection == nil {
		log.Println("IRC connection is not initialized")
		return
	}

	ircConnection.Privmsg(ircChannel, msg)
	log.Printf("Sent to IRC: %s", msg)
}

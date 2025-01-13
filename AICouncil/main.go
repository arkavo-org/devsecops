package main

import (
	"fmt"
	"log"

	irc "github.com/thoj/go-ircevent"
)

const (
	server   = "localhost:6667" // Replace with your IRC server
	nickname = "TestBot1"       // Bot's nickname
	channel  = "#ollama"        // Channel to join
	//password = "your_password"        // Server password (if required)
)

func main() {
	ircConn := irc.IRC(nickname, nickname)
	ircConn.VerboseCallbackHandler = true
	ircConn.Debug = true

	// Set server password if required
	ircConn.Password = ""

	err := ircConn.Connect(server)
	if err != nil {
		log.Fatalf("Failed to connect to server: %v", err)
	}

	// Handle successful connection
	ircConn.AddCallback("001", func(e *irc.Event) {
		fmt.Println("Connected to server")
		ircConn.Privmsg("NickServ", "IDENTIFY your_nickserv_password") // Identify with NickServ
		ircConn.Join(channel)
	})

	// Handle joining the channel
	ircConn.AddCallback("JOIN", func(e *irc.Event) {
		if e.Nick == nickname {
			fmt.Printf("Joined channel: %s\n", channel)
			ircConn.Privmsg(channel, "Hello, IRC!")
		}
	})

	// Handle messages in the channel
	ircConn.AddCallback("PRIVMSG", func(e *irc.Event) {
		if e.Arguments[0] == channel {
			fmt.Printf("Message in %s from %s: %s\n", channel, e.Nick, e.Message())
			if e.Message() == "hello bot" {
				ircConn.Privmsg(channel, fmt.Sprintf("Hello, %s!", e.Nick))
			}
		}
	})

	ircConn.Loop()
}

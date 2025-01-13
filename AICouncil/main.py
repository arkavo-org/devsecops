import irc.client

class TestBot(irc.client.SimpleIRCClient):
    def on_connect(self, connection, event):
        print("Connected to server")
        connection.join("#testchannel")  # Replace with your channel

    def on_join(self, connection, event):
        print(f"Joined channel: {event.target}")
        connection.privmsg(event.target, "Hello, IRC!")

    def on_disconnect(self, connection, event):
        print("Disconnected from server")

if __name__ == "__main__":
    server = "irc.app.codecollective.us"  # Replace with your server
    port = 6667
    nickname = "TestBot"

    bot = TestBot()
    try:
        bot.connect(server, port, nickname)
        bot.start()
    except irc.client.ServerConnectionError as e:
        print(f"Failed to connect: {e}")

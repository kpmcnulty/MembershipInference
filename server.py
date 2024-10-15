import flwr as fl

# Define and start the server
if __name__ == "__main__":
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=5))

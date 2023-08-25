active_connections = []


async def send_message_to_clients(message: str):
    for connection in active_connections:
        await connection.send_text(message)
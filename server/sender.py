import asyncio
import json

active_connections = []


class Message:
    def __init__(self, type, message, vars, event, id):
        self.type = type
        self.message = message
        self.vars = vars
        self.event = event
        self.id = id

    def to_json(self):
        return {
            "type": self.type,
            "message": self.message,
            "vars": self.vars,
            "event": self.event,
            "id": self.id
        }


def send_message_to_clients(type='info', message='', vars={}, event='message', id='global'):
    for connection in active_connections:
        send_message = Message(type, message, vars, event, id)
        asyncio.run(connection.send_text(json.dumps(send_message.to_json())))

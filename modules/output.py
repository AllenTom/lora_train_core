import json

jsonOut = False
def printJsonOutput(message, vars, event, ):
    if not jsonOut:
        return
    out = {
        "message": message,
        "vars": vars,
        "event": event
    }
    print(json.dumps(out))

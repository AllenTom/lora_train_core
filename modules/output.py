import json

jsonOut = False
detail_output = False

def printJsonOutput(message, vars, event,newline=False ):
    if not jsonOut:
        return
    out = {
        "message": message,
        "vars": vars,
        "event": event
    }
    if newline:
        print("\n"+json.dumps(out))
    else:
        print(json.dumps(out))


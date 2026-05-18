import inputs

print("Press buttons on your controller, Ctrl+C to stop...\n")

while True:
    events = inputs.get_gamepad()
    for e in events:
        if e.ev_type != "Sync":  # filter out noise
            print(f"type={e.ev_type:10} | code={e.code:20} | state={e.state}")
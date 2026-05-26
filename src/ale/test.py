import inputs

while True:
    events = inputs.get_gamepad()

    for e in events:
        if e.ev_type != "Sync":
            print(f"type={e.ev_type:10} | code={e.code:20} | state={e.state}")
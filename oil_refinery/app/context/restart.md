# restart.sh / restart.ps1 / stop.sh / stop.ps1

## `stop_port()` / `Stop-Port()` — matched by port, not process name

Unlike the repo-root `restart.sh`/`restart.ps1` (which match `uvicorn api:app` by cmdline/name,
safe there since it's the only thing that ever runs that command on this machine), this app is
meant to run *alongside* the main `/manual` app, which is also a `uvicorn`/`uvicorn.exe` process.
Killing by whichever process is actually listening on this app's own port (8010 for the backend,
5173 for the frontend dev server) avoids taking down the other app's server by matching a name or
cmdline pattern common to both.

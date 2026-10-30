import http.server
import subprocess
import sys
import threading

ALLOWED_SOURCES = {"63.181.197.45"}
CONTAINERS = ["refinery-demo-app-1", "refinery-demo-web-1"]


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        source = self.client_address[0]
        if source not in ALLOWED_SOURCES:
            print(f"rejected request from {source}", file=sys.stderr, flush=True)
            self.send_response(403)
            self.end_headers()
            return
        body = b'{"warm": false}'
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.server.triggered = True
        threading.Thread(target=self.server.shutdown).start()

    def log_message(self, *args):
        pass


server = http.server.HTTPServer(("0.0.0.0", 80), Handler)
server.triggered = False
server.serve_forever()
server.server_close()
if server.triggered:
    subprocess.run(["docker", "start", *CONTAINERS], check=True)

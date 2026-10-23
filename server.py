"""Web UI for the can-carrier counter. Standard library only.

  python3 server.py [--port 8000]

Open http://localhost:8000 on this machine, or the LAN address it prints on a
phone (same wifi) to use the phone camera."""
import argparse, base64, io, json, os, socket
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from PIL import Image
from pipeline import analyse

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_BYTES = 40 * 1024 * 1024

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split('?')[0] in ('/', '/index.html'):
            self._send(200, open(os.path.join(HERE, 'web', 'index.html'), 'rb').read(), 'text/html; charset=utf-8')
        else:
            self._send(404, b'not found', 'text/plain')

    def do_POST(self):
        if self.path != '/analyze':
            return self._send(404, b'not found', 'text/plain')
        n = int(self.headers.get('Content-Length') or 0)
        if not 0 < n <= MAX_BYTES:
            return self._json(413, dict(error='Image missing or too large (40 MB max).'))
        try:
            img = Image.open(io.BytesIO(self.rfile.read(n)))
            img.load()
        except Exception:
            return self._json(400, dict(error="That file isn't an image I can read (try a JPG or PNG)."))
        try:
            r = analyse(img)
        except Exception as e:
            return self._json(500, dict(error=f'Analysis failed: {e}'))
        buf = io.BytesIO(); r['image'].save(buf, 'JPEG', quality=88)
        r['image'] = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
        self._json(200, r)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), 'application/json')

    def log_message(self, fmt, *args):
        print('%s  %s' % (self.address_string(), fmt % args))

def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('10.255.255.255', 1))
        return s.getsockname()[0]
    except OSError:
        return None

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--host', default='0.0.0.0', help='use 127.0.0.1 to keep it off the network')
    a = ap.parse_args()
    print(f'Local:  http://localhost:{a.port}')
    ip = lan_ip()
    if ip and a.host == '0.0.0.0':
        print(f'Phone (same wifi):  http://{ip}:{a.port}')
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()

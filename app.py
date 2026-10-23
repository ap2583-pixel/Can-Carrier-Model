"""Vercel deployment of the can-carrier counter.

POST /api/analyze  raw image bytes -> counts + annotated picture. The upload and
                   the model's output are saved to a private Vercel Blob store
                   so they can be reused for training.
POST /api/label    {id, true_count} -> the person's own count for that photo.
GET  /api/config   whether an access code is required.

Set ACCESS_CODE in the Vercel project to require a shared code (X-Access-Code)."""
import base64, hmac, io, json, os, secrets, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from vercel import blob
from pipeline import analyse

app = FastAPI()
ACCESS_CODE = os.environ.get('ACCESS_CODE', '')
MAX_BYTES = 4_400_000        # Vercel rejects request bodies over 4.5 MB
PIPELINE = 'autostack-v1'

def _check(request: Request):
    if ACCESS_CODE and not hmac.compare_digest(request.headers.get('x-access-code', ''), ACCESS_CODE):
        raise HTTPException(401, 'Wrong or missing access code.')

@app.get('/api/config')
async def config():
    return dict(code_required=bool(ACCESS_CODE))

@app.post('/api/analyze')
async def analyze(request: Request):
    _check(request)
    raw = await request.body()
    if not 0 < len(raw) <= MAX_BYTES:
        raise HTTPException(413, 'Image missing or too large.')
    try:
        img = Image.open(io.BytesIO(raw)); img.load()
    except Exception:
        raise HTTPException(400, "That file isn't an image I can read (try a JPG or PNG).")
    try:
        r = analyse(img)
    except Exception as e:
        raise HTTPException(500, f'Analysis failed: {e}')

    cid = time.strftime('%Y-%m-%d/%H%M%S-') + secrets.token_hex(4)
    saved = True
    try:   # the person still gets their result if storage hiccups
        await blob.put_blob(f'captures/{cid}.jpg', raw, access='private', content_type='image/jpeg')
        meta = dict(id=cid, pipeline=PIPELINE, saved_at=time.time(), image_size=r['size'], total=r['total'],
                    stacks=r['stacks'], colours=r['colours'])
        await blob.put_blob(f'captures/{cid}.json', json.dumps(meta).encode(), access='private',
                            content_type='application/json')
    except Exception as e:
        print('blob save failed:', repr(e)); saved = False

    buf = io.BytesIO(); r['image'].save(buf, 'JPEG', quality=88)
    r['image'] = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
    return dict(r, id=cid, saved=saved)

@app.post('/api/label')
async def label(request: Request):
    _check(request)
    body = await request.json()
    cid, n = str(body.get('id', '')), body.get('true_count')
    import re
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}/\d{6}-[0-9a-f]{8}', cid) or not isinstance(n, int) or not 0 <= n <= 5000:
        raise HTTPException(400, 'Bad id or count.')
    await blob.put_blob(f'captures/{cid}.label.json', json.dumps(dict(id=cid, true_count=n, at=time.time())).encode(),
                        access='private', content_type='application/json', allow_overwrite=True)
    return dict(ok=True)

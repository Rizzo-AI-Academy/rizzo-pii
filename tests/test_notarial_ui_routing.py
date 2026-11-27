"""La UI notarile usa le proprie API e risorse, senza un backend o dati reali."""

from pathlib import Path
import shutil
import subprocess
import unittest


UI = Path(__file__).resolve().parents[1] / "src/app/ui/notarial.html"


@unittest.skipUnless(shutil.which("node"), "Node necessario per i test JavaScript")
class NotarialUIRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = UI.read_text(encoding="utf-8")
        start = cls.html.index("const NOTARIAL_BASE=")
        cls.transport = cls.html[start:cls.html.index("const $ =", start)]

    def js(self, code, prefix=""):
        harness = r"""
const assert=require('node:assert/strict');
const requests=[];
const window={location:{protocol:'http:',origin:'http://127.0.0.1:5005',
  href:'http://127.0.0.1:5005/notarial/'},PII_BASE:'/notarial',PII_TOKEN:'synthetic-token',
  fetch:async(input,options)=>{requests.push({input,options});return {ok:true};}};
"""
        result = subprocess.run(
            ["node", "--unhandled-rejections=strict"],
            input=harness + self.transport + prefix + "\n(async()=>{\n" + code
            + "\n})().catch(e=>{console.error(e);process.exitCode=1;});",
            text=True, encoding="utf-8", capture_output=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_api_transport_keeps_auth_body_and_cleanup_under_notarial_mount(self):
        self.js(r"""
const body=new FormData();body.append('text','esempio sintetico');
await window.fetch('/analyze',{method:'POST',body});
await window.fetch('/review/synthetic/text',{method:'POST',body:'{}'});
await window.fetch('/notarial/session',{method:'DELETE',keepalive:true});
assert.deepEqual(requests.map(r=>r.input),[
  '/notarial/analyze','/notarial/review/synthetic/text','/notarial/session']);
assert.equal(requests[0].options.body,body);
assert.equal(requests[0].options.headers.get('X-PII-Token'),'synthetic-token');
assert.equal(requests[0].options.cache,'no-store');
assert.equal(requests[2].options.keepalive,true);
await assert.rejects(window.fetch('https://example.invalid/analyze'),/Connessioni esterne/);
assert.equal(requests.length,3);
window.location.protocol='file:';
await assert.rejects(window.fetch('/settings'),/Avvia Rizzo PII/);
assert.equal(requests.length,3);
""")

    def test_preview_images_and_download_stay_in_the_notarial_session(self):
        start = self.html.index("function renderPages(")
        pages = self.html[start:self.html.index("/* barra zoom:", start)]
        start = self.html.index("$('dlpdf').onclick=async()=>{")
        download = self.html[start:self.html.index("/* ---- reverse ---- */", start)]
        dom = r"""
const clicked=[];
const make=tag=>({tag,children:[],style:{},dataset:{},innerHTML:'',
  appendChild(child){this.children.push(child);return child;},
  click(){clicked.push(this.href);}});
const document={createElement:make};
const elements=new Map();
const $=id=>{if(!elements.has(id))elements.set(id,make('div'));return elements.get(id);};
const zoomBar=()=>make('div'),drawOcrBoxes=()=>{},toast=()=>{},tt=x=>x;
let output={doc_id:'synthetic',filename:'synthetic.pdf'};
const buildOutPdf=async()=>output;
"""
        self.js(r"""
const view=make('div');
renderPages(view,{doc_id:'synthetic',n_pages:2});
assert.deepEqual(view.children.slice(1).map(p=>p.children[0].src),[
  '/notarial/doc/synthetic/page/0.png','/notarial/doc/synthetic/page/1.png']);
await $('dlpdf').onclick();
output.file='/notarial/doc/synthetic/file.pdf';
await $('dlpdf').onclick();
assert.deepEqual(clicked,['/notarial/doc/synthetic/file.pdf','/notarial/doc/synthetic/file.pdf']);
""", prefix=dom + pages + download)

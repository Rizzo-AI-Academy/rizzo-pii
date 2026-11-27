"""Contratto della revisione UI: sole fixture sintetiche, Node senza librerie."""

from pathlib import Path
import re
import shutil
import subprocess
import unittest


UI = Path(__file__).resolve().parents[1] / "src" / "app" / "ui" / "notarial.html"

DOM = r"""
const assert=require('node:assert/strict');
class Element {
  constructor(tag='div'){this.tag=tag;this.children=[];this.style={};this.hidden=false;
    this.checked=false;this.value='';this._text='';this.classList={toggle(){},add(){},remove(){}};}
  set textContent(value){this._text=String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(c=>typeof c==='string'?c:c.textContent).join('');}
  set innerHTML(value){this._text='';this.children=[];this._html=value;}
  appendChild(child){if(child.tag==='#fragment')this.children.push(...child.children);else this.children.push(child);return child;}
  append(...children){for(const child of children)this.appendChild(child);}
  setAttribute(name,value){this[name]=value;}
  querySelector(tag){for(const c of this.children){if(c.tag===tag)return c;const nested=c.querySelector?.(tag);if(nested)return nested;}return null;}
  scrollIntoView(){}
  focus(){}
}
const elements=new Map();
const $=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
const document={createElement:tag=>new Element(tag),createTextNode:value=>Object.assign(new Element('#text'),{textContent:value}),
  createDocumentFragment:()=>new Element('#fragment')};
let DATA=null,MAP={},GENERATION=0,REVIEW_BUSY=false,ACK_REVISION=null,REVIEW_CATEGORIES=null;
let REVIEW_OVERRIDES={},REVIEW_CLASSIFICATIONS={},OUT_DOC=null,PDF_CONTROLLER=null,VIEW='prev';
let notifications=[],requests=[],nextResponse=null;
const toast=(...args)=>notifications.push(args);
const show=(el,on)=>{el.hidden=!on;};
const tt=key=>key;
const render=()=>{};
const window={confirm:()=>false};
let fetch=async(url,options)=>{requests.push({url,body:JSON.parse(options.body)});
  return {ok:true,json:async()=>structuredClone(nextResponse)};};
const run=async()=>{throw new Error('Una revisione esistente non deve essere rianalizzata.');};
const taxonomy=[{id:'residenza',parent:null,title:'Residenza',labels:[]},
  {id:'residenza.comune',parent:'residenza',title:'Comune',labels:['CITY']},
  {id:'residenza.indirizzo',parent:'residenza',title:'Indirizzo',labels:['STREET']},
  {id:'nascita',parent:null,title:'Nascita',labels:[]},
  {id:'nascita.luogo',parent:'nascita',title:'Luogo',labels:['CITY']},
  {id:'nascita.data',parent:'nascita',title:'Data',labels:['DATE']}];
const sample=()=>({analysis_id:'synthetic',revision:1,taxonomy,warnings:[],mapping:{},mapping_enabled:false,
  occurrences:[{occurrence_id:'e000001',label:'CITY',value:'Padova',category:'residenza.comune',selected:true,
    review_status:'automatic',context_reason:'Indicatore esplicito.'},
    {occurrence_id:'e000002',label:'CITY',value:'Padova',category:'nascita.luogo',selected:false,
      review_status:'reviewed',context_reason:'Scelta confermata.'}]});
"""


@unittest.skipUnless(shutil.which("node"), "Node necessario per i test JavaScript")
class ReviewUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = UI.read_text(encoding="utf-8")
        start = cls.html.index("function occurrenceId(")
        stop = cls.html.index("\nfunction render(", start)
        cls.review = cls.html[start:stop]
        start = cls.html.index("let PDF_JOB=null;")
        stop = cls.html.index("/* ================================ copia / download", start)
        cls.pdf = cls.html[start:stop]

    def js(self, code, prefix=""):
        result = subprocess.run(["node", "--unhandled-rejections=strict"],
                                input=prefix + DOM + self.review + self.pdf +
                                "\n(async()=>{\n" + code +
                                "\n})().catch(e=>{console.error(e);process.exitCode=1;});",
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_parent_tristate_and_manual_category_options(self):
        self.js(r"""
DATA=sample();
const checkbox=categoryCheckbox(taxonomy[0],[{selected:true},{selected:false}]).children[0];
assert.equal(checkbox.checked,false);assert.equal(checkbox.indeterminate,true);
renderReview();
const row=$('occurrenceList').children[0],select=row.children[2].children[0];
assert.deepEqual(select.children.map(o=>o.value),['residenza.comune','nascita.luogo']);
assert.equal($('reviewCount').textContent,'1 selezionate · 0 da revisionare');
""")

    def test_empty_category_parent_reflects_all_partial_or_no_leaf_choices(self):
        self.js(r"""
DATA=sample();DATA.occurrences=[];
REVIEW_CATEGORIES=['residenza.comune','residenza.indirizzo'];
let box=categoryCheckbox(taxonomy[0],[]).children[0];
assert.equal(box.checked,true);assert.equal(box.indeterminate,false);
REVIEW_CATEGORIES=['residenza.comune'];
box=categoryCheckbox(taxonomy[0],[]).children[0];
assert.equal(box.checked,false);assert.equal(box.indeterminate,true);
REVIEW_CATEGORIES=[];
box=categoryCheckbox(taxonomy[0],[]).children[0];
assert.equal(box.checked,false);assert.equal(box.indeterminate,false);
REVIEW_CATEGORIES=null;
box=categoryCheckbox(taxonomy[0],[]).children[0];
assert.equal(box.checked,true);assert.equal(box.indeterminate,false);
""")

    def test_type_filter_counts_all_pending_and_keeps_hidden_export_gate(self):
        self.js(r"""
DATA=sample();DATA.occurrences[0].review_status='pending';
DATA.occurrences.push({occurrence_id:'e000003',label:'DATE',value:'01/01/2001',category:'nascita.data',
  selected:false,review_status:'pending',context_reason:'Data senza contesto.'});
renderReview();
assert.match($('ambiguitySummary').textContent,/Comune: 1/);
assert.match($('ambiguitySummary').textContent,/Data: 1/);
assert.equal($('reviewType').children.length,3);
$('reviewType').value='CITY';$('reviewType').onchange();
assert.equal($('occurrenceList').children.length,2);
$('onlyAmbiguous').checked=true;$('onlyAmbiguous').onchange();
assert.equal($('occurrenceList').children.length,1);
assert.equal(DATA.occurrences[2].review_status,'pending');
assert.equal(confirmExport(),false);assert.equal(requests.length,0);
DATA.occurrences[0].review_status='reviewed';renderReview();
assert.equal($('reviewType').value,'CITY');
assert.match($('occurrenceList').textContent,/Nessuna occorrenza con questi filtri/);
assert.match($('ambiguitySummary').textContent,/Data: 1/);
assert.doesNotMatch($('ambiguitySummary').textContent,/Comune: 1/);
assert.equal(confirmExport(),false);
""")

    def test_pending_context_excerpt_is_literal_and_does_not_slice_unicode_offsets(self):
        self.js(r"""
DATA=sample();DATA.occurrences[0].review_status='pending';
DATA.occurrences[0].start=999;DATA.occurrences[0].end=1000;
DATA.occurrences[0].context_excerpt='😀 luogo sintetico <img src=x> residente a Padova.';
renderReview();
const row=$('occurrenceList').children[0],excerpt=row.children[4];
assert.equal(excerpt.className,'occ-context');
assert.equal(excerpt.textContent,'Contesto: 😀 luogo sintetico <img src=x> residente a Padova.');
assert.equal(excerpt.children.length,0);
assert.equal(requests.length,0);
""")

    def test_linked_field_and_appearance_share_only_their_own_decisions(self):
        self.js(r"""
DATA=sample();DATA.occurrences[0].alias_group='widget-appearance-1';
DATA.occurrences[1].alias_group='widget-appearance-1';
DATA.occurrences[0].review_status='pending';
DATA.occurrences.push({...DATA.occurrences[1],occurrence_id:'e000003',alias_group:'other-widget'});
REVIEW_OVERRIDES={e000003:true};REVIEW_CLASSIFICATIONS={e000003:'residenza.comune'};
const updates=[];updateReview=changes=>updates.push(changes);
renderReview();
const row=$('occurrenceList').children[0],box=row.children[0].children[0];
box.checked=false;box.onchange();
assert.deepEqual(updates.pop(),{overrides:{e000003:true,e000001:false,e000002:false}});
const choices=row.children[2],select=choices.children[0];
select.value='nascita.luogo';select.onchange();
assert.deepEqual(updates.pop(),{classifications:{e000003:'residenza.comune',e000001:'nascita.luogo',e000002:'nascita.luogo'}});
choices.children[1].onclick();
assert.deepEqual(updates.pop(),{overrides:{e000003:true,e000001:true,e000002:true}});
assert.match(row.children[3].textContent,/Campo e apparenza collegati/);
""")

    def test_occurrence_override_posts_current_revision_without_value_key(self):
        self.js(r"""
DATA=sample();nextResponse=sample();nextResponse.revision=2;
nextResponse.occurrences[0].selected=false;nextResponse.occurrences[0].review_status='reviewed';
await updateReview({overrides:{e000001:false}});
assert.equal(requests[0].url,'/review/synthetic');
assert.equal(requests[0].body.revision,1);
assert.deepEqual(requests[0].body.overrides,{e000001:false});
assert.equal(DATA.revision,2);assert.equal(DATA.occurrences[0].selected,false);
assert.equal(DATA.occurrences[1].selected,false);assert.equal(REVIEW_BUSY,false);
assert.deepEqual(MAP,{});
""")

    def test_stale_review_response_does_not_replace_new_document(self):
        self.js(r"""
DATA=sample();let complete;
fetch=()=>new Promise(resolve=>{complete=resolve;});
const pending=updateReview({overrides:{e000001:false}});
GENERATION++;DATA={analysis_id:'new-document',revision:1};REVIEW_BUSY=false;
complete({ok:true,json:async()=>sample()});await pending;
assert.equal(DATA.analysis_id,'new-document');assert.equal(DATA.revision,1);
""")

    def test_pending_and_unverified_contents_require_explicit_review(self):
        self.js(r"""
DATA=sample();DATA.occurrences[0].review_status='pending';
assert.equal(confirmExport(),false);assert.equal(requests.length,0);
DATA.occurrences[0].review_status='reviewed';DATA.warnings=['Fotografia non verificata.'];
window.confirm=()=>false;assert.equal(confirmExport(),false);assert.equal(ACK_REVISION,null);
window.confirm=()=>true;assert.equal(confirmExport(),true);assert.equal(ACK_REVISION,1);
DATA.revision=2;window.confirm=()=>false;assert.equal(confirmExport(),false);
""")

    def test_pdf_uses_review_snapshot_and_shares_preview_artifact(self):
        self.js(r"""
DATA=sample();nextResponse={doc_id:'output',filename:'documento_selettivo.pdf'};
const first=buildOutPdf(),second=buildOutPdf();assert.equal(first,second);
const result=await first;
assert.equal(requests.length,1);assert.equal(requests[0].url,'/pdf/preview');
assert.deepEqual(requests[0].body,{analysis_id:'synthetic',revision:1,ack_unverified:false});
assert.equal(result._revision,1);assert.equal((await buildOutPdf()).doc_id,'output');
assert.equal(requests.length,1);
""")

    def test_denied_pdf_export_clears_pending_job(self):
        self.js(r"""
DATA=sample();DATA.warnings=['Immagine non verificata.'];
assert.equal(await buildOutPdf(),null);assert.equal(PDF_JOB,null);assert.equal(requests.length,0);
""")

    def test_source_preview_preserves_unicode_and_uses_context_color(self):
        start = self.html.index("function render(keepPdf)")
        stop = self.html.index("/* ================== viste a destra", start)
        rendering = self.html[start:stop]
        code = DOM.replace("const render=()=>{};", "") + rendering + r"""
const L='it',T={it:{chars:n=>String(n)}};
const colors=id=>({bg:id,bd:id,tx:id});
const escapeHtml=value=>String(value);
DATA=sample();Object.assign(DATA,{n_entities:1,n_unique:1,n_chars:22,by_source:{},excluded_tags:[],
  segments:[{t:'😀 Nato a '},{t:'Padova',label:'CITY',ph:'[CITY_1]',category:'nascita.luogo',
    occurrence_id:'e000001',selected:false,review_status:'reviewed'},{t:', qui.\r\n'}]});
render(true);
assert.equal($('prev').textContent,'😀 Nato a Padova, qui.\r\n');
assert.equal($('prev').children[1].style.background,'nascita.luogo');
assert.equal($('prev').children[1].className,'ph dim');
"""
        result = subprocess.run(["node"], input=code, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fetch_authentication_and_network_boundary(self):
        start = self.html.index("const NOTARIAL_BASE=")
        stop = self.html.index("const $ =", start)
        wrapper = self.html[start:stop]
        code = """
const assert=require('node:assert/strict');let calls=[];
const window={PII_TOKEN:'synthetic-token',location:{href:'http://127.0.0.1:5005/',origin:'http://127.0.0.1:5005'},
 fetch:async(input,options)=>{calls.push({input,options});return {ok:true};}};
""" + wrapper + """
(async()=>{
await window.fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'}});
assert.equal(calls[0].input,'/notarial/analyze');
assert.equal(calls[0].options.headers.get('X-PII-Token'),'synthetic-token');
assert.equal(calls[0].options.cache,'no-store');
await assert.rejects(window.fetch('https://example.invalid/'),/Connessioni esterne/);
assert.equal(calls.length,1);
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
        result = subprocess.run(["node"], input=code, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_file_page_never_fetches_local_files_as_api(self):
        start = self.html.index("const NOTARIAL_BASE=")
        stop = self.html.index("const $ =", start)
        wrapper = self.html[start:stop]
        code = """
const assert=require('node:assert/strict');let calls=0;
const window={location:{protocol:'file:',href:'file:///synthetic/index.html',origin:'null'},
 fetch:async()=>{calls++;return {ok:true};}};
""" + wrapper + """
(async()=>{
await assert.rejects(window.fetch('/settings'),/Avvia Rizzo PII/);
assert.equal(calls,0);
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
        result = subprocess.run(["node"], input=code, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_settings_errors_are_visible_and_success_loads_tags(self):
        start = self.html.index("function connectionState(")
        stop = self.html.index("function renderOcrPill(", start)
        code = DOM + """
let BACKEND_ERROR='',TAGS=[],TAGS_LOADED=false,SETTINGS={},EXCL=new Set(),MAPPING=false;
window.location={protocol:'http:'};
let tagRenders=0;const renderMapping=()=>{},renderOcrPill=()=>{},renderTags=()=>{tagRenders++;};
""" + self.html[start:stop] + """
(async()=>{
fetch=async()=>({ok:false,status:403,json:async()=>({error:'no'})});
assert.equal(await loadSettings(),false);
assert.equal(TAGS_LOADED,false);assert.equal($('backendWarning').hidden,false);
assert.match($('tgList').textContent,/ricarica/);assert.equal($('go').disabled,true);
fetch=async()=>({ok:true,json:async()=>({tags:[{tag:'CITY'}],excluded_tags:[],mapping_enabled:false})});
assert.equal(await loadSettings(),true);
assert.equal(TAGS_LOADED,true);assert.equal(TAGS[0].tag,'CITY');
assert.equal(tagRenders,1);
assert.equal($('backendWarning').hidden,true);assert.equal($('go').disabled,false);
window.location.protocol='file:';fetch=async()=>{throw new TypeError('Failed to fetch');};
assert.equal(await loadSettings(),false);assert.equal($('backendOpen').hidden,false);
assert.match($('tgList').textContent,/aperta come file/);
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
        result = subprocess.run(["node"], input=code, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_scripts_compile_and_no_dictionary_persistence(self):
        for script in re.findall(r"<script[^>]*>([\s\S]*?)</script>", self.html):
            result = subprocess.run(["node", "--check"], input=script, text=True,
                                    capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotRegex(self.html, r"localStorage\.(?:getItem|setItem)\(['\"]pii_map")
        # The standard mode owns its legacy dictionary key. Merely opening the
        # notarial mode must neither persist a new dictionary nor erase that key.
        self.assertNotIn("localStorage.removeItem('pii_map')", self.html)
        self.assertIn("keepalive:true", self.html)


if __name__ == "__main__":
    unittest.main()

"""Offline review renders physical board history without rewriting evidence."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from blackboard_mvp.preview import write_preview


def fixture_manifest():
    boards=[dict(board_id=f'{side}_{layer}',label=f'{side} {layer}',assembly_id=f'{side}_assembly',layer=layer,roi=[0 if side=='left' else 120,0 if layer=='outer' else 50,100,50]) for side in ('left','right') for layer in ('outer','inner')]
    frames=[]
    for i in range(3):
        states=[]
        for b in boards:
            hidden=b['board_id']=='left_inner' and i==1
            states.append(dict(b,screen_roi=b['roi'],crop_roi=b['roi'],crop_path=f"crops/{b['board_id']}/f{i:04d}.png",mask_path=f"masks/{b['board_id']}/f{i:04d}.png",visible_regions=[] if hidden else [b['roi']],visible_fraction=0 if hidden else 1,visibility='occluded' if hidden else 'visible',identity_status='confirmed',motion_status='moving' if i==1 else 'stationary',version_id=b['board_id']+':v0000'))
        frames.append(dict(frame_id=f'f{i:04d}',timestamp_ms=i*1000,pts_time_ms=i*1000,image_path=f'frames/f{i:04d}.png',boards=states,time_status='verified_pts'))
    frames[1]['boards'][3].update(identity_status='identity_uncertain',visibility_status='provisional_geometry_only',mechanical_prior={'status':'inconsistent_requires_review','residual_pixels':12})
    def event(kind,board,i,tag):
        return dict(event_id=tag,kind=kind,board_id=board,assembly_id=board.split('_')[0]+'_assembly',observed_at_ms=i*1000,interval_ms=[(i-1)*1000,i*1000],before_frame_id=f'f{i-1:04d}',after_frame_id=f'f{i:04d}',before_crop_path=f'crops/{board}/f{i-1:04d}.png',after_crop_path=f'crops/{board}/f{i:04d}.png',version_id=board+':v0000',selected_reasons=['synthetic'],uncertainty=[],change_regions=[])
    events=[event('board_motion_event','left_inner',1,'move-inner'),event('visibility_event','left_inner',1,'occlude-inner'),event('board_motion_event','left_outer',1,'move-outer'),event('board_content_change_event','right_outer',2,'content-right')]
    events[-1].update(comparison_reference_semantics='per_pixel_last_confirmed_visible_evidence',comparison_references=[dict(frame_id='f0000',crop_path='crops/right_outer/f0000.png',timestamp_ms=0,board_local_regions=[[1,1,4,4]]),dict(frame_id='f0001',crop_path='crops/right_outer/f0001.png',timestamp_ms=1000,board_local_regions=[[20,10,4,4]])])
    return dict(schema_version='mechanical-fixture',config=dict(boards=boards,assemblies=[dict(assembly_id='left_assembly',viewport=[0,0,100,100]),dict(assembly_id='right_assembly',viewport=[120,0,100,100])]),source=dict(duration_ms=3000,width=220,height=100),frames=frames,events=events,versions=[dict(version_id=b['board_id']+':v0000',board_id=b['board_id'],frame_id='f0000',timestamp_ms=0,crop_path=f"crops/{b['board_id']}/f0000.png",visibility='visible') for b in boards],position_history=[dict(s,frame_id=f['frame_id'],timestamp_ms=f['timestamp_ms']) for f in frames for s in f['boards']],writing_runs=[])


class MechanicalPreviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='mechanical-preview-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def test_embedded_manifest_preserves_all_positions_and_versions(self):
        manifest=fixture_manifest()
        html=write_preview(manifest,self.root).read_text()
        payload=re.search(r'<script id="manifest-data" type="application/json">(.*?)</script>',html,re.S).group(1)
        self.assertEqual(json.loads(payload),manifest)
        self.assertIn('物理黑板位置与遮挡完整历史',html)
        self.assertIn('机械移动',html)
        self.assertIn("connect-src 'none'",html)
        self.assertNotIn('<script src=',html)

    def test_browser_script_groups_components_filters_events_and_preserves_evidence(self):
        node=shutil.which('node')
        if node is None:
            self.skipTest('Optional JavaScript smoke test requires local node')
        html=write_preview(fixture_manifest(),self.root).read_text()
        script=re.findall(r'<script>\s*(.*?)</script>',html,re.S)[-1]
        harness=r'''
const assert=require('assert'),vm=require('vm');let input='';process.stdin.on('data',x=>input+=x);process.stdin.on('end',()=>{
const p=JSON.parse(input);
class E{constructor(tag){this.tag=tag;this.children=[];this._text='';this.value='';this.listeners={};this.style={};this.attributes={};this.classList={add:()=>{},remove:()=>{}};this.complete=false;}
set textContent(t){this._text=String(t);}get textContent(){return this._text+this.children.map(c=>c.textContent).join('');}
appendChild(c){this.children.push(c);if(this.tag==='select'&&this.children.length===1)this.value=c.value;return c;}
replaceChildren(...c){this.children=c;this._text='';}setAttribute(k,v){this.attributes[k]=v;}addEventListener(k,v){this.listeners[k]=v;}click(){if(this.listeners.click)this.listeners.click();}scrollIntoView(){}get firstElementChild(){return this.children[0];}}
const nodes={};const document={getElementById:id=>nodes[id]||(nodes[id]=new E(['frame-select','event-board','event-kind','version-board','position-board'].includes(id)?'select':'div')),createElement:t=>new E(t),createElementNS:(ns,t)=>{const n=new E(t);n.namespaceURI=ns;return n;}};
document.getElementById('manifest-data').textContent=JSON.stringify(p.manifest);
vm.runInNewContext(p.script,{document,console,Map,Number,Math,JSON,String,Array,decodeURIComponent});
assert.strictEqual(nodes['board-crops'].children.length,2,'two physical components');
assert(nodes['board-crops'].textContent.includes('left_assembly / inner'));
assert.strictEqual(nodes.positions.children.length,3,'all observations retained');
nodes['event-kind'].value='motion';nodes['event-kind'].listeners.change();assert.strictEqual(nodes.timeline.children.length,2);
nodes['event-kind'].value='content';nodes['event-kind'].listeners.change();assert.strictEqual(nodes.timeline.children.length,1);
nodes['version-board'].value='left_inner';nodes['version-board'].listeners.change();
const images=n=>[...(n.tag==='img'?[n.src]:[]),...n.children.flatMap(images)];
assert(images(nodes['version-evidence']).includes('crops/left_inner/f0000.png'),'motion cannot rewrite old version evidence');
assert(!images(nodes['version-evidence']).includes('crops/left_inner/f0001.png'));
nodes.timeline.children[0].click();assert(nodes.evidence.textContent.includes('不同局部区域分别使用其最后一次已确认可见的历史帧'));assert(images(nodes.evidence).includes('crops/right_outer/f0000.png'));assert(images(nodes.evidence).includes('crops/right_outer/f0001.png'));assert(nodes.evidence.textContent.includes('查看原帧 f0000'));assert(nodes.evidence.textContent.includes('查看原帧 f0001'));
nodes.next.click();assert(nodes['board-crops'].textContent.includes('估计可见区域，未确认可比较'));assert(nodes['board-crops'].textContent.includes('inconsistent_requires_review'));assert(nodes['frame-caption'].textContent.includes('均未进行人体分割'));const inner=nodes['board-crops'].children[0].children[2].children[1];
assert(inner.textContent.includes('完全遮挡'),'hidden rear board must show pause message');
assert(!images(inner).includes('crops/left_inner/f0001.png'),'foreground cannot masquerade as inner content');
nodes['position-board'].value='left_inner';nodes['position-board'].listeners.change();assert.strictEqual(nodes.positions.children.length,3);
nodes.positions.children[1].click();assert(nodes['position-evidence'].textContent.includes('完全遮挡'));
assert(nodes['position-evidence'].textContent.includes('left_inner:v0000'));
console.log('offline DOM semantic checks passed');});
'''
        result=subprocess.run([node,'-e',harness],input=json.dumps(dict(script=script,manifest=fixture_manifest())),text=True,capture_output=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('semantic checks passed',result.stdout)

    def test_labels_cannot_inject_html_or_remote_requests(self):
        manifest=fixture_manifest()
        manifest['config']['boards'][0]['label']='</script><img src="https://invalid.example">'
        html=write_preview(manifest,self.root).read_text()
        self.assertNotIn('</script><img',html)
        self.assertIn('\\u003c/script\\u003e',html)
        self.assertIn('localPath',html)
        self.assertIn('textContent',html)


if __name__=='__main__':
    unittest.main()

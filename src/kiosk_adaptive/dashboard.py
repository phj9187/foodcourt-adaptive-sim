"""Read-only live experiment dashboard; no new dependencies and no test-set feedback."""
import argparse
from datetime import datetime
from html import escape
import json
import math
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading


def comparison(pairs):
    if not pairs:
        return None
    baseline = sum(p['baseline_score'] for p in pairs)/len(pairs)
    ai = sum(p['ai_score'] for p in pairs)/len(pairs)
    delta = ai-baseline
    # Preserve negative improvements. Nonpositive denominators are not ordinary
    # percentage growth: show their signed score difference instead.
    rate = 100*delta/baseline if baseline>1e-8 else None
    return dict(baseline=baseline, ai=ai, delta=delta, percent=rate,
                customer_delta=sum(p['customer_delta'] for p in pairs)/len(pairs),
                pairs=len(pairs))


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def snapshot(output):
    root = Path(output)
    manifest = read_json(root/'manifest.json', {})
    cfg = manifest.get('config', {})
    jobs = []
    for objective in manifest.get('objectives', []):
        for rep in range(cfg.get('replicas', 1)):
            path = root/objective['id']/f'rep_{rep}'
            live = read_json(path/'live.json', {})
            progress = read_json(path/'progress.json', {})
            final = read_json(path/'summary.json')
            pairs = [] if live.get('test_stable') else read_json(path/'paired_test.json', [])
            history = live.get('history', [])
            # Backward-compatible read-only viewing of old pilot outputs.
            if not history:
                old = read_json(path/'learning_history.json', [])
                history = [dict(block=h['block'],delta=h['validation_delta'],percent=None)
                           for h in old if 'validation_delta' in h]
            latest = history[-1] if history else None
            jobs.append(dict(id=objective['id'],replica=rep,objective=objective,
                block=live.get('block',progress.get('block',0)),
                total=cfg.get('training_blocks',0),
                phase='complete' if final else live.get('phase','training' if progress else 'waiting'),
                history=history, latest=latest,
                final_stable=live.get('test_stable') or comparison([p for p in pairs if not p['changes']]),
                final_changed=live.get('test_changed') or comparison([p for p in pairs if p['changes']]),
                evaluated=live.get('evaluated'),evaluation_total=live.get('evaluation_total'),
                error=live.get('error')))
    return dict(experiment=root.name,config=cfg,jobs=jobs,
                refreshed=datetime.now().astimezone().isoformat(timespec='seconds'))


def number(value, suffix='', digits=4):
    if value is None or not math.isfinite(value):
        return '—'
    return f'{value:+.{digits}f}{suffix}'


def chart(points, key, title, suffix=''):
    valid = [(p['block'],p.get(key)) for p in points if p.get(key) is not None]
    if not valid:
        reason='기준점수 0 이하 · 점수 차이 Δ로 확인' if points and key=='percent' else '검증 결과 대기'
        return '<div class="empty">'+escape(title)+'：'+reason+'</div>'
    width,height,left,top=480,155,55,18
    right,bottom=465,117
    xs=[x for x,y in valid];ys=[y for x,y in valid]+[0.]
    lo,hi=min(ys),max(ys)
    pad=max((hi-lo)*.15, .1 if key=='percent' else .001)
    lo-=pad;hi+=pad
    xmax=max(1,max(xs))
    px=lambda x:left+(right-left)*x/xmax
    py=lambda y:bottom-(bottom-top)*(y-lo)/(hi-lo)
    coordinates=' '.join(f'{px(x):.2f},{py(y):.2f}' for x,y in valid)
    dots=''.join(f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="3"><title>운영기간 {x}: {number(y,suffix)}</title></circle>' for x,y in valid)
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">
    <title>{escape(title)}</title><text x="{left}" y="12">{escape(title)}</text>
    <line x1="{left}" x2="{right}" y1="{py(0):.2f}" y2="{py(0):.2f}" class="zero"/>
    <text x="2" y="{top+8}">{number(hi,suffix,2 if key=='percent' else 3)}</text>
    <text x="2" y="{bottom}">{number(lo,suffix,2 if key=='percent' else 3)}</text>
    <polyline points="{coordinates}"/><g>{dots}</g>
    <text x="{left}" y="145">0</text><text x="340" y="145">학습 운영기간 → {xmax}</text>
    </svg>'''


STYLE = '''
*{box-sizing:border-box}body{margin:0;background:#f3f6fb;color:#182538;font:15px system-ui,-apple-system,sans-serif}
.wrap{max-width:1450px;margin:auto;padding:26px}h1{font-size:27px;margin:0 0 8px}h2{font-size:19px;margin:0}
.sub,.muted{color:#627187}.notice{background:#fff7df;border-left:4px solid #e4ad34;padding:12px;margin:16px 0}
.formula{background:white;border:1px solid #dfe6ef;border-radius:14px;padding:18px 22px;margin:18px 0}.formula-title{font-weight:700;margin-bottom:9px}.equation{font-size:22px;line-height:1.7;font-weight:600;color:#304b9c;overflow-wrap:anywhere}.legend{line-height:1.8;margin-top:10px;color:#45566d}.legend b{color:#182538}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}.stat{background:white;border-radius:12px;padding:15px 22px;min-width:170px}
.stat b{display:block;font-size:25px;margin-top:5px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}
.card{background:white;border:1px solid #dfe6ef;border-radius:14px;padding:18px;overflow:hidden}.head{display:flex;justify-content:space-between;gap:10px;align-items:start}
.badge{font-size:12px;background:#edf3ff;padding:5px 9px;border-radius:20px;white-space:nowrap}.bar{height:6px;background:#edf1f7;border-radius:4px;margin:12px 0}.bar i{display:block;height:100%;background:#526de8;border-radius:4px}
.metrics{display:flex;gap:20px;margin:13px 0}.metric b{font-size:25px;display:block}.up{color:#087a63}.down{color:#c0444b}.neutral{color:#627187}
svg{width:100%;max-height:170px;display:block}svg text{font-size:10px;fill:#66778a}svg polyline{fill:none;stroke:#536fe5;stroke-width:2}svg circle{fill:#536fe5}svg .zero{stroke:#aeb9c9;stroke-dasharray:4 3}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px}th,td{text-align:right;padding:8px 4px;border-top:1px solid #e9eef5}th:first-child,td:first-child{text-align:left}.empty{padding:30px 10px;color:#8793a4;background:#f8faff;border-radius:8px;margin-top:12px}.small{font-size:12px;margin-top:9px}.error{color:#b62435}
@media(max-width:600px){.grid{grid-template-columns:1fr}.wrap{padding:14px}.metrics{gap:12px}}
'''


def render(data, refresh=False):
    jobs=data['jobs'];complete=sum(j['phase']=='complete' for j in jobs)
    active=sum(j['phase'] not in ('complete','waiting') for j in jobs)
    done=sum(j['block'] for j in jobs);total=sum(j['total'] for j in jobs)
    header=f'''<h1>푸드코트 AI · 실시간 도입 효과</h1>
    <div class="sub">실험 {escape(data['experiment'])} · {escape(data['refreshed'])}</div>
    <section class="formula"><div class="formula-title">보상을 위한 하루 평가점수식 · F</div>
    <div class="equation">F = λ × (aT<sup>p</sup> + bP<sup>p</sup> + cW<sup>p</sup>)<br>
    ＋ (1 − λ) × [q × (R − K) / (N × 6,000) − (1 − q) × D / (N × 6,000)]</div>
    <div class="legend"><b>고객점수:</b> T = 취향 적합도, P = 가격 만족도, W = 대기 만족도.
    세 값은 미구매자를 0으로 포함한 하루 방문자 평균이고, 지수 p는 평균을 구한 뒤 적용해요.<br>
    <b>가게 성과:</b> R = 매출, K = 판매분 식재료비, D = 폐기비용(모두 원), N = 하루 방문자 수(최소 1).
    가게 성과는 방문자당 6,000원으로 나누어 크기를 맞춰요.<br>
    <b>각 카드의 계수:</b> a = 취향, b = 가격, c = 대기(a+b+c=1), λ = 고객 비중,
    q = 판매기여 비중, p = 지수.<br>
    고객 비중 λ가 크면 고객점수를, q가 크면 가게 성과 중 판매기여를 더 중시해요.
    여러 날의 F를 합쳐 학습하고, 비교표에서는 하루 평균 F를 사용해요.</div></section>
    <div class="stats"><div class="stat">학습 운영기간<b>{done:,} / {total:,}</b></div>
    <div class="stat">모델 완료<b>{complete} / {len(jobs)}</b></div>
    <div class="stat">현재 진행 중<b>{active}</b></div></div>
    <div class="notice">음수는 성능 악화로 그대로 표시합니다. 향상률 = (AI 점수 − 미개입 점수) ÷ 미개입 점수 × 100.
    미개입 점수가 0 이하이거나 0에 가까우면 % 대신 점수 차이 Δ를 확인하세요.<br>
    곡선은 검증용 푸드코트의 측정값이며 검증할 때 갱신됩니다. 화면은 3초마다 갱신됩니다.
    최종 시험 결과는 아래 표에 별도로 표시하며 학습·모델 선택에 되돌려 쓰지 않습니다.</div>'''
    phases=dict(waiting='대기',training='학습',validation='검증',testing='최종 시험',complete='완료',failed='오류')
    cards=[]
    for j in jobs:
        latest=j['latest'] or {};delta=latest.get('delta');percent=latest.get('percent')
        color='up' if delta is not None and delta>=0 else 'down' if delta is not None else 'neutral'
        progress=100*j['block']/max(1,j['total'])
        o=j['objective'];labels=dict(taste='a 취향',affordability='b 가격',waiting='c 대기',customer_share='λ 고객 비중',profit_share='q 판매기여 비중',power='p 지수')
        params=' · '.join(f'{labels[k]} {o[k]:.2f}' for k in labels)
        phase=phases.get(j['phase'],j['phase'])
        if j['evaluated'] is not None and j['phase'] in ('validation','testing'):
            phase+=f" {j['evaluated']}/{j['evaluation_total']}"
        rows=[]
        for label,point in [('최근 검증',j['latest']),('최종 · 구성 고정',j['final_stable']),('최종 · 구성 변경',j['final_changed'])]:
            point=point or {}
            rows.append(f"<tr><td>{label}</td><td>{number(point.get('baseline'))}</td><td>{number(point.get('ai'))}</td><td>{number(point.get('delta'))}</td><td>{number(point.get('percent'),'%',2)}</td></tr>")
        cards.append(f'''<section class="card"><div class="head"><h2>{escape(j['id'])}<span class="muted"> · 반복 {j['replica']+1}</span></h2><span class="badge">{escape(phase)}</span></div>
        <div class="small muted">{escape(params)}</div><div class="bar"><i style="width:{progress:.2f}%"></i></div>
        <div class="muted small">학습 운영기간 {j['block']} / {j['total']} · 최근 검증 {latest.get('block','—')}번째</div>
        <div class="metrics"><div class="metric">검증 향상률<b class="{color}">{number(percent,'%',2)}</b></div>
        <div class="metric">검증 점수 차이 Δ<b class="{color}">{number(delta)}</b></div></div>
        {chart(j['history'],'percent','검증 향상률 (%)','%')}
        {chart(j['history'],'delta','검증 점수 차이 Δ')}
        <table><thead><tr><th>측정 구분</th><th>미개입</th><th>AI</th><th>Δ</th><th>향상률</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
        <div class="small muted">검증·시험 환경이 달라 두 결과의 차이를 그대로 학습 추세로 해석하지 않습니다.</div>
        {('<div class="error">'+escape(j['error'])+'</div>') if j.get('error') else ''}</section>''')
    if not jobs:cards=['<div class="empty">실험 시작 대기 중입니다. manifest.json이 생성되면 모델 목록이 표시됩니다.</div>']
    content='<main class="wrap">'+header+'<div class="grid">'+''.join(cards)+'</div></main>'
    script='''<script>setInterval(async()=>{try{const r=await fetch('/fragment',{cache:'no-store'});if(r.ok)document.querySelector('main').outerHTML=await r.text()}catch(e){}},3000)</script>''' if refresh else ''
    return '<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>푸드코트 AI 실시간 대시보드</title><style>'+STYLE+'</style></head><body>'+content+script+'</body></html>'


class LiveDashboard:
    """Colab output updated in place by a read-only background thread."""
    def __init__(self, output, interval=3.):
        self.output=Path(output);self.interval=interval;self.stop_event=threading.Event()

    def start(self):
        from IPython.display import HTML, display
        self.handle=display(HTML(render(snapshot(self.output))),display_id=True)
        def loop():
            while not self.stop_event.wait(self.interval):
                self.handle.update(HTML(render(snapshot(self.output))))
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        if hasattr(self,'thread'):self.thread.join(timeout=5)
        if hasattr(self,'handle'):
            from IPython.display import HTML
            self.handle.update(HTML(render(snapshot(self.output))))


def serve(output, port=8765):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            page=render(snapshot(output),refresh=True)
            if self.path=='/fragment':page=page.split('<main',1)[1].split('</main>',1)[0];page='<main'+page+'</main>'
            elif self.path!='/':self.send_error(404);return
            body=page.encode('utf-8');self.send_response(200)
            self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    print(f'http://127.0.0.1:{port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--port',type=int,default=8765)
    args=p.parse_args();serve(args.output,args.port)

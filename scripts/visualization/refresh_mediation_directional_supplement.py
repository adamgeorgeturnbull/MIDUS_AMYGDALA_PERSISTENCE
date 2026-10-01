"""Refresh only Table S11 and its notes from aggregate directional mediation CSVs.
Preserves other tables, figures, comments, and DOCX package members.
Usage: python script.py INPUT.docx OUTPUT.docx --results results/tables/03b_persistence_age_mediation
"""
import argparse
import csv
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E

NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

def text(node):
    return ''.join(node.xpath('.//w:t/text()', namespaces=NS))

def set_text(node, value):
    ts = node.xpath('.//w:t', namespaces=NS)
    if not ts:
        raise ValueError('Missing text node')
    ts[0].text = value
    for t in ts[1:]:
        t.text = ''

def replace_text(node, old, new):
    ts = node.xpath('.//w:t', namespaces=NS)
    combined = ''.join(t.text or '' for t in ts)
    if combined.count(old) != 1:
        raise ValueError('Expected one exact text match: ' + old)
    start = combined.index(old); end = start + len(old); offset = 0
    spans = []
    for t in ts:
        value = t.text or ''; stop = offset + len(value)
        if stop > start and offset < end:
            spans.append((t, offset, value))
        offset = stop
    if len(old) == len(new):
        for t, offset, value in spans:
            lo=max(start-offset,0); hi=min(end-offset,len(value))
            t.text=value[:lo]+new[offset+lo-start:offset+hi-start]+value[hi:]
    else:
        for index,(t,offset,value) in enumerate(spans):
            lo=max(start-offset,0); hi=min(end-offset,len(value))
            t.text=value[:lo]+(new if index==0 else '')+value[hi:]
            t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
    if text(node) != combined.replace(old,new,1):
        raise ValueError('Replacement verification failed')


def number(x):
    return f'{float(x):.5f}'.replace('-', '−')

def pvalue(x):
    v = float(x)
    return '< .001' if v < .001 else f'{v:.4f}'.lstrip('0')

def refresh(src, dst, results):
    with ZipFile(src) as z:
        parts = {n: z.read(n) for n in z.namelist()}; infos = z.infolist()
    r = E.fromstring(parts['word/document.xml'])
    tables = r.xpath('//w:body/w:tbl', namespaces=NS)
    if len(tables) != 11:
        raise ValueError('Expected eleven supplement tables')
    table = tables[10]; rows = table.xpath('./w:tr', namespaces=NS)
    if len(rows) != 27:
        raise ValueError('Expected 26 S11 model rows')
    age = {'primary_c5page':'Neuroscience age', 'sensitivity_c2page':'Diary age'}
    outcome = {'neg_persist_crossrun_mean_z_L':'Left negative persistence', 'PA_score':'Diary PA','NA_score':'Diary NA','NA_score_log':'Diary log NA'}
    path = {'a':'a','b':'b','c_prime':'c′','c_total':'c'}
    expected = {}
    with (results/'paths.csv').open() as f:
        for row in csv.DictReader(f):
            key = (age[row['specification']],path[row['path']],outcome[row['outcome']])
            if key in expected: raise ValueError('Duplicate path')
            expected[key] = [str(int(float(row['n']))), number(row['beta']), '['+number(row['ci_low'])+', '+number(row['ci_high'])+']',pvalue(row['p_one_tailed'])]
    with (results/'mediation.csv').open() as f:
        for row in csv.DictReader(f):
            if row['boot_significant_one_tailed'].lower() != 'false':
                raise ValueError('Indirect inference changed; review prose before refreshing')
            key = (age[row['specification']],'a × b',outcome[row['outcome']])
            expected[key] = [str(int(float(row['n']))),number(row['indirect_ab']),'['+number(row['boot_ci_low'])+', '+number(row['boot_ci_high'])+']','—']
    seen = set()
    for row in rows[1:]:
        cells = row.xpath('./w:tc',namespaces=NS); key = tuple(text(c) for c in cells[:3])
        if key in seen or key not in expected: raise ValueError('Unexpected table row')
        seen.add(key)
        for c,val in zip(cells[3:],expected[key]):
            if text(c) != val: set_text(c,val)
    if seen != set(expected): raise ValueError('Missing table row')
    set_text(rows[0].xpath('./w:tc',namespaces=NS)[6], 'p (1-sided)')
    notes = [p for p in r.xpath('//w:body/w:p',namespaces=NS) if text(p).startswith('a = age–persistence;')]
    if len(notes)!=1:raise ValueError('Missing S11 note')
    set_text(notes[0], 'a = age–persistence; b = persistence–affect adjusted for age; c′ = direct age–affect; c = total age–affect. Path tests are one-tailed: a < 0; b < 0 for PA and b > 0 for NA; c and c′ > 0 for PA and < 0 for NA. Reported CIs remain two-sided 95% intervals: normal-Wald for paths and percentile bootstrap for indirect effects, based on 2,000 successful family-level draws per specification. Indirect effects were tested in the predicted direction (positive for PA, negative for NA) using one-sided 95% percentile bounds. No indirect effect met the directional significance criterion. Dashes indicate that indirect inference uses bootstrap bounds rather than a path p value. Assessment order varied across participants, so these are statistical indirect associations and are not interpreted causally.')
    intro = [p for p in r.xpath('//w:body/w:p',namespaces=NS) if text(p).startswith('All coefficients are unstandardized.')]
    if len(intro)!=1:raise ValueError('Missing supplement intro')
    replace_text(intro[0], 'p (2-sided) is also shown.', 'p (2-sided) is also shown except for the directional mediation analyses in Table S11.')
    parts['word/document.xml'] = E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True)
    with ZipFile(dst,'w') as z:
        for info in infos:z.writestr(info,parts[info.filename])
    return len(seen)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('input',type=Path);a.add_argument('output',type=Path);a.add_argument('--results',type=Path,required=True);args=a.parse_args()
    if args.input.resolve()==args.output.resolve():raise ValueError('Use a separate output file')
    print('Refreshed',refresh(args.input,args.output,args.results),'mediation rows.')

#!/usr/bin/env python3
"""Append OLS/Pearson comparison tables using existing aggregate exports only.

Preserves existing supplement tables, figures, and other DOCX package parts.
No model fitting or participant input access. Run from project root with
--document PATH; adds S7–S11 once, backing up the document before modification.
"""
import argparse,csv,hashlib,json,math,shutil
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS={'w':W}
FAMILIES=['00a_diary_panas','00d_erq_context','01_affect_age','02_persistence_affect','03_persistence_age','04_fc_affect','04ex_fc_affect_antpost','05_fc_persistence']
LABELS={'PA_score':'Diary PA','NA_score':'Diary NA','NA_score_log':'Diary log NA','C5SPGP':'PANAS PA','C5SPGN':'PANAS NA','C5SPGN_log':'PANAS log NA','C5SER':'Reappraisal','C5SES':'Suppression','C5PAGE':'Age (neuroscience)','C2PAGE':'Age (diary)','RA5PAGE':'Age (neuroscience)','RA5SPGP':'PANAS PA','RA5SPGN':'PANAS NA','RA5SPGN_log':'PANAS log NA','neg_persist_crossrun_mean_z_L':'Left negative persistence','neg_persist_crossrun_mean_z_R':'Right negative persistence','pos_persist_crossrun_mean_z_L':'Left positive persistence','pos_persist_crossrun_mean_z_R':'Right positive persistence'}

def label(x):
    if x in LABELS:return LABELS[x]
    if '_image_mean_z' in x:return x.replace('_image_mean_z',' image-to-face persistence (Fisher z)').replace('_',' ')
    return x.replace('l_amyg','Left amygdala').replace('r_amyg','Right amygdala').replace('ant_vmPFC','anterior vmPFC').replace('post_vmPFC','posterior vmPFC').replace('_neg_vs_neu',' (negative − neutral)').replace('_pos_vs_neu',' (positive − neutral)').replace('_',' ')

def number(x,d=5):
    try:return f'{float(x):.{d}f}'.replace('-','−') if math.isfinite(float(x)) else '—'
    except (ValueError,TypeError):return '—'

def pvalue(x):
    try:
        v=float(x)
        if not math.isfinite(v):return '—'
        return '< .001' if v<.001 else f'{v:.4f}'.lstrip('0')
    except (ValueError,TypeError):return '—'

def interval(row,prefix=''):
    if row.get(prefix+'ci_status')!='ok':return 'Unavailable'
    lo,hi=[float(row[prefix+k]) for k in ['ci_low','ci_high']]
    if not all(map(math.isfinite,[lo,hi])) or lo>hi:raise ValueError('Invalid saved CI')
    if float(row[prefix+'ci_level'])!=95 or row[prefix+'ci_sidedness']!='two-sided':raise ValueError('Unexpected CI specification')
    return '['+number(lo)+', '+number(hi)+']'

def element(parent,tag,**attrs):
    node=E.SubElement(parent,'{'+W+'}'+tag)
    for k,v in attrs.items():node.set('{'+W+'}'+k,str(v))
    return node

def paragraph(text,bold=False,size=18):
    p=E.Element('{'+W+'}p');props=element(p,'pPr');element(props,'spacing',after=70)
    if bold:element(props,'keepNext')
    r=element(p,'r');rp=element(r,'rPr');element(rp,'rFonts',ascii='Times New Roman',hAnsi='Times New Roman');element(rp,'sz',val=size)
    if bold:element(rp,'b')
    element(r,'t').text=text
    return p

def table(headers,rows):
    widths=[2200,2200,1300,450,850,1800,800,850]
    t=E.Element('{'+W+'}tbl');pr=element(t,'tblPr');element(pr,'tblW',w=sum(widths),type='dxa');element(pr,'tblLayout',type='fixed')
    borders=element(pr,'tblBorders')
    for side in ['top','bottom','insideH']:element(borders,side,val='single',sz=4,color='BBBBBB')
    grid=element(t,'tblGrid')
    for width in widths:element(grid,'gridCol',w=width)
    for index,values in enumerate([headers]+rows):
        tr=element(t,'tr');rp=element(tr,'trPr');element(rp,'cantSplit')
        if index==0:element(rp,'tblHeader')
        for value,width in zip(values,widths):
            cell=element(tr,'tc');cp=element(cell,'tcPr');element(cp,'tcW',w=width,type='dxa')
            if index==0:element(cp,'shd',fill='E8EDF2')
            cell.append(paragraph(str(value),bold=index==0))
    return t

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--document',required=True,type=Path)
    args=ap.parse_args();root=Path.cwd();doc=args.document
    with ZipFile(doc) as z:entries=[(i,z.read(i.filename)) for i in z.infolist()]
    parts={i.filename:b for i,b in entries};tree=E.fromstring(parts['word/document.xml']);body=tree.find('w:body',NS)
    if 'Table S7.' in ''.join(tree.xpath('//w:t/text()',namespaces=NS)):raise SystemExit('Comparison tables already present; no changes made.')
    source_manifest=[];long=[]
    def read(path):
        rows=list(csv.DictReader(path.open(newline='')))
        source_manifest.append(dict(source=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(rows)))
        return rows
    def group(path):
        parts=path.relative_to(root).parts
        return ' / '.join(parts[parts.index('tables')+1:-1]).replace('_',' ')
    def format_row(path,r,method,moderation=False):
        prefix='interaction_' if moderation else ''
        estimate=r['beta_interaction'] if moderation else r['r'] if method=='Pearson' else r['beta']
        p=r['p_interaction'] if moderation else r['p']
        # Only 00a omits the explicit two-tailed column; its tests are two-sided.
        two=p if moderation or '00a_diary_panas' in path.parts else r.get('p_two_tailed','')
        predictor=label(r['predictor'])+(' × '+label(r['moderator']) if moderation else '')
        record=dict(table_method=method,source=str(path.relative_to(root)),**r)
        long.append(record)
        return [group(path),predictor,label(r['outcome']),r['n'],number(estimate),interval(r,prefix),pvalue(p),pvalue(two)]
    specs=[]
    for method,filename,num in [('OLS','regressions.csv',7),('Pearson','correlations.csv',8)]:
        rows=[]
        for family in FAMILIES:
            # Anchor scope to the existing MLM families, but retain method-specific rows/Ns.
            for mlm in sorted((root/'results/tables'/family).rglob('mlm.csv')):
                if 'full_sample' in mlm.parts:continue
                path=mlm.with_name(filename)
                for r in read(path):rows.append(format_row(path,r,method))
        path=root/'results/tables/sensitivity_novel_sample'/('novel_sample_'+filename)
        for r in read(path):rows.append(format_row(path,r,method))
        specs.append((num,f'M3 {method} associations and sensitivity comparisons',rows,'β' if method=='OLS' else 'r'))
    rows=[]
    for family in ['06_persistence_affect_moderation','07_fc_affect_moderation']:
        for mlm in sorted((root/'results/tables'/family).rglob('moderation_mlm.csv')):
            if 'full_sample' in mlm.parts:continue
            path=mlm.with_name('moderation_ols.csv')
            for r in read(path):rows.append(format_row(path,r,'OLS interaction',True))
    specs.append((9,'M3 OLS moderation comparisons',rows,'β interaction'))
    rows=[]
    for fd in ['fd_mean_across_runs','fd_neg_mean']:
        for method,suffix in [('Pearson','correlations'),('OLS','regressions')]:
            path=root/'results/tables/00e_motion_check'/f'part1_{fd}_{suffix}.csv'
            for r in read(path):
                v=format_row(path,r,method);v[0]=method+' / motion association';rows.append(v)
        path=root/'results/tables/00e_motion_check'/f'part2_motion_controlled_{fd}.csv'
        for r in read(path):
            if r['analysis_type']=='ols':
                v=format_row(path,r,'OLS');v[0]='OLS / adjusted for '+fd.replace('_',' ');rows.append(v)
    specs.append((10,'M3 correlation and OLS motion comparisons',rows,'r / β'))
    rows=[]
    for family in ['02_persistence_affect','03_persistence_age','04_fc_affect','05_fc_persistence']:
        for ols in sorted((root/'MR1_validation/results/tables'/family).rglob('regressions.csv')):
            if 'full_sample' in ols.parts:continue
            path=ols.with_name('correlations.csv')
            for r in read(path):rows.append(format_row(path,r,'Pearson'))
    specs.append((11,'MR1 zero-order correlation comparisons',rows,'r'))
    # Insert before the existing figure's page break, preserving all prior XML nodes.
    children=list(body);index=next((i for i,n in enumerate(children) if ''.join(n.xpath('.//w:t/text()',namespaces=NS)).startswith('Supplementary Figure S1.')),len(children)-1)
    if index>0 and children[index-1].find('.//w:br',NS) is not None:index-=1
    nodes=[]
    for num,title,rows,estimate in specs:
        br=E.Element('{'+W+'}p');element(element(br,'r'),'br',type='page');nodes.append(br)
        nodes.append(paragraph(f'Table S{num}. {title}',True,26))
        note='Comparison estimates retain each method’s own sample size and saved inference. p (analysis) preserves the test used in the source export; p (2-sided) provides the corresponding two-sided probability. All CIs are two-sided 95%; OLS uses Student-t intervals and Pearson correlations use Fisher-z intervals. OLS coefficients are unstandardized. Comparisons are not used to select the primary model or redefine hypotheses. Broader pre-QC full_sample outputs are excluded.'
        if num==9:note+=' These are two-sided interaction tests; there is no zero-order-correlation analogue of an adjusted interaction. OLS cannot resolve invalid MLM inference.'
        if num==10:note+=' r denotes a zero-order correlation and β an OLS coefficient; these have different scales. Motion-adjusted regressions have no zero-order equivalent.'
        nodes.append(paragraph(note,size=20))
        nodes.append(table(['Analysis','Predictor / interaction','Outcome','N',estimate,'95% CI','p (analysis)','p (2-sided)'],rows))
    for offset,node in enumerate(nodes):body.insert(index+offset,node)
    # Original tables must remain exactly intact at the beginning.
    old=E.fromstring(parts['word/document.xml'])
    assert [E.tostring(t) for t in old.findall('.//w:tbl',NS)]==[E.tostring(t) for t in tree.findall('.//w:tbl',NS)][:len(old.findall('.//w:tbl',NS))]
    audit=root/'workspace_snapshots/author_review'/('supplement_comparisons_'+datetime.now().strftime('%Y%m%d_%H%M%S'));audit.mkdir(parents=True)
    shutil.copy2(doc,audit/doc.name)
    with ZipFile(doc,'w') as z:
        for info,b in entries:z.writestr(info,E.tostring(tree,xml_declaration=True,encoding='UTF-8',standalone=True) if info.filename=='word/document.xml' else b)
    with ZipFile(doc) as z:
        assert z.testzip() is None
        for info,b in entries:
            if info.filename!='word/document.xml':assert z.read(info.filename)==b
    out=root/'results/tables/supplementary_method_comparisons';out.mkdir(exist_ok=True)
    columns=list(dict.fromkeys(k for row in long for k in row))
    with (out/'comparisons.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(long)
    report={'table_rows':{f'S{n}':len(rows) for n,_,rows,_ in specs},'sources':source_manifest,'backup':str(audit.relative_to(root))}
    (out/'sources.json').write_text(json.dumps({k:v for k,v in report.items() if k!='backup'},indent=2)+'\n')
    (audit/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Added comparison tables:',report['table_rows']);print('Preserved existing tables and figure. Backup:',audit)
if __name__=='__main__':main()

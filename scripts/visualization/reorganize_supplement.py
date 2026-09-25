#!/usr/bin/env python3
"""Reorganize the reviewed supplement from aggregate exports; never fits models.
Requires the existing 11-table supplement; preserves task/mediation tables and
all embedded figures. Writes a backup, reorganized DOCX and coverage manifest.
"""
import argparse,copy,csv,hashlib,json,shutil
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from add_supplement_method_comparisons import FAMILIES,label,number,pvalue,interval,paragraph,element,W,NS

TITLES={1:'M3 main results: mixed-effects models',2:'MR1 main results: OLS models',3:'M3 sensitivity results: mixed-effects models',4:'MR1 sensitivity results: OLS models',5:'M3 main results: OLS and correlation comparisons',6:'MR1 main results: OLS and correlation comparisons',7:'M3 sensitivity results: OLS and correlation comparisons',8:'MR1 sensitivity results: OLS and correlation comparisons',9:'Contextual associations: MLM, OLS, and correlations',10:'Task-condition means and paired differences',11:'Statistical indirect associations and component paths'}
G={'00a_diary_panas':'Diary/PANAS convergence','00d_erq_context':'ERQ context','01_affect_age':'Age and affect','02_persistence_affect':'Persistence and affect','03_persistence_age':'Age and persistence','04_fc_affect':'Connectivity and affect','04ex_fc_affect_antpost':'Anterior/posterior contrast','05_fc_persistence':'Connectivity and persistence','06_persistence_affect_moderation':'Exploratory persistence moderation','07_fc_affect_moderation':'Exploratory connectivity moderation','sensitivity_novel_sample':'Potential-overlap exclusion subsample','neuro_sample':'Neuroscience subsample','sensitivity_panas':'PANAS','sensitivity_other_persistence':'Positive persistence','sensitivity_positive_persistence':'Positive persistence','sensitivity_vmpfc_persistence':'vmPFC persistence','sensitivity_vmPFC':'vmPFC persistence','sensitivity_right_hemisphere':'Right hemisphere','sensitivity_right_amygdala':'Right amygdala','sensitivity_roi_activations':'Regional activation','sensitivity_neg_condition':'Negative condition','sensitivity_pos_vs_neu':'Positive minus neutral'}

def table(headers,rows,widths):
    t=E.Element('{'+W+'}tbl');pr=element(t,'tblPr');element(pr,'tblW',w=sum(widths),type='dxa');element(pr,'tblLayout',type='fixed')
    borders=element(pr,'tblBorders')
    for side in ['top','bottom','insideH']:element(borders,side,val='single',sz=4,color='BBBBBB')
    grid=element(t,'tblGrid')
    for width in widths:element(grid,'gridCol',w=width)
    for index,row in enumerate([headers]+rows):
        tr=element(t,'tr');rp=element(tr,'trPr');element(rp,'cantSplit')
        if index==0:element(rp,'tblHeader')
        for value,width in zip(row,widths):
            cell=element(tr,'tc');cp=element(cell,'tcPr');element(cp,'tcW',w=width,type='dxa')
            if index==0:element(cp,'shd',fill='E8EDF2')
            for line in str(value).split('\n'):cell.append(paragraph(line,bold=index==0,size=18))
    return t

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--document',required=True,type=Path);args=ap.parse_args()
    root=Path.cwd();doc=args.document
    with ZipFile(doc) as z:entries=[(i,z.read(i.filename)) for i in z.infolist()]
    parts={i.filename:b for i,b in entries};tree=E.fromstring(parts['word/document.xml']);body=tree.find('w:body',NS)
    oldtables=body.findall('w:tbl',NS)
    if len(oldtables)!=11 or TITLES[1] in ''.join(tree.xpath('//w:t/text()',namespaces=NS)):raise SystemExit('Expected prior 11-table layout; no changes made.')
    oldrows=lambda t:[[ ''.join(c.xpath('.//w:t/text()',namespaces=NS)) for c in tr.findall('w:tc',NS)] for tr in t.findall('w:tr',NS)[1:]]
    sources={};records=[]
    def read(path):
        rel=str(path.relative_to(root));sources[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
        return list(csv.DictReader(path.open(newline='')))
    def add(path,r,method,cohort='M3',kind='association',motion=None):
        rel=path.relative_to(root);parts=rel.parts;fam=parts[parts.index('tables')+1];tail=parts[parts.index('tables')+2:-1]
        context=fam.startswith(('00a','00d'))
        sensitivity=(not context) and (r['outcome'].endswith('_log') or any('sensitivity' in x for x in [fam,*tail]) or 'neuro_sample' in tail or fam=='04ex_fc_affect_antpost' or motion is not None)
        scope='context' if context else 'sensitivity' if sensitivity else 'main'
        group=' / '.join([G.get(fam,fam.replace('_',' '))]+[G.get(x,x.replace('_',' ')) for x in tail])
        if motion:group=motion
        if r['outcome'].endswith('_log') and not context:group+=' / log-NA robustness'
        interaction=kind=='interaction';prefix='interaction_' if interaction else ''
        p=r['p_interaction'] if interaction else r['p']
        p2=p if interaction or fam=='00a_diary_panas' else r.get('p_two_tailed','')
        value=r['beta_interaction'] if interaction else r['r'] if method=='Pearson' else r['beta']
        formatted=[r['n'],number(value),interval(r,prefix),pvalue(p),pvalue(p2)]
        key=(cohort,fam,tuple(tail),motion,r['predictor'],r.get('moderator',''),r['outcome'])
        records.append(dict(key=key,scope=scope,cohort=cohort,method=method,kind=kind,group=group,predictor=label(r['predictor'])+(' × '+label(r['moderator']) if interaction else ''),outcome=label(r['outcome']),values=formatted,source=str(rel),status=r.get(prefix+'ci_status','')))
    for fam in FAMILIES:
        for mlm in sorted((root/'results/tables'/fam).rglob('mlm.csv')):
            if 'full_sample' in mlm.parts:continue
            for method,filename in [('MLM','mlm.csv'),('OLS','regressions.csv'),('Pearson','correlations.csv')]:
                path=mlm.with_name(filename)
                for r in read(path):add(path,r,method)
    for method,suffix in [('MLM','mlm'),('OLS','regressions'),('Pearson','correlations')]:
        path=root/'results/tables/sensitivity_novel_sample'/f'novel_sample_{suffix}.csv'
        for r in read(path):add(path,r,method)
    for fam in ['06_persistence_affect_moderation','07_fc_affect_moderation']:
        for mlm in sorted((root/'results/tables'/fam).rglob('moderation_mlm.csv')):
            if 'full_sample' in mlm.parts:continue
            for method,filename in [('MLM','moderation_mlm.csv'),('OLS','moderation_ols.csv')]:
                path=mlm.with_name(filename)
                for r in read(path):add(path,r,method,kind='interaction')
    for fd in ['fd_mean_across_runs','fd_neg_mean']:
        for method,suffix in [('MLM','mlm'),('OLS','regressions'),('Pearson','correlations')]:
            path=root/'results/tables/00e_motion_check'/f'part1_{fd}_{suffix}.csv'
            for r in read(path):add(path,r,method,motion='Motion association / '+fd.replace('_',' '))
        path=root/'results/tables/00e_motion_check'/f'part2_motion_controlled_{fd}.csv'
        for r in read(path):add(path,r,r['analysis_type'].upper(),motion='Adjusted for '+fd.replace('_',' '))
    for fam in ['02_persistence_affect','03_persistence_age','04_fc_affect','05_fc_persistence']:
        for ols in sorted((root/'MR1_validation/results/tables'/fam).rglob('regressions.csv')):
            if 'full_sample' in ols.parts:continue
            for method,filename in [('OLS','regressions.csv'),('Pearson','correlations.csv')]:
                path=ols.with_name(filename)
                for r in read(path):add(path,r,method,'MR1')
    # Match every prior table's numerical rows, before changing layout.
    def counts(cohort,method,kind=None,motion=False,association=False):
        chosen=[r for r in records if r['cohort']==cohort and r['method']==method and (kind is None or r['kind']==kind)]
        chosen=[r for r in chosen if ('00e_motion_check' in r['source'])==motion]
        if association:chosen=[r for r in chosen if r['kind']=='association']
        return Counter(tuple(r['values']) for r in chosen)
    assert counts('M3','MLM',association=True)==Counter(tuple(r[3:8]) for r in oldrows(oldtables[0]))
    assert counts('MR1','OLS',association=True)==Counter(tuple(r[3:8]) for r in oldrows(oldtables[5]))
    for idx,method in [(6,'OLS'),(7,'Pearson')]:assert counts('M3',method,association=True)==Counter(tuple(r[3:8]) for r in oldrows(oldtables[idx]))
    assert counts('MR1','Pearson',association=True)==Counter(tuple(r[3:8]) for r in oldrows(oldtables[10]))
    assert counts('M3','OLS',kind='interaction')==Counter(tuple(r[3:8]) for r in oldrows(oldtables[8]))
    assert Counter(tuple(r['values'][:4]) for r in records if r['method']=='MLM' and r['kind']=='interaction')==Counter(tuple(r[3:7]) for r in oldrows(oldtables[2]))
    assert counts('M3','MLM',motion=True)==Counter(tuple(r[3:8]) for r in oldrows(oldtables[4]))
    assert counts('M3','OLS',motion=True)+counts('M3','Pearson',motion=True)==Counter(tuple(r[3:8]) for r in oldrows(oldtables[9]))
    buckets={i:[] for i in range(1,10)};groups=defaultdict(dict)
    for r in records:
        if r['method'] in groups[r['key']]:raise ValueError('Duplicate method identity')
        groups[r['key']][r['method']]=r
        if r['scope']=='context':continue
        if (r['cohort']=='M3' and r['method']=='MLM') or (r['cohort']=='MR1' and r['method']=='OLS'):
            target=(1 if r['cohort']=='M3' else 2)+(2 if r['scope']=='sensitivity' else 0)
            buckets[target].append([r['group'],r['predictor'],r['outcome'],*r['values']])
    def cell(r):
        if r is None:return 'Not applicable'
        n,b,ci,p,p2=r['values'];symbol='r' if r['method']=='Pearson' else 'β interaction' if r['kind']=='interaction' else 'β'
        return f'N = {n}; {symbol} = {b}\n95% CI {ci}\np (analysis) {p}; p (2-sided) {p2}'
    for methods in groups.values():
        r=next(iter(methods.values()));base=[r['group'],r['predictor']+' → '+r['outcome']]
        if r['scope']=='context':buckets[9].append(base+[cell(methods.get(m)) for m in ['MLM','OLS','Pearson']])
        else:
            target=(5 if r['cohort']=='M3' else 6)+(2 if r['scope']=='sensitivity' else 0)
            buckets[target].append(base+[cell(methods.get(m)) for m in ['OLS','Pearson']])
    # Preserve original introduction, task/mediation notes, figure and section.
    children=list(body);first_heading=next(i for i,x in enumerate(children) if ''.join(x.xpath('.//w:t/text()',namespaces=NS)).startswith('Table S1.'))
    intro=children[:first_heading]
    figure_index=next(i for i,x in enumerate(children) if ''.join(x.xpath('.//w:t/text()',namespaces=NS)).startswith('Supplementary Figure S1.'))
    if children[figure_index-1].find('.//w:br',NS) is not None:figure_index-=1
    ending=children[figure_index:]
    def note_for(t):
        index=children.index(t);return copy.deepcopy(children[index-1])
    task_note=note_for(oldtables[1]);med_note=note_for(oldtables[3])
    # Rebuild introductory guidance to avoid obsolete numbering.
    intro=[paragraph('Supplementary Results',True,30),paragraph('Amygdala Circuitry and Daily Affect in Aging: A Replication and Extension',size=25),paragraph('M3 primary adjusted analyses use mixed-effects models (MLM); MR1 uses OLS. Main tables include untransformed primary outcomes and explicitly labeled exploratory moderation. Log-NA robustness, alternative samples/measures, and motion checks appear in sensitivity tables. Statistical indirect effects and their age sensitivity remain together in Table S11.',size=21),paragraph('All coefficients are unstandardized. Confidence intervals are two-sided 95%: normal-Wald for MLM, Student-t for OLS and condition means/differences, Fisher-z for Pearson correlations, and family-level percentile bootstrap for indirect effects. p (analysis) preserves the saved directional or two-sided test; p (2-sided) is also shown. Each method retains its own complete-case N. Comparisons are not independent replications or a basis for selecting the primary method. MR1 OLS estimates are repeated in comparison tables for convenience.',size=21),paragraph('PANAS negative affect uses the natural log without an added constant; diary negative affect retains its separate offset log transformation. Exploratory and sensitivity p values are nominal, without multiplicity adjustment. Unavailable CIs indicate unavailable inference, not nonsignificance. Correlations have no counterpart for interaction coefficients or motion-adjusted regressions.',size=21),paragraph('Contents: S1–S2 main results; S3–S4 sensitivity results; S5–S6 main-method comparisons; S7–S8 sensitivity-method comparisons; S9 contextual associations; S10 task-condition results; S11 statistical indirect effects. Supplementary Figure S1 follows the tables.',size=21)]
    body.clear()
    for n in intro:body.append(n)
    notes={3:'The negative-condition anterior FC sensitivity required a targeted increase to 2,000 optimizer iterations with unchanged model/sample. The persistence × reappraisal raw-PANAS-NA MLM has invalid fitted covariance: affected inference is unavailable despite alternative-optimizer checks. Coefficients are retained for transparency.',7:'OLS results cannot resolve unavailable inference in an MLM. Motion-adjusted models and moderation have no zero-order correlation counterpart.',1:'Moderation rows are exploratory. Main versus sensitivity organization does not change preregistration status.',2:'MR1 is a targeted replication; moderation and mediation were not fitted.',4:'MR1 sensitivity analyses retain OLS as the primary adjusted method.'}
    for i in range(1,12):
        br=E.Element('{'+W+'}p');element(element(br,'r'),'br',type='page');body.append(br)
        body.append(paragraph(f'Table S{i}. {TITLES[i]}',True,26))
        if i in notes:body.append(paragraph(notes[i],size=20))
        if i<=4:body.append(table(['Analysis / specification','Predictor / interaction','Outcome','N','β','95% CI','p (analysis)','p (2-sided)'],buckets[i],[2450,2450,1300,450,800,2000,950,1000]))
        elif i<=8:body.append(table(['Analysis / specification','Association / interaction','OLS','Pearson correlation'],buckets[i],[2450,3100,3650,3650]))
        elif i==9:body.append(table(['Analysis','Association','MLM','OLS','Pearson correlation'],buckets[i],[1800,2500,2850,2850,2850]))
        else:
            body.append(task_note if i==10 else med_note);body.append(copy.deepcopy(oldtables[1 if i==10 else 3]))
    for n in ending:
        for text_node in n.findall(".//w:t",NS):
            if text_node.text:text_node.text=text_node.text.replace("all interaction estimates are reported in Supplementary Table S3.","all interaction estimates are reported in Supplementary Tables S1 and S3.")
        body.append(n)
    assert len(body.findall('w:tbl',NS))==11
    outroot=root/'workspace_snapshots/author_review'/('supplement_reorganization_'+datetime.now().strftime('%Y%m%d_%H%M%S'));outroot.mkdir(parents=True)
    shutil.copy2(doc,outroot/doc.name)
    with ZipFile(doc,'w') as z:
        for info,b in entries:z.writestr(info,E.tostring(tree,xml_declaration=True,encoding='UTF-8',standalone=True) if info.filename=='word/document.xml' else b)
    with ZipFile(doc) as z:
        assert z.testzip() is None
        for info,b in entries:
            if info.filename!='word/document.xml':assert z.read(info.filename)==b
    report=dict(table_rows={f'S{i}':len(buckets[i]) for i in range(1,10)},source_records=len(records),all_prior_numeric_rows_preserved=True,sources=sources)
    report['table_rows'].update(S10=40,S11=26)
    out=root/'results/tables/supplementary_method_comparisons/reorganization.json';out.write_text(json.dumps(report,indent=2)+'\n')
    (outroot/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Backup:',outroot);print('Reorganized rows:',report['table_rows']);print('All prior numerical rows and figure preserved.')
if __name__=='__main__':main()

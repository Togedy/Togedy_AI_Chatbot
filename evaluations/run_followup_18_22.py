"""Live multi-turn API audit for questions 18 and 22, with explicit test replies."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main_cli():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():
        ap.error('Existing runs are never overwritten')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    import main
    import generate_answers as answers
    from settings import Settings
    config=Settings.from_env()
    if not config.openai_api_key or answers.OpenAI is None:
        ap.error('Live OpenAI configuration is required')
    originals=json.loads((ROOT/'evaluations/expected_questions_source_20261102.json').read_text(encoding='utf-8'))
    questions={i['id']:i['question'] for i in originals['items']}
    cases=[
        {'id':'18_stepwise','question_id':18,'inputs':[questions[18], '수시요', '공과대학자유전공학부요', '학생부교과 KU지역균형 전형이요']},
        {'id':'18_unfilled','question_id':18,'inputs':[questions[18], '잘 모르겠어요']},
        {'id':'22_stepwise','question_id':22,'inputs':[questions[22], '건국대요', '현대미술학과요', '학생부종합 KU자기추천 전형에서 기타과목이 얼마나 중요한지 궁금해요.']},
        {'id':'22_complete_reply','question_id':22,'inputs':[questions[22], '건국대 현대미술학과 학생부종합 KU자기추천이요']},
    ]
    report={'started_at':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
            'mode':'real NER and live OpenAI; unmodified production API; request uses previous user input and previous NER only',
            'test_followups':'The follow-up replies are explicit evaluation scenarios, not actual user preferences.',
            'models':{'openai':config.openai_model,'gemini_configured':config.gemini_model},'cases':[]}
    p=main.get_pipeline()
    p.type_ex._predict('수시')
    p.kw_ex.ke._model_predict('제출 서류')
    calls, details=[],{}
    original_call=answers.call_llm
    original_single=answers.run_single_turn
    original_followup=answers.run_followup_turn

    def traced_call(*a,**k):
        start=time.perf_counter()
        role='semantic_review' if any(f.function=='review_document_answer' for f in inspect.stack()) else 'generation'
        try:
            text=original_call(*a,**k)
        except Exception as exc:
            calls.append({'role':role,'error_type':type(exc).__name__,'seconds':round(time.perf_counter()-start,3)})
            raise RuntimeError('Live LLM request failed: '+type(exc).__name__) from None
        if '현재 LLM 호출' in text:
            raise RuntimeError('Missing-LLM fallback cannot be evaluated as a live response')
        calls.append({'role':role,'result':text,'seconds':round(time.perf_counter()-start,3)})
        return text

    def trace_turn(fn):
        def wrapped(*a,**k):
            result=fn(*a,**k)
            details.update(decision=result[1],rows=result[3],stats=result[4])
            return result
        return wrapped

    answers.call_llm=traced_call
    answers.run_single_turn=trace_turn(original_single)
    answers.run_followup_turn=trace_turn(original_followup)
    app=main.create_app()
    app.logger.disabled=True
    client=app.test_client()
    for case in cases:
        record={**case,'turns':[]}
        report['cases'].append(record)
        ner={}
        for index,text in enumerate(case['inputs']):
            calls.clear();details.clear()
            body={'first':index==0,'question_1':case['inputs'][index-1] if index else text}
            if index:
                body.update(question_2=text,NER=ner)
            start=time.perf_counter()
            response=client.post('/answer',json=body)
            payload=response.get_json(silent=True)
            turn={'turn':index+1,'request':body,'http_status':response.status_code,
                  'api_response':payload,**details,'llm_calls':list(calls),
                  'seconds':round(time.perf_counter()-start,3)}
            record['turns'].append(turn)
            args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({'case':case['id'],'turn':index+1,'http':response.status_code,
                              'reply':(payload or {}).get('reply'),'external_calls':len(calls)},ensure_ascii=False),flush=True)
            if payload is None:
                break
            ner=payload.get('NER',{})
    report['completed_at']=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved',args.output,flush=True)


if __name__=='__main__':
    main_cli()

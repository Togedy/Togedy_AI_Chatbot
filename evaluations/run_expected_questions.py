"""Audit PDF questions against the current API; preserve missing-LLM limits."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-live', action='store_true',
                        help='Require a successful real OpenAI preflight; never accept missing-key fallback as live evaluation')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output file; never overwrite prior evaluations.')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import main
    import generate_answers as answers
    from settings import Settings
    config = Settings.from_env()
    preflight = None
    if args.require_live:
        if not config.openai_api_key.strip():
            parser.error('Actual LLM evaluation requires OPENAI_API_KEY in the project .env or environment. No evaluation was run.')
        if answers.OpenAI is None:
            parser.error('Actual LLM evaluation requires the OpenAI SDK. No evaluation was run.')
        try:
            started = time.perf_counter()
            text = answers.call_llm(user_prompt='Return only the JSON object {"status":"ok"}.',
                                    model=config.openai_model, temperature=0,
                                    system_prompt='You are performing a connection check.')
            if not text.strip() or '현재 LLM 호출' in text:
                raise RuntimeError('No actual model response received')
            preflight = {'success': True, 'response': text,
                         'seconds': round(time.perf_counter()-started, 3)}
        except Exception as exc:
            # Do not log raw API exception strings, headers or credentials.
            parser.error(f'Live OpenAI preflight failed ({type(exc).__name__}). No question evaluation was run.')
    source = json.loads(args.input.read_text(encoding='utf-8'))
    report = {
        'started_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'mode': 'real local NER; independent first-turn production API; no saved NER or answer reuse',
        'require_live': args.require_live, 'openai_preflight': preflight,
        'source': source['source'], 'source_sha256': source['sha256'],
        'environment': {
            'google_key_configured': bool(config.google_api_key),
            'openai_key_configured': bool(config.openai_api_key),
            'openai_sdk_installed': bool(importlib.util.find_spec('openai')),
            'openai_model': config.openai_model, 'gemini_model': config.gemini_model,
        },
        'items': [],
    }
    pipeline = main.get_pipeline()
    if not (pipeline.uni_ex.model is not None and callable(pipeline.type_ex._predict)
            and pipeline.kw_ex.ke is not None and callable(pipeline.kw_ex.ke._model_predict)):
        raise RuntimeError('Local NER adapters are incomplete')
    pipeline.type_ex._predict('정시')
    pipeline.kw_ex.ke._model_predict('제출 서류')
    original_llm = answers.call_llm
    original_single = answers.run_single_turn
    calls, details = [], {}

    def traced_llm(*a, **k):
        started = time.perf_counter()
        text = original_llm(*a, **k)
        role = 'semantic_review' if any(f.function == 'review_document_answer' for f in inspect.stack()) else 'generation'
        if args.require_live and '현재 LLM 호출' in text:
            raise RuntimeError('LLM environment fallback is not a live model answer')
        calls.append({'role': role, 'result': text, 'seconds': round(time.perf_counter()-started, 3)})
        return text

    def traced_single(*a, **k):
        result = original_single(*a, **k)
        details.update(decision=result[1], rows=result[3], stats=result[4])
        return result

    answers.call_llm = traced_llm
    answers.run_single_turn = traced_single
    client = main.create_app().test_client()
    for question in source['items']:
        calls.clear()
        details.clear()
        start = time.perf_counter()
        response = client.post('/answer', json={'first': True, 'question_1': question['question']})
        payload = response.get_json(silent=True)
        item = {
            'id': question['id'], 'question': question['question'],
            'http_status': response.status_code, 'api_response': payload,
            **details, 'llm_attempts': list(calls),
            'llm_environment_limited': bool(calls) and not (
                config.openai_api_key and importlib.util.find_spec('openai')),
            'elapsed_seconds': round(time.perf_counter()-start, 3),
        }
        report['items'].append(item)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'id':item['id'], 'http':item['http_status'], 'decision':item.get('decision'),
                          'llm_environment_limited':item['llm_environment_limited'],
                          'seconds':item['elapsed_seconds']},ensure_ascii=False), flush=True)
    report['completed_at'] = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved', len(report['items']), 'new API results:', args.output, flush=True)


if __name__ == '__main__':
    main_cli()

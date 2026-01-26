# server.py
import os
import json
import re
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# 환경 설정
# ==========================================================
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# JSON 응답용 모델 (Gemini 2.0 Flash - 더 안정적인 JSON 생성)
json_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    model_kwargs={"response_mime_type": "application/json"}
)

# 텍스트 응답용 모델 (Gemini 2.0 Flash)
text_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY
)

# ==========================================================
# AgentState 정의
# ==========================================================
def merge_crispe_info(x: Dict[str, str], y: Dict[str, str]) -> Dict[str, str]:
    """CRISPE 정보 병합 - 새 정보가 있으면 업데이트"""
    if not y:
        return x or {}
    if not x:
        return y or {}
    result = dict(x)
    for key, value in y.items():
        if value:  # 빈 문자열이 아닌 경우만 업데이트
            result[key] = value
    return result

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    next_node: str
    response_type: str
    memory_summary: str
    missing_elements: List[str]  # CRISPE 부족 요소들
    crispe_info: Annotated[Dict[str, str], merge_crispe_info]  # CRISPE 분석에서 수집된 정보 (병합 함수 적용)
    clarification_count: int  # 선제적 질문 횟수 추적

CONTEXT_BUDGET_TOKENS = 8000
SUMMARIZE_AT_TOKENS = 12000
SUMMARY_MAX_TOKENS = 600
MAX_CLARIFICATION_ROUNDS = 3  # 최대 선제적 질문 횟수

# ==========================================================
# 헬퍼 함수들
# ==========================================================
def clean_non_korean(text: str) -> str:
    """
    한글, 영문, 숫자, 기본 문장부호만 허용하고
    중국어, 일본어, 태국어 등 제거
    """
    cleaned = re.sub(r'[\u4e00-\u9fff]', '', text)  # 한자
    cleaned = re.sub(r'[\u3040-\u309f]', '', cleaned)  # 히라가나
    cleaned = re.sub(r'[\u30a0-\u30ff]', '', cleaned)  # 가타카나
    cleaned = re.sub(r'[\u0e00-\u0e7f]', '', cleaned)  # 태국어

    # 빈 공백 정리
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned

def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)

def _build_serialized_history(messages: List[BaseMessage]) -> str:
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            if not getattr(m, "tool_calls", []):
                lines.append(f"Assistant: {m.content}")
    return "\n".join(lines)

def _get_conversation_history_windowed(state: AgentState, budget_tokens: int = CONTEXT_BUDGET_TOKENS) -> str:
    recent_blocks: List[str] = []
    used = 0
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            s = f"User: {m.content}"
        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", []):
            s = f"Assistant: {m.content}"
        else:
            continue
        t = _approx_tokens(s)
        if used + t > budget_tokens:
            break
        recent_blocks.append(s)
        used += t

    recent_part = "\n".join(reversed(recent_blocks)).strip()
    summary_part = (state.get("memory_summary") or "").strip()
    if summary_part and recent_part:
        return f"[SUMMARY]\n{summary_part}\n\n[RECENT]\n{recent_part}"
    elif summary_part:
        return f"[SUMMARY]\n{summary_part}"
    else:
        return recent_part

def _maybe_update_memory_summary(state: AgentState) -> Optional[str]:
    full = _build_serialized_history(state["messages"])
    if _approx_tokens(full) < SUMMARIZE_AT_TOKENS:
        return None
    midpoint = max(1, len(full) * 6 // 10)
    old_chunk = full[:midpoint]
    prompt = f"""
당신은 대화 러닝 서머리 요약기입니다.
아래 '오래된 대화'를 읽고, 앞으로의 작업에 필요한 핵심만 한국어로 간결하게 요약하세요.
- {SUMMARY_MAX_TOKENS} 토큰 이내 목표

[오래된 대화]
{old_chunk}
"""
    try:
        summary = text_model.invoke(prompt).content.strip()
    except Exception:
        summary = ""
    prev = (state.get("memory_summary") or "").strip()
    new_summary = (prev + "\n" + summary).strip() if summary else prev or ""
    return new_summary or None

def _get_conversation_history(state: AgentState) -> str:
    history = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            history.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            if not getattr(msg, 'tool_calls', []):
                history.append(f"Assistant: {msg.content}")
    return "\n".join(history)

def _get_last_user_message(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

def _extract_crispe_from_conversation(state: AgentState) -> Dict[str, str]:
    """
    대화 내역에서 CRISPE 정보를 추출합니다.
    사용자가 명시적으로 답변한 정보 + state에 저장된 정보를 통합합니다.
    """
    # state에 이미 저장된 crispe_info가 있으면 시작점으로 사용
    crispe_info = dict(state.get("crispe_info", {}) or {})

    conversation_history = _get_conversation_history_windowed(state)

    system_prompt = """
## Role
대화 내역에서 사용자가 제공한 CRISPE 정보를 추출하는 분석기입니다.

## Task
아래 대화에서 사용자가 명시적으로 언급한 정보를 추출하세요.

## 추출할 정보
1. context: 사용자의 상황/배경 (예: 초보자, 혼자 여행, 가족과 함께, 학생, 직장인)
2. intent: 목적/용도 (예: 문화체험, 업무용, 학습용, 취미)
3. scope: 범위/제약조건 (예: 예산 200만원, 기간 1주일, 지역 교토, Python 사용)
4. preference: 선호도/우선순위 (예: 가성비 중시, 조용한 곳 선호, 간단한 것 선호)

## 규칙
- 사용자가 명시적으로 말한 것만 추출
- 추론하지 말고 언급된 것만 기록
- 없으면 빈 문자열 ""
- "아무거나", "상관없어요", "잘 모르겠어요" 등은 빈 문자열로 처리

## 출력 형식 (JSON만)
{
    "context": "추출된 맥락",
    "intent": "추출된 목적",
    "scope": "추출된 범위/조건",
    "preference": "추출된 선호도"
}
"""
    try:
        response = json_model.invoke(system_prompt + f"\n\n[대화 내역]\n{conversation_history}")
        extracted = json.loads(response.content)

        # 기존 정보와 병합 (새로 추출된 정보가 우선)
        for key in ["context", "intent", "scope", "preference"]:
            new_val = extracted.get(key, "")
            if new_val:  # 새 값이 있으면 업데이트
                crispe_info[key] = new_val
            elif key not in crispe_info:  # 키가 없으면 빈 문자열로 초기화
                crispe_info[key] = ""

        return crispe_info
    except Exception as e:
        print(f"Error extracting CRISPE info: {e}")
        return crispe_info

# ==========================================================
# LLM 판단 함수들
# ==========================================================
def is_ambiguous_with_llm(query: str, existing_crispe: Dict[str, str] = None) -> dict:
    """
    CRISPE 프레임워크 기반 모호성 판단 (모든 도메인 지원)
    Returns: {"is_ambiguous": bool, "missing_elements": [...], "analysis": {...}, "provided_info": {...}, "detected_domain": str}
    """
    existing_info = ""
    if existing_crispe:
        existing_parts = []
        for k, v in existing_crispe.items():
            if v:
                existing_parts.append(f"- {k}: {v}")
        if existing_parts:
            existing_info = "\n## 이미 수집된 정보:\n" + "\n".join(existing_parts)

    system_prompt = f"""
## Role
사용자 요청의 완성도를 CRISPE 기준으로 **엄격하게** 판단합니다.
**어떤 도메인의 요청이든** 답변을 제공하기 위해 필요한 정보가 충분한지 평가합니다.

## 핵심 원칙
1. **모든 종류의 요청**에 대해 모호성을 판단합니다 (여행, 추천, 학습, 코딩, 글쓰기, 일반 질문 등)
2. 답변의 방향성을 결정할 수 있는 충분한 정보가 없으면 is_ambiguous: true
3. 단순 사실 질문(예: "한국의 수도는?")은 is_ambiguous: false

## CRISPE 분석 방법

각 요소를 아래 질문으로 판단:

| 요소 | 판단 질문 | not_needed 조건 |
|------|----------|-----------------|
| Context | "누가/어떤 상황인지에 따라 답이 달라지는가?" | 아니오 → not_needed |
| Request | "원하는 결과물이 명시되어 있는가?" | 예 → provided (항상 필요) |
| Intent | "목적에 따라 답이 달라지는가?" | 아니오 → not_needed |
| Scope | "범위를 좁혀야 답할 수 있는가?" | 아니오 → not_needed |
| Preference | "개인 선호에 따라 답이 달라지는가?" | 아니오 → not_needed |

## 상태 판단 (엄격한 기준)
- **provided**: 사용자가 **구체적이고 명확하게** 명시함
  - 숫자, 날짜, 지명, 금액, 기술명 등 명확한 정보가 있어야 함
  - 예: "2명", "200만원", "교토", "5일", "Python", "React", "업무용"
- **missing**: 필요한데 없거나, 모호하거나 불명확한 답변
- **not_needed**: 이 질문에는 해당 정보가 필요 없음

## ⚠️ 중요: 아래는 모두 missing으로 처리 (provided 아님!)
1. **회피/불확실 표현**: "글쎄요", "잘 모르겠어요", "생각 중", "아직 안 정했어요"
2. **무관심 표현**: "아무거나", "상관없어요", "괜찮아요", "다 좋아요", "뭐든"
3. **애매한 표현**: "적당히", "보통", "그냥", "대충", "어느 정도"
4. **불명확한 범위**: "여유 있게", "넉넉하게", "충분히" (구체적 수치 없음)
5. **미결정 표현**: "나중에", "알아보고", "고민 중"
6. **너무 넓은 범위**: "일본" (도시 미지정), "코딩" (언어 미지정), "공부" (주제 미지정)
{existing_info}

## 도메인별 필수 정보 체크리스트

### 여행 요청 시:
- 목적지 (구체적 도시/지역)
- 기간 (숫자로 된 일수 또는 날짜)
- 인원 (숫자)
- 예산 (금액 범위)
→ 위 4개 중 **2개 이상** missing이면 is_ambiguous: true

### 제품/서비스 추천 시:
- 용도/목적 (구체적)
- 예산 (금액 범위)
- 주요 요구사항 1개 이상
→ 위 3개 중 **1개 이상** missing이면 is_ambiguous: true

### 학습/공부 요청 시:
- 학습 주제/과목 (구체적)
- 현재 수준 (초급/중급/고급 또는 구체적 설명)
- 학습 목표
→ 위 3개 중 **1개 이상** missing이면 is_ambiguous: true

### 코딩/개발 요청 시:
- 사용 언어/프레임워크
- 구현하고 싶은 기능 (구체적)
- 제약조건 (있다면)
→ Request(구현 기능)가 불명확하면 is_ambiguous: true

### 글쓰기 요청 시:
- 글의 목적/용도
- 대상 독자
- 원하는 톤/스타일 또는 길이
→ 위 3개 중 **1개 이상** missing이면 is_ambiguous: true

### 음식/맛집 추천 시:
- 지역/위치
- 음식 종류 또는 선호도
- 예산 또는 분위기
→ 위 3개 중 **1개 이상** missing이면 is_ambiguous: true

### 건강/운동 관련 시:
- 현재 상태/수준
- 목표 (체중감량, 근력증가 등)
- 제약조건 (시간, 장비, 부상 등)
→ 위 3개 중 **1개 이상** missing이면 is_ambiguous: true

### 일반 조언/도움 요청 시:
- 구체적인 상황 설명
- 원하는 결과/목표
→ 상황이 불명확하면 is_ambiguous: true

### 단순 사실 질문 (예: "한국의 수도는?", "파이썬이 뭐야?"):
→ 추가 정보 불필요, is_ambiguous: false

## 판단 규칙 (엄격)
1. 먼저 요청의 도메인을 파악합니다
2. 해당 도메인의 필수 정보를 체크합니다
3. missing이 기준 이상이면 → is_ambiguous: true
4. 모든 필수 요소가 **구체적으로** provided 또는 not_needed → is_ambiguous: false

## 출력 형식 (JSON만)
{{
    "detected_domain": "여행|제품추천|학습|코딩|글쓰기|음식|건강|일반조언|단순질문|기타",
    "analysis": {{
        "context": {{"status": "provided|missing|not_needed", "reason": "...", "value": "구체적 값 또는 null"}},
        "request": {{"status": "provided|missing|not_needed", "reason": "...", "value": "구체적 값 또는 null"}},
        "intent": {{"status": "provided|missing|not_needed", "reason": "...", "value": "구체적 값 또는 null"}},
        "scope": {{"status": "provided|missing|not_needed", "reason": "...", "value": "구체적 값 또는 null"}},
        "preference": {{"status": "provided|missing|not_needed", "reason": "...", "value": "구체적 값 또는 null"}}
    }},
    "is_ambiguous": boolean,
    "missing_elements": ["missing 상태인 요소들"],
    "provided_info": {{
        "context": "구체적으로 추출된 정보 또는 null",
        "intent": "구체적으로 추출된 정보 또는 null",
        "scope": "구체적으로 추출된 정보 또는 null",
        "preference": "구체적으로 추출된 정보 또는 null"
    }}
}}
"""
    try:
        response = json_model.invoke(system_prompt + f"\n\n[사용자 요청]\n{query}")
        result = json.loads(response.content)
        is_ambiguous = result.get("is_ambiguous", True)
        missing = result.get("missing_elements", [])
        provided_info = result.get("provided_info", {})
        detected_domain = result.get("detected_domain", "기타")
        print(f"CRISPE 분석: domain={detected_domain}, is_ambiguous={is_ambiguous}, missing={missing}, provided={provided_info}")
        return result
    except Exception as e:
        print(f"Error in is_ambiguous_with_llm: {e}")
        return {
            "is_ambiguous": True,
            "missing_elements": ["context", "request", "intent", "scope", "preference"],
            "provided_info": {},
            "detected_domain": "기타"
        }

def classify_intent_with_llm(query: str) -> str:
    system_prompt = """
    Classify the user request into 'search' or 'report'.
    - 'search': find specific information
    - 'report': create structured document
    Respond ONLY with 'search' or 'report'.
    """
    try:
        response = text_model.invoke(system_prompt + f"\n\nUser Request:\n\"{query}\"")
        result = response.content.strip().lower()
        if result in ['search', 'report']:
            return result
        return 'search'
    except Exception as e:
        print(f"Error in classify_intent_with_llm: {e}")
        return 'search'

def needs_research_with_llm(query: str, crispe_info: Dict[str, str] = None) -> dict:
    """
    LLM이 웹 검색이 필요한지 스스로 판단합니다.
    Returns: {"needs_research": bool, "reason": str}
    """
    crispe_context = ""
    if crispe_info:
        crispe_parts = [f"- {k}: {v}" for k, v in crispe_info.items() if v]
        if crispe_parts:
            crispe_context = "\n## 사용자가 제공한 추가 정보:\n" + "\n".join(crispe_parts)

    system_prompt = f"""## Role
사용자의 요청을 처리하기 위해 웹 검색이 필요한지 판단하는 분석기입니다.

## 판단 기준

### 웹 검색이 필요한 경우 (needs_research: true):
- 최신 정보가 필요한 경우 (뉴스, 트렌드, 현재 가격, 날씨 등)
- 특정 장소/업체 정보 (맛집, 관광지, 호텔, 병원 등)
- 사실 확인이 필요한 정보 (통계, 역사적 사실, 법률 등)
- 제품/서비스 비교 및 추천 (실제 제품 정보 필요)
- 특정 지역/국가 관련 정보 (여행, 문화, 규정 등)
- 전문 지식이나 연구 결과가 필요한 경우

### 웹 검색이 필요 없는 경우 (needs_research: false):
- 글쓰기/작성 요청 (자소서, 이메일, 편지, 에세이, 블로그 글 등)
- 코드 작성/프로그래밍 도움
- 번역 요청
- 문법/맞춤법 교정
- 창작 활동 (시, 소설, 스토리 등)
- 일반적인 조언/상담 (연애, 인간관계, 고민 상담 등)
- 브레인스토밍/아이디어 생성
- 설명/교육 요청 (개념 설명, 학습 도움)
- 요약/정리 요청
- 계산/분석 요청 (데이터가 주어진 경우)
- 대화/잡담
{crispe_context}

## 출력 형식 (JSON만)
{{
    "needs_research": boolean,
    "reason": "판단 이유 (한국어로 간단히)"
}}
"""
    try:
        response = json_model.invoke(system_prompt + f"\n\n[사용자 요청]\n{query}")
        result = json.loads(response.content)
        needs_research = result.get("needs_research", True)
        reason = result.get("reason", "")
        print(f"Research 필요 여부 판단: needs_research={needs_research}, reason={reason}")
        return result
    except Exception as e:
        print(f"Error in needs_research_with_llm: {e}")
        # 오류 시 기본적으로 검색 수행 (안전한 선택)
        return {"needs_research": True, "reason": "판단 오류로 기본값 사용"}

def _generate_search_query_with_llm(plan_description: str, conversation_history: str) -> str:
    system_prompt = """
    Convert the plan description into a concise search query.
    Return ONLY the search query as plain string.
    검색 쿼리는 한국어로 작성하세요.
    """
    try:
        response = text_model.invoke(system_prompt + f"\n\nPlan: \"{plan_description}\"\nHistory: \"{conversation_history}\"")
        query = response.content.strip()
        return query if query else plan_description
    except Exception as e:
        print(f"Error in _generate_search_query_with_llm: {e}")
        return plan_description or conversation_history[-100:]

# ==========================================================
# Tool 함수들
# ==========================================================
def generate_clarifying_question_tool(query: str, missing_elements: List[str] = None, detected_domain: str = "기타", max_retries: int = 3) -> List[dict]:
    """
    CRISPE 프레임워크 기반 질문 생성 (모든 도메인 지원)
    missing_elements: ["context", "request", "intent", "scope", "preference"] 중 부족한 요소들
    detected_domain: 감지된 요청 도메인
    """
    # missing_elements가 없으면 모든 요소 질문
    if not missing_elements:
        missing_elements = ["context", "request", "intent", "scope", "preference"]

    missing_str = ", ".join(missing_elements)

    system_prompt = f"""## Role
사용자의 요청을 명확하게 만들기 위한 질문 생성기입니다.

## 최우선 규칙 - 언어 제한 (CRITICAL):
- 모든 출력은 반드시 100% 한국어로만 작성하세요.
- 중국어, 일본어, 영어 단어를 절대 사용하지 마세요.
- 지명/고유명사도 한글로 표기 (도쿄, 교토, 오사카 등)

## 감지된 도메인: {detected_domain}
## 부족한 CRISPE 요소: {missing_str}

## CRISPE 요소별 질문 가이드:
- context: 사용자의 상황/배경 (누구와, 어떤 상황, 현재 수준)
- request: 원하는 결과물 형태
- intent: 목적/용도 (왜 필요한지)
- scope: 범위/제약조건 (예산, 기간, 지역, 기술 스택 등)
- preference: 선호도/우선순위

## 도메인별 질문 예시:

### 여행 요청:
- 목적지, 기간, 예산, 동행자, 관심사, 숙소 선호도

### 제품/서비스 추천:
- 용도/목적, 예산, 주요 선호 조건, 제약 조건

### 학습/공부:
- 학습 주제, 현재 수준, 학습 목표, 선호 학습 방식

### 코딩/개발:
- 사용 언어/프레임워크, 구현 목표, 현재 수준, 제약조건

### 글쓰기:
- 글의 목적, 대상 독자, 원하는 톤/길이, 포함할 내용

### 음식/맛집:
- 지역/위치, 음식 종류, 예산, 분위기, 인원

### 건강/운동:
- 현재 상태, 목표, 가용 시간, 장비/장소 제약

### 일반 조언:
- 구체적 상황, 원하는 결과, 제약조건

## 핵심 원칙:
1. 이미 제공된 정보는 절대 다시 묻지 않음
2. 각 질문에 3-5개의 구체적인 선택지 제공
3. 마지막 선택지는 "직접 입력" 포함
4. 최대 3개 질문만 생성
5. 도메인에 맞는 질문을 생성

## 출력 형식 (반드시 JSON 배열로만 응답):
[
    {{
        "question": "질문 내용",
        "category": "context|request|intent|scope|preference",
        "choices": ["선택지1", "선택지2", "선택지3", "선택지4", "직접 입력"],
        "allowMultiple": false
    }}
]

## 도메인별 예시:

### 여행 ("일본 여행 계획해줘"):
[
    {{"question": "여행 인원이 어떻게 되나요?", "category": "context", "choices": ["혼자", "커플/2인", "가족 (3-4인)", "친구들과 (5인 이상)", "직접 입력"], "allowMultiple": false}},
    {{"question": "예산은 어느 정도로 생각하시나요?", "category": "scope", "choices": ["100만원 이하", "100-200만원", "200-300만원", "300만원 이상", "직접 입력"], "allowMultiple": false}},
    {{"question": "여행 기간은 얼마나 되나요?", "category": "scope", "choices": ["2-3일", "4-5일", "1주일", "2주 이상", "직접 입력"], "allowMultiple": false}}
]

### 코딩 ("웹사이트 만들어줘"):
[
    {{"question": "어떤 종류의 웹사이트인가요?", "category": "request", "choices": ["개인 포트폴리오", "쇼핑몰", "블로그", "회사 소개", "직접 입력"], "allowMultiple": false}},
    {{"question": "선호하는 기술 스택이 있나요?", "category": "scope", "choices": ["React", "Vue", "순수 HTML/CSS/JS", "잘 모르겠어요 (추천해주세요)", "직접 입력"], "allowMultiple": false}},
    {{"question": "현재 코딩 경험은 어느 정도인가요?", "category": "context", "choices": ["처음이에요", "기초는 알아요", "중급", "고급", "직접 입력"], "allowMultiple": false}}
]

### 학습 ("영어 공부하고 싶어"):
[
    {{"question": "현재 영어 수준은 어느 정도인가요?", "category": "context", "choices": ["완전 초급", "기초 회화 가능", "중급 (일상 대화 가능)", "고급 (비즈니스 가능)", "직접 입력"], "allowMultiple": false}},
    {{"question": "영어 학습의 주요 목적은 무엇인가요?", "category": "intent", "choices": ["여행/일상 회화", "취업/이직", "시험 준비 (토익/토플 등)", "업무/비즈니스", "직접 입력"], "allowMultiple": false}},
    {{"question": "선호하는 학습 방식이 있나요?", "category": "preference", "choices": ["앱/온라인 강의", "책/교재", "1:1 과외", "그룹 스터디", "직접 입력"], "allowMultiple": false}}
]

### 글쓰기 ("자기소개서 써줘"):
[
    {{"question": "어떤 용도의 자기소개서인가요?", "category": "intent", "choices": ["취업용", "대학원 입학용", "장학금 신청용", "동아리/단체 가입용", "직접 입력"], "allowMultiple": false}},
    {{"question": "지원하는 분야/직무가 무엇인가요?", "category": "scope", "choices": ["IT/개발", "마케팅/기획", "디자인", "영업/서비스", "직접 입력"], "allowMultiple": false}},
    {{"question": "강조하고 싶은 경험이나 역량이 있나요?", "category": "preference", "choices": ["프로젝트 경험", "리더십 경험", "해외 경험", "자격증/기술", "직접 입력"], "allowMultiple": false}}
]

### 음식/맛집 ("맛집 추천해줘"):
[
    {{"question": "어느 지역에서 찾으시나요?", "category": "scope", "choices": ["서울 강남권", "서울 홍대/마포", "서울 종로/을지로", "다른 지역", "직접 입력"], "allowMultiple": false}},
    {{"question": "어떤 종류의 음식을 원하시나요?", "category": "preference", "choices": ["한식", "일식", "양식", "중식", "직접 입력"], "allowMultiple": false}},
    {{"question": "예산은 어느 정도인가요? (1인 기준)", "category": "scope", "choices": ["1만원 이하", "1-3만원", "3-5만원", "5만원 이상", "직접 입력"], "allowMultiple": false}}
]

### 건강/운동 ("운동 추천해줘"):
[
    {{"question": "운동의 주요 목표가 무엇인가요?", "category": "intent", "choices": ["체중 감량", "근력 증가", "체력 향상", "스트레스 해소", "직접 입력"], "allowMultiple": false}},
    {{"question": "운동에 투자할 수 있는 시간은 어느 정도인가요?", "category": "scope", "choices": ["하루 30분 이하", "하루 30분-1시간", "하루 1시간 이상", "주말에만", "직접 입력"], "allowMultiple": false}},
    {{"question": "운동 경험/현재 체력 수준은 어떤가요?", "category": "context", "choices": ["운동 경험 거의 없음", "가끔 운동함", "규칙적으로 운동 중", "운동을 많이 해왔음", "직접 입력"], "allowMultiple": false}}
]

### 일반 ("도움이 필요해", "추천해줘", "알려줘"):
[
    {{"question": "어떤 종류의 도움이 필요하신가요?", "category": "request", "choices": ["정보/지식 검색", "추천/조언", "문서/글 작성", "문제 해결", "직접 입력"], "allowMultiple": false}},
    {{"question": "좀 더 구체적으로 어떤 분야인가요?", "category": "scope", "choices": ["일/업무 관련", "학습/공부 관련", "일상생활 관련", "취미/여가 관련", "직접 입력"], "allowMultiple": false}}
]
"""

    for attempt in range(max_retries):
        try:
            response = json_model.invoke(system_prompt + f'\n\n[사용자 요청]\n{query}')
            print(f"[DEBUG] LLM 원본 응답: {response.content[:500]}")
            result = json.loads(response.content)

            # LLM이 {"questions": [...]} 또는 {"clarifyingQuestions": [...]} 형태로 반환할 경우 처리
            if isinstance(result, dict):
                if "questions" in result:
                    questions = result["questions"]
                    print(f"[DEBUG] questions 키로 반환됨: {questions}")
                    if questions and isinstance(questions, list) and len(questions) > 0:
                        return questions
                if "clarifyingQuestions" in result:
                    questions = result["clarifyingQuestions"]
                    print(f"[DEBUG] clarifyingQuestions 키로 반환됨: {questions}")
                    if questions and isinstance(questions, list) and len(questions) > 0:
                        return questions

            # 직접 배열로 반환된 경우
            if isinstance(result, list) and len(result) > 0:
                print(f"[DEBUG] 직접 배열로 반환됨: {result}")
                return result

            # 빈 결과인 경우 재시도
            print(f"[DEBUG] 빈 결과, 재시도 {attempt + 1}/{max_retries}")
            continue

        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print("기본 질문으로 폴백합니다.")
                return _get_fallback_questions(query, detected_domain)
        except Exception as e:
            print(f"질문 생성 오류 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return _get_fallback_questions(query, detected_domain)

    # 모든 시도 실패 시
    return _get_fallback_questions(query, detected_domain)


def _get_fallback_questions(query: str, detected_domain: str = "기타") -> List[dict]:
    """도메인에 따른 기본 질문 반환"""
    query_lower = query.lower()

    # 도메인 기반 폴백
    domain_questions = {
        "여행": [
            {"question": "여행 인원이 어떻게 되나요?", "category": "context", "choices": ["혼자", "커플/2인", "가족 (3-4인)", "친구들과", "직접 입력"], "allowMultiple": False},
            {"question": "예산은 어느 정도로 생각하시나요?", "category": "scope", "choices": ["100만원 이하", "100-200만원", "200-300만원", "300만원 이상", "직접 입력"], "allowMultiple": False},
            {"question": "여행 기간은 얼마나 되나요?", "category": "scope", "choices": ["2-3일", "4-5일", "1주일", "2주 이상", "직접 입력"], "allowMultiple": False}
        ],
        "제품추천": [
            {"question": "어떤 용도로 사용하시나요?", "category": "intent", "choices": ["업무/학습용", "취미/여가용", "선물용", "기타", "직접 입력"], "allowMultiple": False},
            {"question": "예산은 어느 정도인가요?", "category": "scope", "choices": ["저렴한 것", "중간 가격대", "고급/프리미엄", "상관없음", "직접 입력"], "allowMultiple": False}
        ],
        "학습": [
            {"question": "현재 수준은 어느 정도인가요?", "category": "context", "choices": ["완전 초급", "기초 있음", "중급", "고급", "직접 입력"], "allowMultiple": False},
            {"question": "학습의 주요 목적은 무엇인가요?", "category": "intent", "choices": ["취업/이직", "자기계발", "시험 준비", "업무 활용", "직접 입력"], "allowMultiple": False},
            {"question": "선호하는 학습 방식이 있나요?", "category": "preference", "choices": ["온라인 강의", "책/교재", "실습 위주", "1:1 과외", "직접 입력"], "allowMultiple": False}
        ],
        "코딩": [
            {"question": "어떤 것을 만들고 싶으신가요?", "category": "request", "choices": ["웹사이트", "앱", "자동화 스크립트", "데이터 분석", "직접 입력"], "allowMultiple": False},
            {"question": "선호하는 언어/기술이 있나요?", "category": "scope", "choices": ["Python", "JavaScript", "Java", "잘 모르겠어요", "직접 입력"], "allowMultiple": False},
            {"question": "현재 코딩 경험은 어느 정도인가요?", "category": "context", "choices": ["처음이에요", "기초는 알아요", "중급", "고급", "직접 입력"], "allowMultiple": False}
        ],
        "글쓰기": [
            {"question": "어떤 용도의 글인가요?", "category": "intent", "choices": ["업무/공식 문서", "학교 과제", "개인 블로그/SNS", "지원서/자기소개", "직접 입력"], "allowMultiple": False},
            {"question": "대상 독자는 누구인가요?", "category": "context", "choices": ["회사/상사", "교수/선생님", "일반 대중", "친구/지인", "직접 입력"], "allowMultiple": False},
            {"question": "원하는 톤/스타일은?", "category": "preference", "choices": ["격식체/공식적", "친근한/캐주얼", "전문적/학술적", "상관없음", "직접 입력"], "allowMultiple": False}
        ],
        "음식": [
            {"question": "어느 지역에서 찾으시나요?", "category": "scope", "choices": ["서울", "경기/인천", "부산", "기타 지역", "직접 입력"], "allowMultiple": False},
            {"question": "어떤 종류의 음식을 원하시나요?", "category": "preference", "choices": ["한식", "일식", "양식", "중식", "직접 입력"], "allowMultiple": False},
            {"question": "예산은 어느 정도인가요? (1인 기준)", "category": "scope", "choices": ["1만원 이하", "1-3만원", "3-5만원", "5만원 이상", "직접 입력"], "allowMultiple": False}
        ],
        "건강": [
            {"question": "주요 목표가 무엇인가요?", "category": "intent", "choices": ["체중 감량", "근력 증가", "체력 향상", "스트레스 해소", "직접 입력"], "allowMultiple": False},
            {"question": "운동에 투자할 수 있는 시간은?", "category": "scope", "choices": ["하루 30분 이하", "하루 30분-1시간", "하루 1시간 이상", "주말에만", "직접 입력"], "allowMultiple": False},
            {"question": "현재 운동 경험/체력 수준은?", "category": "context", "choices": ["운동 경험 거의 없음", "가끔 운동함", "규칙적으로 운동 중", "많이 해왔음", "직접 입력"], "allowMultiple": False}
        ],
        "일반조언": [
            {"question": "어떤 종류의 도움이 필요하신가요?", "category": "request", "choices": ["정보/지식 검색", "추천/조언", "문서/글 작성", "문제 해결", "직접 입력"], "allowMultiple": False},
            {"question": "어떤 분야와 관련된 건가요?", "category": "scope", "choices": ["일/업무", "학습/공부", "일상생활", "취미/여가", "직접 입력"], "allowMultiple": False}
        ]
    }

    # 도메인 매칭
    if detected_domain in domain_questions:
        return domain_questions[detected_domain]

    # 키워드 기반 폴백 (도메인 감지 실패 시)
    if any(kw in query_lower for kw in ["여행", "travel", "여행지", "관광", "trip"]):
        return domain_questions["여행"]
    if any(kw in query_lower for kw in ["추천", "recommend", "뭐가 좋", "골라"]):
        return domain_questions["제품추천"]
    if any(kw in query_lower for kw in ["공부", "학습", "배우", "study", "learn"]):
        return domain_questions["학습"]
    if any(kw in query_lower for kw in ["코딩", "프로그래밍", "개발", "코드", "만들어"]):
        return domain_questions["코딩"]
    if any(kw in query_lower for kw in ["써줘", "작성", "글", "write", "문서"]):
        return domain_questions["글쓰기"]
    if any(kw in query_lower for kw in ["맛집", "음식", "먹", "식당", "레스토랑"]):
        return domain_questions["음식"]
    if any(kw in query_lower for kw in ["운동", "헬스", "다이어트", "건강", "살"]):
        return domain_questions["건강"]

    # 최종 기본 폴백
    return [
        {"question": "어떤 종류의 도움이 필요하신가요?", "category": "request", "choices": ["정보 검색", "추천/조언", "글/문서 작성", "문제 해결", "직접 입력"], "allowMultiple": False},
        {"question": "좀 더 구체적으로 알려주시겠어요?", "category": "scope", "choices": ["일/업무 관련", "학습/공부 관련", "일상생활 관련", "취미/여가 관련", "직접 입력"], "allowMultiple": False}
    ]


def create_plan_tool(query: str) -> List[Dict[str, Any]]:
    system_prompt = """
    Create a research plan as JSON array.
    Each step: {"id": 1, "title": "...", "description": "...", "status": "pending", "research": true, "dependencies": [], "complexity": 3}
    Return ONLY JSON array.
    중요: title과 description은 반드시 한국어로 작성하세요.
    """
    response = json_model.invoke(system_prompt + f"\n\nUser Request: \"{query}\"")
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return []

def web_search_tool(query: str) -> List[Dict[str, Any]]:
    try:
        # 한국어 결과 우선을 위해 검색어에 "한국어" 추가
        korean_query = f"{query} 한국어"

        search_results = tavily_client.search(
            korean_query,
            search_depth="advanced",
            max_results=5,
            include_domains=[
                "naver.com", "tistory.com", "velog.io",
                "brunch.co.kr", "korea.kr", "donga.com",
                "chosun.com", "mk.co.kr", "hani.co.kr",
                "yna.co.kr", "khan.co.kr", "joongang.co.kr"
            ],  # 한국 사이트만 검색
        )
        return search_results.get('results', []) or []
    except Exception:
        return []

def reflect_and_critique_tool(conversation: str) -> str:
    system_prompt = """
    Review the conversation and critique the draft. Suggest improvements.
    반드시 한국어로만 답변하세요. 중국어, 일본어, 영어를 섞지 마세요.
    """
    response = text_model.invoke(system_prompt + f"\n\nConversation:\n\"{conversation}\"")
    return response.content

# ==========================================================
# 노드들
# ==========================================================
def qa_node(state: AgentState):
    # 전체 대화 히스토리를 전달하여 이미 제공된 정보를 파악할 수 있게 함
    conversation_history = _get_conversation_history_windowed(state)

    # CRISPE 분석 결과에서 missing_elements와 detected_domain 가져오기
    missing_elements = state.get("missing_elements", None)
    detected_domain = state.get("detected_domain", "기타")

    questions = generate_clarifying_question_tool(conversation_history, missing_elements, detected_domain)

    # clarification_count 증가
    current_count = state.get("clarification_count", 0)

    return {
        "messages": [
            ToolMessage(
                name="generate_clarifying_question_tool",
                content=json.dumps(questions, ensure_ascii=False),
                tool_call_id="manual_qa"
            )
        ],
        "clarification_count": current_count + 1
    }

def planner_node(state: AgentState):
    last_user = _get_last_user_message(state)
    plan = create_plan_tool(last_user)
    cleaned = []
    for step in (plan or []):
        if isinstance(step, dict):
            step = {**step}
            step["research"] = bool(step.get("research", True))
            cleaned.append(step)
    return {
        "messages": [
            ToolMessage(
                name="create_plan_tool",
                content=json.dumps(cleaned, ensure_ascii=False),
                tool_call_id="manual_planner"
            )
        ]
    }

def researcher_node(state: AgentState):
    plan = []
    for m in reversed(state["messages"]):
        if isinstance(m, ToolMessage) and m.name == "create_plan_tool":
            try:
                plan = json.loads(m.content) or []
            except json.JSONDecodeError:
                plan = []
            break

    conversation_history = _get_conversation_history_windowed(state)
    all_search_results = []

    print(f"Researcher: Starting research with plan: {len(plan) if plan else 0} steps")

    # 사용자의 마지막 메시지로 검색
    last_user_msg = _get_last_user_message(state)
    if last_user_msg:
        print(f"Researcher: Searching for: {last_user_msg}")
        try:
            results = web_search_tool(last_user_msg)
            if results:
                all_search_results.extend(results)
                print(f"Researcher: Found {len(results)} results")
        except Exception as e:
            print(f"Researcher: Error in web search: {e}")

    # 만약 직접 검색으로 결과를 못 찾았다면, plan을 기반으로 검색
    if not all_search_results and plan:
        for step in plan:
            if step.get("research"):
                try:
                    optimized_query = _generate_search_query_with_llm(step.get("description", ""), conversation_history)
                    results_list = web_search_tool(optimized_query)
                    if results_list:
                        all_search_results.extend(results_list)
                except Exception as e:
                    print(f"Researcher: Error processing plan step: {e}")

    return {
        "messages": [
            ToolMessage(
                content=json.dumps(all_search_results, ensure_ascii=False),
                name="web_search_tool",
                tool_call_id="manual_researcher"
            )
        ]
    }

def generator_node(state: AgentState):
    conversation_history = _get_conversation_history_windowed(state)
    research_results = []
    plan = []
    has_search_tool_message = False  # researcher를 거쳤는지 확인

    for m in reversed(state["messages"]):
        if isinstance(m, ToolMessage) and m.name == "web_search_tool":
            has_search_tool_message = True
            try:
                raw_results = json.loads(m.content)
                if isinstance(raw_results, list):
                    research_results = raw_results
            except:
                pass
            break

    for m in reversed(state["messages"]):
        if isinstance(m, ToolMessage) and m.name == "create_plan_tool":
            try:
                raw_plan = json.loads(m.content)
                if isinstance(raw_plan, list):
                    plan = raw_plan
            except:
                pass
            break

    # CRISPE 정보 추출 (대화에서 수집된 정보)
    crispe_info = _extract_crispe_from_conversation(state)
    context_info = crispe_info.get("context", "정보 없음")
    intent_info = crispe_info.get("intent", "정보 없음")
    scope_info = crispe_info.get("scope", "정보 없음")
    preference_info = crispe_info.get("preference", "정보 없음")

    print(f"Generator: CRISPE info - context={context_info}, intent={intent_info}, scope={scope_info}, preference={preference_info}")
    print(f"Generator: has_search_tool_message={has_search_tool_message}, research_results_count={len(research_results)}")

    results_str = "\n\n".join(
        [f"제목: {r.get('title', 'N/A')}\n링크: {r.get('url', 'N/A')}\n내용: {r.get('content', 'N/A')}" for r in research_results]
    )

    try:
        # 검색을 거쳤는데 결과가 없는 경우
        if has_search_tool_message and not results_str.strip():
            response_content = "죄송합니다. 요청하신 내용에 대한 검색 결과를 찾을 수 없습니다."
            response_type = "search_summary"
            print("Generator: No search results found")
        # 검색 없이 직접 생성하는 경우 (자소서, 코드 작성 등)
        elif not has_search_tool_message:
            print("Generator: Direct generation mode (no research needed)")

            # CRISPE 정보가 있는지 확인
            has_crispe_info = any([
                context_info and context_info != "정보 없음",
                intent_info and intent_info != "정보 없음",
                scope_info and scope_info != "정보 없음",
                preference_info and preference_info != "정보 없음"
            ])

            if has_crispe_info:
                prompt = f"""## Role
사용자의 요청에 맞춰 고품질의 콘텐츠를 생성하는 한국어 어시스턴트입니다.

## 언어 규칙
- 반드시 100% 한국어로만 작성하세요
- 외국어 혼용 금지

## 사용자가 제공한 정보 (CRISPE)
- Context (상황): {context_info}
- Intent (목적): {intent_info}
- Scope (범위/조건): {scope_info}
- Preference (선호): {preference_info}

## 핵심 규칙
1. 사용자가 이미 말한 정보는 반복하지 마세요
2. 요청한 콘텐츠를 직접 작성해주세요
3. 구체적이고 실용적인 내용으로 작성하세요
4. 사용자의 상황과 목적에 맞게 맞춤형으로 작성하세요

[대화 내역]
{conversation_history}

위 정보를 바탕으로 사용자가 요청한 콘텐츠를 작성하세요:"""
            else:
                prompt = f"""## Role
사용자의 요청에 맞춰 고품질의 콘텐츠를 생성하는 한국어 어시스턴트입니다.

## 언어 규칙
- 반드시 100% 한국어로만 작성하세요
- 외국어 혼용 금지

## 핵심 규칙
1. 요청한 콘텐츠를 직접 작성해주세요
2. 구체적이고 실용적인 내용으로 작성하세요
3. 필요한 경우 구조화된 형식(제목, 소제목, 목록 등)을 사용하세요

[대화 내역]
{conversation_history}

위 요청에 맞는 콘텐츠를 작성하세요:"""

            try:
                response = text_model.invoke(prompt)
                response_content = response.content
                response_type = "generated_content"
                print(f"Generator: Generated content of length {len(response_content)}")
            except Exception as api_error:
                error_msg = str(api_error)
                print(f"Generator: API Error in direct generation: {error_msg}")
                response_content = "죄송합니다. 콘텐츠 생성 중 일시적인 오류가 발생했습니다. 다시 시도해주세요."
                response_type = "error"
        else:
            # CRISPE 정보가 있는지 확인
            has_crispe_info = any([
                context_info and context_info != "정보 없음",
                intent_info and intent_info != "정보 없음",
                scope_info and scope_info != "정보 없음",
                preference_info and preference_info != "정보 없음"
            ])

            if has_crispe_info:
                # CRISPE 정보가 있으면 맞춤형 프롬프트 사용
                prompt = f"""## Role
사용자 맞춤형 정보를 제공하는 한국어 어시스턴트입니다.

## 언어 규칙 (필수!!! 반드시 준수)
- 반드시 100% 한국어로만 작성하세요
- 중국어(汉语/漢語), 일본어 한자(日本語), 태국어, 영어 등 절대 사용 금지
- 일본 고유명사는 반드시 한글로 표기:
  - 후시미이나리 (○) / 伏見稲荷 (X)
  - 기요미즈데라 (○) / 清水寺 (X)
  - 센본토리이 (○) / 千本鳥居 (X)
  - 교토 (○) / 京都 (X)
- 음식명도 한글로: 텐푸라 (○), 라멘 (○), 덴뿌라 (X), 天ぷら (X)

## 사용자가 제공한 정보 (CRISPE) - 참고용, 반복 금지!
- Context (상황): {context_info}
- Intent (목적): {intent_info}
- Scope (범위/조건): {scope_info}
- Preference (선호): {preference_info}

## 핵심 규칙 (필수!!!)
1. **사용자가 이미 말한 정보는 절대 반복하지 마세요**
   - "예산이 200만원이시군요" (X) - 사용자가 이미 알고 있음
   - "1주일 여행을 계획하고 계시네요" (X) - 사용자가 이미 말함
   - 바로 새로운 정보/추천으로 시작하세요
2. **사용자가 모르는 새로운 정보만 제공하세요**
   - 현지인만 아는 팁, 숨겨진 명소, 실용적인 조언
   - 검색 결과에서 발견한 유용한 인사이트
3. **"~입니다" 단순 나열 대신 실용적인 팁 위주로 작성하세요**
   - "기요미즈데라는 유명한 절입니다" (X) - 단순 나열
   - "기요미즈데라는 새벽 6시에 가면 인파 없이 사진 찍기 좋아요" (O) - 유용한 팁

## 중요 지시사항
1. 위 CRISPE 정보에 **정확히 맞춰서** 답변을 작성하세요
2. 사용자가 말한 조건(지역, 예산, 기간 등)을 벗어나는 내용은 추천하지 마세요
3. 검색 결과 중 사용자 조건에 맞는 것만 선별해서 제공하세요
4. 사용자가 교토를 원하면 도쿄나 오사카 정보는 제외하세요
5. 예산 범위를 벗어나는 옵션은 언급하지 마세요

[대화 내역]
{conversation_history}

[검색 결과]
{results_str}

위 CRISPE 조건에 맞는 **새로운 정보와 유용한 팁** 위주로 답변하세요 (사용자가 말한 내용 반복 금지):"""
            else:
                # CRISPE 정보가 없으면 일반 프롬프트 사용
                prompt = f"""## Role
유용한 정보를 제공하는 한국어 어시스턴트입니다.

## 언어 규칙 (필수!!! 반드시 준수)
- 반드시 100% 한국어로만 작성하세요
- 중국어(汉语/漢語), 일본어 한자(日本語), 태국어, 영어 등 절대 사용 금지
- 외국 고유명사는 반드시 한글로 표기하세요
- 예: 후시미이나리 (○) / 伏見稲荷 (X), 도쿄 (○) / 東京 (X)

[대화 내역]
{conversation_history}

[검색 결과]
{results_str}

위 정보를 바탕으로 상세하고 유용한 답변을 한국어로만 작성하세요:"""

            try:
                response = text_model.invoke(prompt)
                response_content = response.content
                response_type = "search_summary"
                print(f"Generator: Generated response of length {len(response_content)}")
            except Exception as api_error:
                # API 할당량 초과 또는 기타 API 오류 시 검색 결과 기반으로 간단한 응답 생성
                error_msg = str(api_error)
                print(f"Generator: API Error: {error_msg}")

                if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg or "quota" in error_msg.lower():
                    # 할당량 초과 시, 검색 결과만으로 답변 작성
                    response_content = "검색 결과를 정리해드립니다:\n"
                    for i, result in enumerate(research_results, 1):
                        response_content += f"\n{i}. {result.get('title', 'N/A')}\n"
                        response_content += f"   {result.get('content', 'N/A')[:200]}...\n"
                        response_content += f"   출처: {result.get('url', 'N/A')}\n"

                    response_type = "search_summary"
                else:
                    response_content = "죄송합니다. 답변 생성 중 일시적인 오류가 발생했습니다. 다시 시도해주세요."
                    response_type = "search_summary"

    except Exception as e:
        print(f"Generator: Unexpected error: {e}")
        response_content = "죄송합니다. 처리 중 오류가 발생했습니다."
        response_type = "search_summary"

    # 중국어/일본어 등 비한글 문자 제거
    response_content = clean_non_korean(response_content)

    return {
        "messages": [AIMessage(content=response_content)],
        "response_type": response_type,
        "crispe_info": crispe_info  # CRISPE 정보를 state에 저장
    }

def reflect_node(state: AgentState):
    conversation_history = _get_conversation_history_windowed(state)
    critique = reflect_and_critique_tool(conversation_history)
    final_response = f"## 최종 보고서 초안:\n{state['messages'][-1].content}\n\n## 검토 및 제언:\n{critique}"
    return {
        "messages": [AIMessage(content=final_response)],
        "response_type": "final"
    }

def router_node(state: AgentState):
    msgs = state["messages"]
    updates = {}

    try:
        new_sum = _maybe_update_memory_summary(state)
        if new_sum is not None:
            updates["memory_summary"] = new_sum
    except Exception as e:
        print(f"Error updating memory summary: {e}")

    if len(msgs) == 0:
        return {**updates, "next_node": "end"}

    if isinstance(msgs[-1], HumanMessage):
        conversation_history = _get_conversation_history_windowed(state)
        print(f"Router: Processing user message: {conversation_history[:100]}...")

        # 기존 CRISPE 정보 가져오기
        existing_crispe = dict(state.get("crispe_info", {}) or {})

        # CRISPE 프레임워크 기반 모호성 판단 (기존 정보 전달)
        crispe_result = is_ambiguous_with_llm(conversation_history, existing_crispe)
        is_ambiguous = crispe_result.get("is_ambiguous", True)
        missing_elements = crispe_result.get("missing_elements", [])
        provided_info = crispe_result.get("provided_info", {})
        detected_domain = crispe_result.get("detected_domain", "기타")

        # 새로 제공된 정보와 기존 정보 병합
        for key, value in provided_info.items():
            if value:  # 새로 제공된 정보가 있으면 업데이트
                existing_crispe[key] = value

        print(f"Router: CRISPE provided_info: {provided_info}, merged: {existing_crispe}")

        # 선제적 질문 횟수 확인
        clarification_count = state.get("clarification_count", 0)

        if is_ambiguous:
            # 최대 질문 횟수 초과 시 그냥 진행
            if clarification_count >= MAX_CLARIFICATION_ROUNDS:
                print(f"Router: Max clarification rounds ({MAX_CLARIFICATION_ROUNDS}) reached, proceeding to researcher")
                return {
                    **updates,
                    "next_node": "researcher",
                    "missing_elements": [],
                    "crispe_info": existing_crispe,
                    "detected_domain": detected_domain
                }

            print(f"Router: Request is ambiguous (missing: {missing_elements}), routing to qa node (round {clarification_count + 1})")
            return {
                **updates,
                "next_node": "qa",
                "missing_elements": missing_elements,
                "crispe_info": existing_crispe,
                "detected_domain": detected_domain
            }

        # 명확한 요청 - 검색 필요 여부 판단
        research_result = needs_research_with_llm(conversation_history, existing_crispe)
        needs_research = research_result.get("needs_research", True)

        if needs_research:
            print(f"Router: Request needs research, routing to researcher. Reason: {research_result.get('reason', '')}")
            return {
                **updates,
                "next_node": "researcher",
                "missing_elements": [],
                "crispe_info": existing_crispe,
                "detected_domain": detected_domain
            }
        else:
            print(f"Router: Request doesn't need research, routing directly to generator. Reason: {research_result.get('reason', '')}")
            return {
                **updates,
                "next_node": "generator",
                "missing_elements": [],
                "crispe_info": existing_crispe,
                "detected_domain": detected_domain
            }

    last = msgs[-1]
    if isinstance(last, AIMessage):
        rt = state.get("response_type")
        print(f"Router: AIMessage with response_type: {rt}")
        if rt == "final" or rt == "search_summary":
            return {**updates, "next_node": "end"}
        if rt == "report_draft":
            return {**updates, "response_type": "reflecting", "next_node": "reflect"}
        return {**updates, "next_node": "end"}

    elif isinstance(last, ToolMessage):
        print(f"Router: ToolMessage with name: {last.name}")
        if last.name == "generate_clarifying_question_tool":
            return {**updates, "next_node": "end"}
        elif last.name == "create_plan_tool":
            return {**updates, "next_node": "researcher"}
        elif last.name == "web_search_tool":
            return {**updates, "next_node": "generator"}

    print(f"Router: Default to end")
    return {**updates, "next_node": "end"}

# ==========================================================
# 그래프 조립
# ==========================================================
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("qa", qa_node)
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("generator", generator_node)
workflow.add_node("reflect", reflect_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    lambda state: state["next_node"],
    {
        "qa": "qa",
        "planner": "planner",
        "researcher": "researcher",
        "generator": "generator",
        "reflect": "reflect",
        "end": END
    }
)

workflow.add_edge("planner", "router")
workflow.add_edge("researcher", "router")
workflow.add_edge("generator", "router")
workflow.add_edge("qa", END)
workflow.add_edge("reflect", END)

memory = InMemorySaver()
langgraph_app = workflow.compile(checkpointer=memory)

# ==========================================================
# FastAPI 서버
# ==========================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ChatResponse(BaseModel):
    response: str
    response_type: str
    questions: Optional[List[dict]] = None

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    
    print(f"Chat API called with message: {request.message[:100]}")
    
    try:
        # messages만 전달하면 나머지 state는 기존 값 유지됨 (crispe_info, clarification_count 누적)
        events = langgraph_app.stream(
            {
                "messages": [HumanMessage(content=request.message)],
            },
            config,
            stream_mode="values"
        )
        
        final_response = ""
        response_type = "message"
        questions = None
        
        for event in events:
            if not event.get("messages"):
                continue
                
            last_message = event["messages"][-1]
            
            if isinstance(last_message, AIMessage):
                final_response = last_message.content
                response_type = event.get("response_type", "message")
                print(f"Got AIMessage response: {len(final_response)} chars")
            elif isinstance(last_message, ToolMessage):
                if last_message.name == "generate_clarifying_question_tool":
                    try:
                        questions = json.loads(last_message.content)
                        # LLM이 {"questions": [...]} 또는 {"clarifyingQuestions": [...]} 형태로 반환할 경우 처리
                        if isinstance(questions, dict):
                            if "questions" in questions:
                                questions = questions["questions"]
                            elif "clarifyingQuestions" in questions:
                                questions = questions["clarifyingQuestions"]
                        response_type = "clarifying_questions"
                        # 질문들을 텍스트로 포맷팅하여 response에 포함
                        formatted_questions = "더 정확한 답변을 드리기 위해 몇 가지 여쭤볼게요:\n\n"
                        for i, q in enumerate(questions, 1):
                            question_text = q.get("question", "")
                            choices = q.get("choices", [])
                            formatted_questions += f"**{i}. {question_text}**\n"
                            for j, choice in enumerate(choices, 1):
                                formatted_questions += f"   {j}. {choice}\n"
                            formatted_questions += "\n"
                        final_response = formatted_questions.strip()
                    except Exception as e:
                        print(f"Error formatting questions: {e}")
                        pass
        
        if not final_response:
            final_response = "죄송합니다. 처리 중 오류가 발생했습니다."
            
        print(f"Returning response: type={response_type}, length={len(final_response)}")
        
        return ChatResponse(
            response=final_response,
            response_type=response_type,
            questions=questions
        )
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return ChatResponse(
            response=f"오류가 발생했습니다: {str(e)}",
            response_type="error"
        )

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
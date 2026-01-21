# server.py
import os
import json
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# ==========================================================
# 환경 설정
# ==========================================================
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Groq 무료 모델 사용 (llama-3.3-70b-versatile)
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

# JSON 응답용 모델 (Groq은 json_mode 지원)
json_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# ==========================================================
# AgentState 정의
# ==========================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    next_node: str
    response_type: str
    memory_summary: str

CONTEXT_BUDGET_TOKENS = 8000
SUMMARIZE_AT_TOKENS = 12000
SUMMARY_MAX_TOKENS = 600

# ==========================================================
# 헬퍼 함수들
# ==========================================================
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
        summary = model.invoke(prompt).content.strip()
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

# ==========================================================
# LLM 판단 함수들
# ==========================================================
def is_ambiguous_with_llm(query: str) -> bool:
    system_prompt = """
    You are an expert at judging the ambiguity of user requests.

    Return is_ambiguous: true if ANY of the following apply:
    - Travel requests: duration, budget, companions, or interests/purpose not specified
    - Recommendations: use case, budget, or preferences not specified
    - Technical questions: environment, purpose, or constraints unclear
    - General requests: no specific conditions or constraints given
    - Simple keywords only: e.g., "Japan travel", "restaurant recommendations"

    A request is only clear if at least 3 specific conditions are provided.
    If ambiguous return true, if clear return false.

    Respond ONLY in JSON: {"is_ambiguous": boolean, "reason": "..."}
    """
    try:
        response = json_model.invoke(system_prompt + f"\n\nConversation History:\n\"{query}\"")
        result = json.loads(response.content)
        return result.get("is_ambiguous", True)
    except Exception as e:
        print(f"Error in is_ambiguous_with_llm: {e}")
        return True

def classify_intent_with_llm(query: str) -> str:
    system_prompt = """
    Classify the user request into 'search' or 'report'.
    - 'search': find specific information
    - 'report': create structured document
    Respond ONLY with 'search' or 'report'.
    """
    try:
        response = model.invoke(system_prompt + f"\n\nUser Request:\n\"{query}\"")
        result = response.content.strip().lower()
        if result in ['search', 'report']:
            return result
        return 'search'
    except Exception as e:
        print(f"Error in classify_intent_with_llm: {e}")
        return 'search'

def _generate_search_query_with_llm(plan_description: str, conversation_history: str) -> str:
    system_prompt = """
    Convert the plan description into a concise search query.
    Return ONLY the search query as plain string.
    """
    try:
        response = model.invoke(system_prompt + f"\n\nPlan: \"{plan_description}\"\nHistory: \"{conversation_history}\"")
        query = response.content.strip()
        return query if query else plan_description
    except Exception as e:
        print(f"Error in _generate_search_query_with_llm: {e}")
        return plan_description or conversation_history[-100:]

# ==========================================================
# Tool 함수들
# ==========================================================
def generate_clarifying_question_tool(query: str) -> List[dict]:
    system_prompt = """
    Generate clarifying questions in JSON array format.
    Each item: {"question": "...", "choices": ["...", "...", "..."]}
    ALWAYS respond in Korean.
    """
    response = json_model.invoke(system_prompt + f'\n\nuserInput: "{query}"')
    result = json.loads(response.content)
    # LLM이 {"questions": [...]} 또는 {"clarifyingQuestions": [...]} 형태로 반환할 경우 처리
    if isinstance(result, dict):
        if "questions" in result:
            return result["questions"]
        if "clarifyingQuestions" in result:
            return result["clarifyingQuestions"]
    return result

def create_plan_tool(query: str) -> List[Dict[str, Any]]:
    system_prompt = """
    Create a research plan as JSON array.
    Each step: {"id": 1, "title": "...", "description": "...", "status": "pending", "research": true, "dependencies": [], "complexity": 3}
    Return ONLY JSON array.
    """
    response = json_model.invoke(system_prompt + f"\n\nUser Request: \"{query}\"")
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return []

def web_search_tool(query: str) -> List[Dict[str, Any]]:
    try:
        search_results = tavily_client.search(query, search_depth="advanced", max_results=3)
        return search_results.get('results', []) or []
    except Exception:
        return []

def reflect_and_critique_tool(conversation: str) -> str:
    system_prompt = """
    Review the conversation and critique the draft. Suggest improvements.
    """
    response = model.invoke(system_prompt + f"\n\nConversation:\n\"{conversation}\"")
    return response.content

# ==========================================================
# 노드들
# ==========================================================
def qa_node(state: AgentState):
    last_user = _get_last_user_message(state)
    questions = generate_clarifying_question_tool(last_user)
    return {
        "messages": [
            ToolMessage(
                name="generate_clarifying_question_tool",
                content=json.dumps(questions, ensure_ascii=False),
                tool_call_id="manual_qa"
            )
        ]
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

    for m in reversed(state["messages"]):
        if isinstance(m, ToolMessage) and m.name == "web_search_tool":
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

    results_str = "\n\n".join(
        [f"제목: {r.get('title', 'N/A')}\n링크: {r.get('url', 'N/A')}\n내용: {r.get('content', 'N/A')}" for r in research_results]
    )

    try:
        if not results_str.strip():
            response_content = "죄송합니다. 요청하신 내용에 대한 검색 결과를 찾을 수 없습니다."
            response_type = "search_summary"
            print("Generator: No search results found")
        else:
            prompt = f"""당신은 유용한 정보를 제공하는 어시스턴트입니다.
사용자의 요청과 검색 결과를 바탕으로 한국어로 친절하고 상세한 답변을 제공하세요.

[사용자 요청]
{conversation_history}

[검색 결과]
{results_str}

상세하고 유용한 답변을 한국어로 작성하세요:"""
            
            try:
                response = model.invoke(prompt)
                response_content = response.content
                response_type = "search_summary"
                print(f"Generator: Generated response of length {len(response_content)}")
            except Exception as api_error:
                # API 할당량 초과 또는 기타 API 오류 시 검색 결과 기반으로 간단한 응답 생성
                error_msg = str(api_error)
                print(f"Generator: API Error: {error_msg}")
                
                if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg or "quota" in error_msg.lower():
                    # 할당량 초과 시, 검색 결과만으로 답변 작성
                    response_content = f"""일본 여행 계획에 대한 검색 결과입니다:

"""
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

    return {
        "messages": [AIMessage(content=response_content)],
        "response_type": response_type
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

        # 선제적 질문 시스템: 모호한 요청인지 판단
        if is_ambiguous_with_llm(conversation_history):
            print("Router: Request is ambiguous, routing to qa node")
            return {**updates, "next_node": "qa"}

        # 명확한 요청이면 researcher로 라우팅
        print("Router: Request is clear, routing to researcher")
        return {**updates, "next_node": "researcher"}

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
        events = langgraph_app.stream(
            {"messages": [HumanMessage(content=request.message)]},
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
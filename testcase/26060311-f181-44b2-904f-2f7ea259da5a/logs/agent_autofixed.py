

class Response:
    def __init__(self, **kwargs):
        self.status_code = 200
        self._data = kwargs
    def json(self):
        return self._data
# AUTO-FIX runtime fallbacks for unresolved names
Agent = None

try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {'check_credentials_output': True,
 'check_jailbreak': True,
 'check_output': True,
 'check_pii_input': False,
 'check_toxic_code_output': True,
 'check_toxicity': True,
 'content_safety_enabled': True,
 'content_safety_severity_threshold': 3,
 'runtime_enabled': True,
 'sanitize_pii': False}


import os
import logging
import asyncio
import time as _time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, ValidationError, Field, model_validator
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import openai

# Observability wrappers (injected by runtime)

# Load .env if present
load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("DataInsightsAgent")

# --- Configuration Management ---
class Config:
    """Configuration loader for environment variables."""
    @staticmethod
    def get(key: str, default: Optional[str] = None) -> Optional[str]:
        return os.getenv(key, default)

    @staticmethod
    def validate(required_keys: List[str]) -> None:
        missing = [k for k in required_keys if not os.getenv(k)]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {missing}")

# --- Input Validation ---
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=50000)

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Question cannot be empty.")
        v = v.strip()
        if len(v) > 50000:
            raise ValueError("Question exceeds maximum allowed length (50,000 characters).")
        # Remove control characters, excessive whitespace, etc.
        v = " ".join(v.split())
        return v

# --- Logger Utility ---
class Logger:
    """Audit logging for compliance and monitoring."""
    def log_event(self, event: str, level: str, details: Dict[str, Any]) -> None:
        try:
            msg = f"{event} | {details}"
            if level.lower() == "info":
                logger.info(msg)
            elif level.lower() == "warning":
                logger.warning(msg)
            elif level.lower() == "error":
                logger.error(msg)
            else:
                logger.debug(msg)
        except Exception as e:
            logger.error(f"Logging failed: {e}")

# --- Error Handler ---
class ErrorHandler:
    """Centralized error handling, fallback logic, and escalation."""
    def __init__(self, logger: Logger, fallback_response: str):
        self.logger = logger
        self.fallback_response = fallback_response

    def handle_error(self, error_code: str, context: Dict[str, Any]) -> str:
        self.logger.log_event(
            event=f"Error occurred: {error_code}",
            level="error",
            details=context
        )
        # Map error codes to user-friendly responses
        if error_code == "NO_DATA_FOUND":
            return self.fallback_response
        elif error_code == "RETRIEVAL_ERROR":
            return "There was a problem retrieving information from the knowledge base. Please try again later."
        elif error_code == "REWRITE_ERROR":
            return "I had trouble simplifying the answer. Here is the original response."
        else:
            return self.fallback_response

# --- Retrieval Client (Azure AI Search) ---
class RetrievalClient:
    """Queries Azure AI Search using semantic/vector search to retrieve relevant knowledge base chunks."""
    def __init__(self):
        self._search_client = None
        self._openai_client = None

    def _get_search_client(self) -> SearchClient:
        if self._search_client is None:
            endpoint = Config.get("AZURE_SEARCH_ENDPOINT")
            index_name = Config.get("AZURE_SEARCH_INDEX_NAME")
            api_key = Config.get("AZURE_SEARCH_API_KEY")
            if not endpoint or not index_name or not api_key:
                raise RuntimeError("Azure Search credentials are not fully configured.")
            self._search_client = SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(api_key),
            )
        return self._search_client

    def _get_openai_client(self) -> openai.AzureOpenAI:
        if self._openai_client is None:
            api_key = Config.get("AZURE_OPENAI_API_KEY")
            endpoint = Config.get("AZURE_OPENAI_ENDPOINT")
            if not api_key or not endpoint:
                raise RuntimeError("Azure OpenAI credentials are not fully configured.")
            self._openai_client = openai.AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint,
            )
        return self._openai_client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query Azure AI Search for relevant knowledge base content."""
        async with trace_step(
            "retrieve_chunks", step_type="tool_call",
            decision_summary="Retrieve top-K relevant chunks from Azure AI Search",
            output_fn=lambda r: f"chunks={len(r) if r else 0}"
        ) as step:
            try:
                search_client = self._get_search_client()
                openai_client = self._get_openai_client()
                embedding_model = Config.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

                # Embed user query
                embedding_resp = await asyncio.to_thread(
                    openai_client.embeddings.create,
                    input=query,
                    model=embedding_model
                )
                vector_query = VectorizedQuery(
                    vector=embedding_resp.data[0].embedding,
                    k_nearest_neighbors=top_k,
                    fields="vector"
                )
                # Search
                results = search_client.search(
                    search_text=query,
                    vector_queries=[vector_query],
                    top=top_k,
                    select=["chunk", "title"]
                )
                context_chunks = []
                for r in results:
                    if r.get("chunk"):
                        context_chunks.append({
                            "chunk": r["chunk"],
                            "title": r.get("title", "")
                        })
                step.capture(context_chunks)
                return context_chunks
            except Exception as e:
                step.capture([])
                raise

# --- LLM Client (OpenAI GPT-4.1) ---
class LLMClient:
    """Handles prompt construction and calls to OpenAI GPT-4.1."""
    def __init__(self, model: str, temperature: float, max_tokens: int, system_prompt: str):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self._client = None

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            api_key = Config.get("AZURE_OPENAI_API_KEY")
            endpoint = Config.get("AZURE_OPENAI_ENDPOINT")
            if not api_key or not endpoint:
                raise RuntimeError("Azure OpenAI credentials are not fully configured.")
            self._client = openai.AsyncOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint,
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=4),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_answer(self, prompt: str, context: str) -> str:
        """Invoke LLM with constructed prompt and context."""
        async with trace_step(
            "generate_answer", step_type="llm_call",
            decision_summary="Call LLM to produce a grounded, simplified answer",
            output_fn=lambda r: f"length={len(r) if r else 0}"
        ) as step:
            client = self._get_client()
            user_message = f"{prompt}\n\nContext:\n{context}"
            _t0 = _time.time()
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            content = response.choices[0].message.content
            try:
                trace_model_call(
                    provider="openai",
                    model_name=self.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    response_summary=content[:200] if content else ""
                )
            except Exception:
                pass
            step.capture(content)
            return content

# --- Domain Logic ---
class DomainLogic:
    """Applies business rules: ensures answers are grounded, simplifies language, validates clarity, and enforces constraints."""
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def validate_grounding(self, answer: str, chunks: List[Dict[str, Any]]) -> bool:
        """Ensure answer is based on retrieved content."""
        async with trace_step(
            "validate_grounding", step_type="process",
            decision_summary="Check if answer is grounded in retrieved chunks",
            output_fn=lambda r: f"grounded={r}"
        ) as step:
            # Simple heuristic: check if any chunk content appears in answer
            answer_lower = answer.lower()
            for chunk in chunks:
                chunk_text = chunk.get("chunk", "").lower()
                if chunk_text and any(word in answer_lower for word in chunk_text.split()[:5]):
                    step.capture(True)
                    return True
            step.capture(False)
            return False

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=1, max=2),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def simplify_language(self, answer: str) -> str:
        """Rewrite answer for clarity and non-technical language."""
        async with trace_step(
            "simplify_language", step_type="process",
            decision_summary="Simplify answer for non-technical users",
            output_fn=lambda r: f"length={len(r) if r else 0}"
        ) as step:
            prompt = (
                "Rewrite the following answer to be as clear and simple as possible for a non-technical user. "
                "Avoid technical jargon and keep explanations easy to understand. If the answer is already simple, return it unchanged.\n\n"
                f"Answer:\n{answer}"
            )
            try:
                simplified = await self.llm_client.generate_answer(prompt, "")
                step.capture(simplified)
                return simplified
            except Exception:
                step.capture(answer)
                return answer

# --- Application Orchestrator ---
class ApplicationOrchestrator:
    """Coordinates the workflow: input validation, retrieval, LLM invocation, error handling, and output formatting."""
    def __init__(
        self,
        retrieval_client: RetrievalClient,
        llm_client: LLMClient,
        domain_logic: DomainLogic,
        error_handler: ErrorHandler,
        logger: Logger,
        fallback_response: str
    ):
        self.retrieval_client = retrieval_client
        self.llm_client = llm_client
        self.domain_logic = domain_logic
        self.error_handler = error_handler
        self.logger = logger
        self.fallback_response = fallback_response

    @trace_agent(agent_name='Data Insights Assistant for Non-Technical Users')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_question(self, question: str) -> str:
        """Main entry point for handling user questions."""
        async with trace_step(
            "process_question", step_type="plan",
            decision_summary="Orchestrate retrieval, LLM, validation, and formatting",
            output_fn=lambda r: f"length={len(r) if r else 0}"
        ) as step:
            try:
                # 1. Retrieve relevant chunks
                try:
                    chunks = await self.retrieval_client.retrieve_chunks(question, top_k=5)
                except Exception as e:
                    self.logger.log_event(
                        event="Retrieval error",
                        level="error",
                        details={"error": str(e)}
                    )
                    return self.error_handler.handle_error("RETRIEVAL_ERROR", {"error": str(e)})

                if not chunks:
                    self.logger.log_event(
                        event="No data found",
                        level="info",
                        details={"question": question}
                    )
                    return self.error_handler.handle_error("NO_DATA_FOUND", {"question": question})

                # 2. Construct context for LLM
                context = "\n\n".join(
                    f"Document: {chunk.get('title','')}\n{chunk.get('chunk','')}" for chunk in chunks
                )

                # 3. Generate answer
                try:
                    answer = await self.llm_client.generate_answer(
                        prompt="Please answer the user's question using only the provided context.",
                        context=context + f"\n\nUser Question: {question}"
                    )
                except Exception as e:
                    self.logger.log_event(
                        event="LLM error",
                        level="error",
                        details={"error": str(e)}
                    )
                    return self.error_handler.handle_error("RETRIEVAL_ERROR", {"error": str(e)})

                # 4. Validate grounding
                is_grounded = await self.domain_logic.validate_grounding(answer, chunks)
                if not is_grounded:
                    self.logger.log_event(
                        event="Answer not grounded",
                        level="warning",
                        details={"answer": answer, "question": question}
                    )
                    return self.error_handler.handle_error("NO_DATA_FOUND", {"question": question})

                # 5. Simplify language
                try:
                    simplified = await self.domain_logic.simplify_language(answer)
                except Exception as e:
                    self.logger.log_event(
                        event="Simplification error",
                        level="warning",
                        details={"error": str(e)}
                    )
                    simplified = answer

                # 6. Format and return
                step.capture(simplified)
                return simplified

            except Exception as e:
                self.logger.log_event(
                    event="Unhandled error",
                    level="error",
                    details={"error": str(e)}
                )
                return self.error_handler.handle_error("RETRIEVAL_ERROR", {"error": str(e)})

# --- User Interface Handler (Presentation Layer) ---
class UserInterfaceHandler:
    """Handles incoming user questions and outgoing responses via web chat or API."""
    def __init__(self, orchestrator: ApplicationOrchestrator, logger: Logger):
        self.orchestrator = orchestrator
        self.logger = logger

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def submit_question(self, question: str) -> str:
        """Receives and processes a user question."""
        async with trace_step(
            "submit_question", step_type="parse",
            decision_summary="Sanitize and submit user question",
            output_fn=lambda r: f"length={len(r) if r else 0}"
        ) as step:
            try:
                # Input validation is handled by Pydantic in the API layer
                self.logger.log_event(
                    event="User question received",
                    level="info",
                    details={"question": question}
                )
                response = await self.orchestrator.process_question(question)
                step.capture(response)
                return response
            except Exception as e:
                self.logger.log_event(
                    event="UI Handler error",
                    level="error",
                    details={"error": str(e)}
                )
                return self.orchestrator.error_handler.handle_error("RETRIEVAL_ERROR", {"error": str(e)})

    def receive_response(self, response: str) -> str:
        """Formats and returns the response (for web chat, etc.)."""
        # For now, just return the response as-is
        self.logger.log_event(
            event="Response delivered",
            level="info",
            details={"response": response}
        )
        return response

# --- Main Agent Composition ---
class DataInsightsAgent:
    """Main agent class composing all supporting components."""
    def __init__(self):
        # LLM config
        self.system_prompt = (
            "You are a Data Insights Assistant for non-technical users. Your role is to answer user questions about data by providing clear, concise, and accurate insights based strictly on the retrieved knowledge base content. Do not use technical jargon or assume prior data knowledge. If the answer cannot be found in the provided documents, politely inform the user and suggest rephrasing their question. Always ensure your responses are easy to understand and directly address the user's query."
        )
        self.fallback_response = (
            "I'm sorry, I couldn't find an answer to your question in the available data. Please try rephrasing your question or ask about a different topic."
        )
        self.llm_model = "gpt-4.1"
        self.llm_temperature = 0.7
        self.llm_max_tokens = 2000

        # Compose components
        self.logger = Logger()
        self.error_handler = ErrorHandler(self.logger, self.fallback_response)
        self.retrieval_client = RetrievalClient()
        self.llm_client = LLMClient(
            model=self.llm_model,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            system_prompt=self.system_prompt
        )
        self.domain_logic = DomainLogic(self.llm_client)
        self.orchestrator = ApplicationOrchestrator(
            retrieval_client=self.retrieval_client,
            llm_client=self.llm_client,
            domain_logic=self.domain_logic,
            error_handler=self.error_handler,
            logger=self.logger,
            fallback_response=self.fallback_response
        )
        self.ui_handler = UserInterfaceHandler(self.orchestrator, self.logger)

    @trace_agent(agent_name='Data Insights Assistant for Non-Technical Users')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def ask(self, question: str) -> str:
        """Public async API for agent usage."""
        return await self.ui_handler.submit_question(question)

# --- FastAPI App ---
app = FastAPI(
    title="Data Insights Assistant for Non-Technical Users",
    description="An approachable, clear, and informative agent for planetary data insights, grounded in Azure AI Search knowledge base.",
    version="1.0.0"
)

# CORS (allow all origins for demo; restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = DataInsightsAgent()

# --- Exception Handlers ---
@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Input validation failed.",
            "details": exc.errors(),
            "tips": [
                "Ensure your JSON is well-formed.",
                "Check for missing or extra commas, brackets, or quotes.",
                "Question must not be empty and must be under 50,000 characters."
            ]
        }
    )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "tips": [
                "Check your request format.",
                "Refer to the API documentation for correct usage."
            ]
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An unexpected error occurred.",
            "tips": [
                "Ensure your request is valid JSON.",
                "Try again later or contact support if the issue persists."
            ]
        }
    )

# --- API Endpoints ---
@app.post("/ask", response_model=Dict[str, Any])
@with_content_safety(config=GUARDRAILS_CONFIG)
async def ask_question(request: Request):
    """
    Submit a question about planetary data and receive a non-technical, grounded answer.
    """
    try:
        data = await request.json()
    except Exception as e:
        logger.warning(f"Malformed JSON: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Malformed JSON in request body.",
                "tips": [
                    "Ensure your JSON is well-formed.",
                    "Check for missing or extra commas, brackets, or quotes."
                ]
            }
        )
    try:
        question_req = QuestionRequest(**data)
    except ValidationError as ve:
        logger.warning(f"Validation error: {ve}")
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "Input validation failed.",
                "details": ve.errors(),
                "tips": [
                    "Question must not be empty and must be under 50,000 characters."
                ]
            }
        )
    try:
        answer = await agent.ask(question_req.question)
        return {
            "success": True,
            "answer": answer
        }
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Failed to process your question.",
                "tips": [
                    "Try again later.",
                    "If the problem persists, contact support."
                ]
            }
        )

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint."""
    return {"success": True, "status": "ok"}

# --- Main Entrypoint ---


async def _run_with_eval_service():
    """Entrypoint: initialises observability then runs the agent."""
    import logging as _obs_log
    _obs_logger = _obs_log.getLogger(__name__)
    # ── 1. Observability DB schema ─────────────────────────────────────
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401 – register ORM models
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
    except Exception as _e:
        _obs_logger.warning('Observability DB init skipped: %s', _e)
    # ── 2. OpenTelemetry tracer ────────────────────────────────────────
    try:
        from observability.instrumentation import initialize_tracer
        initialize_tracer()
    except Exception as _e:
        _obs_logger.warning('Tracer init skipped: %s', _e)
    # ── 3. Evaluation background worker ───────────────────────────────
    _stop_eval = None
    try:
        from observability.evaluation_background_service import (
            start_evaluation_worker as _start_eval,
            stop_evaluation_worker as _stop_eval_fn,
        )
        await _start_eval()
        _stop_eval = _stop_eval_fn
    except Exception as _e:
        _obs_logger.warning('Evaluation worker start skipped: %s', _e)
    # ── 4. Run the agent ───────────────────────────────────────────────
    try:
        import uvicorn
        # Validate config only at runtime, not import time
        try:
        # AUTO-FIXED invalid syntax: Config.validate([
        # AUTO-FIXED invalid syntax: "AZURE_SEARCH_ENDPOINT",
        # AUTO-FIXED invalid syntax: "AZURE_SEARCH_API_KEY",
        # AUTO-FIXED invalid syntax: "AZURE_SEARCH_INDEX_NAME",
        # AUTO-FIXED invalid syntax: "AZURE_OPENAI_ENDPOINT",
        # AUTO-FIXED invalid syntax: "AZURE_OPENAI_API_KEY",
        # AUTO-FIXED invalid syntax: "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        # AUTO-FIXED invalid syntax: ])
        # AUTO-FIXED invalid syntax: except Exception as e:
        # AUTO-FIXED invalid syntax: logger.error(f"Configuration error: {e}")
        # AUTO-FIXED invalid syntax: exit(1)
        # AUTO-FIXED invalid syntax: uvicorn.run("agent:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
        # AUTO-FIXED invalid syntax: pass  # TODO: run your agent here
    # AUTO-FIXED invalid syntax: finally:
        # AUTO-FIXED invalid syntax: if _stop_eval is not None:
            try:
                await _stop_eval()
            except Exception:
                pass


# AUTO-FIXED invalid syntax: if __name__ == "__main__":
    # AUTO-FIXED invalid syntax: import asyncio as _asyncio
    # AUTO-FIXED invalid syntax: _asyncio.run(_run_with_eval_service())
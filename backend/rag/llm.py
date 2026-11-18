"""Cliente de modelo de lenguaje independiente del proveedor.

El enunciado (5.5, opcion A) pide "invocar a un modelo de lenguaje (API LLM o
modelo local)". Para que la demo no dependa de una cuenta concreta, el
proveedor se elige por variables de entorno:

    LLM_PROVIDER = anthropic | openai | gemini | groq | ollama | local | extractive
                   (si no se define, se detecta por la clave disponible)
    ANTHROPIC_API_KEY        -> anthropic  (SDK oficial, modelo claude-opus-5)
    GEMINI_API_KEY           -> gemini     (endpoint compatible OpenAI de Google)
    GROQ_API_KEY             -> groq
    OPENAI_API_KEY           -> openai
    LLM_BASE_URL + LLM_API_KEY -> cualquier API compatible con /chat/completions
    LLM_MODEL                -> sobreescribe el modelo por defecto del proveedor

`local` carga un modelo pequeno con transformers en CPU (Qwen2.5-1.5B-Instruct,
LLM_LOCAL_MODEL para cambiarlo): sin costo pero lento y con una descarga de
~3 GB la primera vez, por eso no se activa solo.

`extractive` no genera: arma la respuesta con los fragmentos recuperados. Es
el respaldo cuando no hay ninguna clave, para que el pipeline completo
(recuperacion, prompt, citas, metricas) siga siendo demostrable.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.5-flash-lite",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.2",
    "local": "Qwen/Qwen2.5-1.5B-Instruct",
    "extractive": "extractivo",
}

OPENAI_COMPAT = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "ollama": ("http://127.0.0.1:11434/v1", None),
}


@dataclass
class Generation:
    text: str
    provider: str
    model: str
    ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    note: str | None = None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def detect_provider(env: dict | None = None) -> str:
    env = os.environ if env is None else env
    forzado = (env.get("LLM_PROVIDER") or "").strip().lower()
    if forzado:
        return forzado
    if env.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if env.get("GEMINI_API_KEY"):
        return "gemini"
    if env.get("GROQ_API_KEY"):
        return "groq"
    if env.get("OPENAI_API_KEY"):
        return "openai"
    if env.get("LLM_BASE_URL"):
        return "openai"
    return "extractive"


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None, env: dict | None = None):
        env = os.environ if env is None else env
        self.provider = provider or detect_provider(env)
        self.model = model or env.get("LLM_MODEL") or DEFAULT_MODELS.get(self.provider, "")
        self.env = env
        self._local = None

    # --------------------------------------------------------------- publico

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model, "generates": self.provider != "extractive"}

    def generate(self, system: str, user: str, max_tokens: int = 1024, fragments: list[str] | None = None) -> Generation:
        t0 = time.perf_counter()
        if self.provider == "anthropic":
            text, meta = self._anthropic(system, user, max_tokens)
        elif self.provider in OPENAI_COMPAT or self.provider == "openai":
            text, meta = self._openai_compat(system, user, max_tokens)
        elif self.provider == "local":
            text, meta = self._local_generate(system, user, max_tokens)
        elif self.provider == "extractive":
            text, meta = self._extractive(fragments or []), {"note": "sin LLM configurado: respuesta extractiva"}
        else:
            raise ValueError(f"LLM_PROVIDER desconocido: {self.provider}")
        return Generation(text=text, provider=self.provider, model=self.model,
                          ms=round((time.perf_counter() - t0) * 1000.0, 1), **meta)

    # ------------------------------------------------------------ anthropic

    def _anthropic(self, system: str, user: str, max_tokens: int) -> tuple[str, dict]:
        import anthropic

        client = anthropic.Anthropic()
        messages = [{"role": "user", "content": user}]
        try:
            # Fallback del lado del servidor ante un rechazo de seguridad: la
            # misma peticion se reintenta en otro modelo dentro de la llamada.
            resp = client.beta.messages.create(
                model=self.model, max_tokens=max_tokens, system=system, messages=messages,
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            )
        except (TypeError, anthropic.BadRequestError):
            resp = client.messages.create(model=self.model, max_tokens=max_tokens, system=system, messages=messages)
        if resp.stop_reason == "refusal":
            detalle = getattr(resp, "stop_details", None)
            return ("El modelo declino responder esta pregunta"
                    + (f" ({detalle.category})." if detalle and getattr(detalle, "category", None) else "."),
                    {"note": "refusal"})
        texto = "".join(b.text for b in resp.content if b.type == "text").strip()
        uso = getattr(resp, "usage", None)
        return texto, {"input_tokens": getattr(uso, "input_tokens", None),
                       "output_tokens": getattr(uso, "output_tokens", None)}

    # ------------------------------------------------------ openai-compatible

    def _openai_compat(self, system: str, user: str, max_tokens: int) -> tuple[str, dict]:
        base_default, key_var = OPENAI_COMPAT.get(self.provider, OPENAI_COMPAT["openai"])
        base = (self.env.get("LLM_BASE_URL") or base_default).rstrip("/")
        key = self.env.get("LLM_API_KEY") or (self.env.get(key_var) if key_var else None) or "ollama"
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
        except urllib.error.HTTPError as exc:
            detalle = exc.read().decode("utf-8", "ignore")[:300]
            raise RuntimeError(f"{self.provider} respondio HTTP {exc.code}: {detalle}") from None
        texto = (data["choices"][0]["message"].get("content") or "").strip()
        uso = data.get("usage") or {}
        return texto, {"input_tokens": uso.get("prompt_tokens"), "output_tokens": uso.get("completion_tokens")}

    # ----------------------------------------------------------------- local

    def _local_generate(self, system: str, user: str, max_tokens: int) -> tuple[str, dict]:
        if self._local is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            nombre = self.env.get("LLM_LOCAL_MODEL") or self.model
            tok = AutoTokenizer.from_pretrained(nombre)
            mdl = AutoModelForCausalLM.from_pretrained(nombre, torch_dtype=torch.float32)
            mdl.eval()
            self._local = (tok, mdl)
            self.model = nombre
        tok, mdl = self._local
        import torch

        chat = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
        entrada = tok(chat, return_tensors="pt")
        with torch.inference_mode():
            salida = mdl.generate(**entrada, max_new_tokens=min(max_tokens, 400), do_sample=False)
        nuevos = salida[0][entrada["input_ids"].shape[1]:]
        return tok.decode(nuevos, skip_special_tokens=True).strip(), {
            "input_tokens": int(entrada["input_ids"].shape[1]), "output_tokens": int(len(nuevos))}

    # ------------------------------------------------------------ extractive

    @staticmethod
    def _extractive(fragments: list[str], max_chars: int = 900) -> str:
        if not fragments:
            return "No se recuperaron fragmentos relevantes para responder."
        partes, usado = [], 0
        for i, frag in enumerate(fragments, 1):
            trozo = frag.strip()
            if usado + len(trozo) > max_chars:
                trozo = trozo[: max(0, max_chars - usado)].rsplit(" ", 1)[0] + "…"
            partes.append(f"[{i}] {trozo}")
            usado += len(trozo)
            if usado >= max_chars:
                break
        return ("Respuesta extractiva (no hay LLM configurado; define ANTHROPIC_API_KEY o GEMINI_API_KEY "
                "para generar texto). Los fragmentos mas relevantes dicen:\n\n" + "\n\n".join(partes))

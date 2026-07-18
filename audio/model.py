from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Word(BaseModel):
    """A single word/token, normalized across providers."""
    text: str
    start: float | None = None  # seconds
    end: float | None = None  # seconds
    confidence: Optional[float] = None  # normalized to 0-1 where possible
    speaker: Optional[str] = None
    raw: Optional[dict] = None  # original provider fields for this word (logprob, etc.)


class Utterance(BaseModel):
    """One speaker turn / sentence. This is the atomic chunking unit."""
    text: str
    start: float | None = None  # seconds
    end: float | None = None  # seconds
    speaker: str | None = None  # always coerced to str, e.g. "0", "A", "speaker_0" -> pick one convention
    confidence: Optional[float] = None
    words: List[Word] = Field(default_factory=list)  # empty if provider has no word-level data
    raw: Optional[dict] = None


class Transcript(BaseModel):
    """Full normalized transcript for one audio file, regardless of provider."""
    provider: Literal["gladia", "whisper", "assemblyai", "deepgram", "elevenlabs"]
    language: Optional[str] = None
    source_file: Optional[str] = None
    utterances: List[Utterance] = Field(default_factory=list)
    duration: Optional[float] = None
    raw: Optional[dict] = None  # full original provider response, if you want to keep it

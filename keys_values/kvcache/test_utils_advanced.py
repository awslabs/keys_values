# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
import os
from typing import Optional, List

from tokenizers import Tokenizer
import torch
from transformers import AutoTokenizer

from keys_values.attention import (
    DefaultKeysAndValues,
)
from keys_values.kvcache.buffers import DefaultKVCacheBuffers
from keys_values.kvcache.gradient.accumulate import GradientAccumulator
from keys_values.kvcache.gradient.checkpoints import KVCacheBufferCheckpoints


def exchange_kv_cache_checkpoints(
    accumulator: GradientAccumulator,
    device: Optional[torch.device] = None,
):
    """
    Ensures that `accumulator._kv_cache_checkpoints` are of testing type
    :class:`KVCacheBufferTestingCheckpoints`. These do not quantize checkpoints,
    which simplifies gradient testing a lot.

    """

    def wrapped_create_checkpoints_and_buffers(
        orig_func,
        model_part,
    ):
        cache_buffers, checkpoints = orig_func(model_part)
        # Need to replace checkpoints
        chunk_numbers = checkpoints[0].chunk_numbers
        checkpoints = [
            KVCacheBufferTestingCheckpoints(
                chunk_numbers=chunk_numbers,
                device=device,
            )
            for _ in range(len(checkpoints))
        ]
        return cache_buffers, checkpoints

    accumulator._create_checkpoints_and_buffers = partial(
        wrapped_create_checkpoints_and_buffers,
        accumulator._create_checkpoints_and_buffers,
    )


class KVCacheBufferTestingCheckpoints(KVCacheBufferCheckpoints):
    """
    Checkpointing class used for testing. The checkpoints are not quantized,
    but the buffers are stored as they are. This is not recommended in
    practice, but simplifies gradient testing. Also, we do not reserve
    memory for checkpoints up front, but copy them as they come in.

    """

    def __init__(
        self,
        chunk_numbers: List[int],
        device: Optional[torch.device] = None,
    ):
        super().__init__(chunk_numbers)
        self._checkpoints: List[Optional[DefaultKeysAndValues]] = [None] * len(
            chunk_numbers
        )
        if device is None:
            device = torch.get_default_device()
        self.device = device

    def _set_checkpoint(
        self,
        pos: int,
        buffers: DefaultKVCacheBuffers,
    ) -> int:
        k_and_v = buffers.get_keys_values()
        self._checkpoints[pos] = DefaultKeysAndValues(
            keys=k_and_v.keys().to(device=self.device, copy=True),
            values=k_and_v.values().to(device=self.device, copy=True),
        )
        return pos

    def _get_checkpoint(
        self,
        pos: int,
        out: DefaultKVCacheBuffers,
    ):
        checkpoint = self._checkpoints[pos]
        if checkpoint is None:
            raise ValueError(
                f"checkpoint at pos={pos} is still empty. Use 'set_checkpoint'"
            )
        out.prefill_from_keys_values(checkpoint)


def load_tokenizer(
    model_name="prajjwal1/bert-tiny", cache_dir=None,
) -> Tokenizer:
    """
    Load a subword tokenizer from Hugging Face Hub.

    Args:
        model_name: Name of the model on HF Hub. Default is a tiny BERT model
            that uses WordPiece tokenization and requires minimal disk space.
        cache_dir: Directory to cache subword tokenizer model

    Returns:
        A HuggingFace tokenizer instance

    Examples of other compact options:
        - "prajjwal1/bert-tiny" (~17MB, WordPiece)
        - "google/bert_uncased_L-2_H-128_A-2" (~12MB, WordPiece)
        - "distilbert-base-uncased" (~232MB, WordPiece)
        - "microsoft/xtremedistil-l6-h256-uncased" (~85MB, WordPiece)

    """
    if cache_dir is None:
        cache_dir = os.getenv("HF_HOME", None)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    return tokenizer


SEPARATORS = (" ", ", ", ". ", ": ", "; ")


def sequence_of_words(num_words: int) -> str:
    """
    Generate a sequence of random English words. Two words are separated by one
    of :const:`SEPARATORS`.

    Args:
        num_words: Number of words to generate

    Returns:
        A string containing num_words random English words separated by space
        and punctuation as detailed above.

    """
    # Common English words corpus (~1000 words)
    # Using a combination of most frequent English words
    words = [
        # Articles, pronouns, conjunctions
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their",

        # Common verbs
        "make", "get", "go", "know", "take", "see", "come", "think", "look", "want",
        "give", "use", "find", "tell", "ask", "work", "seem", "feel", "try", "leave",
        "call", "put", "mean", "keep", "let", "begin", "show", "hear", "play", "run",
        "move", "like", "live", "believe", "hold", "bring", "happen", "write", "sit",
        "stand", "lose", "pay", "meet", "include", "continue", "set", "learn", "change",
        "lead", "understand", "watch", "follow", "stop", "create", "speak", "read",
        "allow", "add", "spend", "grow", "open", "walk", "win", "offer", "remember",
        "love", "consider", "appear", "buy", "wait", "serve", "die", "send", "expect",
        "build", "stay", "fall", "cut", "reach", "kill", "remain", "suggest", "raise",
        "pass", "sell", "require", "report", "decide", "pull", "break", "throw", "return",

        # Common nouns
        "time", "person", "year", "way", "day", "thing", "man", "world", "life", "hand",
        "part", "child", "eye", "woman", "place", "work", "week", "case", "point", "company",
        "number", "group", "problem", "fact", "water", "room", "money", "story", "home", "night",
        "study", "book", "word", "business", "issue", "side", "kind", "head", "house", "service",
        "friend", "father", "power", "hour", "game", "line", "end", "member", "law", "car",
        "city", "community", "name", "president", "team", "minute", "idea", "kid", "body", "information",
        "back", "parent", "face", "others", "level", "office", "door", "health", "art", "war",
        "history", "party", "result", "change", "morning", "reason", "research", "girl", "guy", "moment",
        "air", "teacher", "force", "education", "food", "land", "family", "mother", "area", "student",
        "street", "computer", "question", "school", "state", "music", "program", "system", "paper", "government",

        # Common adjectives
        "good", "new", "first", "last", "long", "great", "little", "own", "other", "old",
        "right", "big", "high", "different", "small", "large", "next", "early", "young", "important",
        "few", "public", "bad", "same", "able", "full", "hot", "cold", "hard", "real",
        "best", "better", "true", "free", "strong", "political", "possible", "whole", "white", "black",
        "social", "human", "sure", "low", "red", "difficult", "clear", "blue", "dark", "green",
        "easy", "late", "happy", "alone", "certain", "short", "nice", "fine", "beautiful", "special",

        # Additional common words to reach ~1000
        "country", "american", "national", "economic", "medical", "poor", "natural", "dead", "central",
        "happy", "serious", "ready", "simple", "left", "physical", "general", "environmental", "financial",
        "blue", "democratic", "dark", "various", "entire", "close", "legal", "religious", "cold", "final",
        "main", "green", "nice", "huge", "popular", "traditional", "cultural", "church", "police", "original",

        "nation", "century", "town", "rate", "field", "difference", "light", "control", "court", "table",
        "player", "century", "technology", "economy", "agreement", "effort", "hotel", "space", "science",
        "union", "college", "south", "north", "west", "east", "building", "election", "market", "plan",
        "wife", "husband", "month", "product", "society", "activity", "character", "ground", "film", "interest",
        "hospital", "rock", "fire", "window", "relationship", "knowledge", "quality", "hospital", "phone", "wall",
        "oil", "outside", "station", "cost", "theory", "military", "army", "gun", "culture", "organization",
        "practice", "value", "trade", "church", "movement", "fear", "voice", "season", "blood", "thought",
        "direction", "peace", "energy", "defense", "security", "radio", "opportunity", "trial", "administration",

        "receive", "produce", "apply", "claim", "support", "reduce", "ensure", "protect", "involve", "manage",
        "improve", "establish", "explain", "identify", "develop", "perform", "recognize", "remain", "reflect",
        "maintain", "indicate", "describe", "avoid", "refuse", "gain", "arrive", "admit", "forget", "miss",
        "discover", "travel", "express", "announce", "assume", "prepare", "design", "enjoy", "encourage",
        "realize", "achieve", "increase", "compare", "discuss", "imagine", "suffer", "replace", "fight",

        "south", "certainly", "court", "everyone", "forward", "north", "thus", "film", "radio", "western",
        "spring", "summer", "fall", "winter", "animal", "species", "nature", "disease", "cancer", "doctor",
        "patient", "heart", "drug", "treatment", "trial", "pain", "test", "medicine", "hospital", "blood",
        "cell", "risk", "mental", "brain", "physical", "diet", "exercise", "stress", "sleep", "weight",

        "mountain", "river", "ocean", "forest", "tree", "plant", "flower", "bird", "fish", "horse",
        "dog", "cat", "island", "beach", "stone", "wood", "metal", "glass", "paper", "plastic",
        "wind", "rain", "snow", "sun", "moon", "star", "sky", "cloud", "weather", "temperature",

        "letter", "sound", "picture", "color", "shape", "size", "weight", "length", "width", "height",
        "distance", "speed", "age", "date", "yesterday", "today", "tomorrow", "second", "minute", "hour",
        "clock", "watch", "calendar", "holiday", "birthday", "anniversary", "celebration", "gift", "party",

        "language", "english", "french", "spanish", "german", "italian", "chinese", "japanese", "russian",
        "sentence", "grammar", "vocabulary", "pronunciation", "accent", "translation", "dictionary", "alphabet",

        "number", "count", "amount", "total", "sum", "average", "percent", "half", "quarter", "double",
        "triple", "dozen", "hundred", "thousand", "million", "billion", "first", "second", "third", "fourth",

        "restaurant", "menu", "food", "meal", "breakfast", "lunch", "dinner", "drink", "coffee", "tea",
        "milk", "juice", "beer", "wine", "bread", "meat", "fish", "chicken", "vegetable", "fruit",
        "apple", "orange", "banana", "potato", "tomato", "cheese", "egg", "rice", "sugar", "salt",

        "clothes", "shirt", "pants", "dress", "shoe", "hat", "coat", "jacket", "suit", "tie",
        "color", "size", "style", "fashion", "wear", "buy", "shop", "store", "price", "sale",

        "sport", "football", "basketball", "baseball", "tennis", "golf", "swimming", "running", "game",
        "match", "championship", "victory", "defeat", "score", "goal", "competition", "athlete", "coach",

        "computer", "internet", "website", "email", "software", "hardware", "program", "data", "file",
        "screen", "keyboard", "mouse", "printer", "network", "server", "database", "security", "password",
        "technology", "digital", "online", "click", "download", "upload", "search", "browse", "connect",

        "phone", "call", "message", "text", "answer", "ring", "mobile", "smartphone", "application",
        "camera", "photo", "picture", "video", "record", "battery", "charge", "screen", "button",

        "book", "page", "chapter", "paragraph", "sentence", "author", "writer", "novel", "story",
        "title", "reading", "library", "bookstore", "publisher", "magazine", "newspaper", "article",

        "movie", "film", "actor", "actress", "director", "scene", "screen", "theater", "cinema",
        "ticket", "show", "performance", "entertainment", "audience", "review", "rating", "comedy",
        "drama", "action", "horror", "documentary", "animation", "character", "plot", "ending",

        "music", "song", "singer", "band", "concert", "album", "track", "melody", "rhythm", "beat",
        "instrument", "guitar", "piano", "drum", "violin", "sound", "listen", "play", "record",

        "art", "artist", "painting", "drawing", "sculpture", "museum", "gallery", "exhibition",
        "creative", "design", "style", "modern", "classic", "contemporary", "abstract", "portrait",

        "travel", "trip", "journey", "vacation", "holiday", "tourist", "destination", "flight",
        "airport", "plane", "train", "bus", "taxi", "ticket", "reservation", "hotel", "room",
        "luggage", "passport", "visa", "border", "customs", "arrive", "depart", "visit", "explore",

        "business", "company", "corporation", "enterprise", "firm", "industry", "market", "customer",
        "client", "service", "product", "brand", "sale", "purchase", "profit", "loss", "revenue",
        "investment", "finance", "budget", "expense", "income", "salary", "wage", "employee", "employer",
        "manager", "director", "executive", "staff", "department", "office", "meeting", "presentation",

        "education", "school", "university", "college", "student", "teacher", "professor", "lesson",
        "course", "class", "subject", "exam", "test", "grade", "degree", "diploma", "certificate",
        "homework", "assignment", "project", "research", "study", "learn", "teach", "knowledge",

        "law", "legal", "court", "judge", "lawyer", "attorney", "trial", "case", "crime", "criminal",
        "police", "officer", "arrest", "prison", "jail", "guilty", "innocent", "justice", "rights",
        "freedom", "constitution", "government", "politics", "political", "democracy", "election",

        "science", "scientific", "research", "study", "experiment", "theory", "hypothesis", "result",
        "discovery", "invention", "technology", "innovation", "development", "progress", "advance",
        "biology", "chemistry", "physics", "mathematics", "astronomy", "geology", "psychology",

        "environment", "nature", "natural", "earth", "planet", "climate", "weather", "temperature",
        "pollution", "waste", "recycle", "conservation", "protection", "sustainable", "renewable",
        "energy", "solar", "wind", "nuclear", "fossil", "fuel", "carbon", "emission", "greenhouse",

        "health", "medical", "medicine", "doctor", "nurse", "patient", "hospital", "clinic", "treatment",
        "surgery", "operation", "diagnosis", "symptom", "disease", "illness", "infection", "virus",
        "bacteria", "vaccine", "drug", "medication", "prescription", "therapy", "recovery", "cure",

        "family", "parent", "father", "mother", "son", "daughter", "brother", "sister", "husband",
        "wife", "child", "baby", "grandfather", "grandmother", "uncle", "aunt", "cousin", "nephew",
        "niece", "relative", "relationship", "marriage", "wedding", "divorce", "birth", "death",

        "emotion", "feeling", "happy", "sad", "angry", "fear", "love", "hate", "joy", "sorrow",
        "excited", "nervous", "calm", "peaceful", "worried", "anxious", "proud", "ashamed", "grateful",
        "surprised", "shocked", "confused", "interested", "bored", "tired", "energetic", "relaxed",

        "quality", "excellent", "good", "bad", "poor", "superior", "inferior", "perfect", "imperfect",
        "strong", "weak", "hard", "soft", "heavy", "light", "thick", "thin", "wide", "narrow",
        "deep", "shallow", "high", "low", "tall", "short", "long", "brief", "fast", "slow",

        "quantity", "many", "few", "much", "little", "some", "any", "all", "none", "several",
        "numerous", "multiple", "single", "double", "triple", "enough", "plenty", "excess", "shortage",

        "position", "location", "place", "site", "spot", "point", "area", "region", "zone", "district",
        "neighborhood", "center", "middle", "edge", "corner", "side", "top", "bottom", "front", "back",
        "inside", "outside", "above", "below", "over", "under", "between", "among", "near", "far",

        "direction", "north", "south", "east", "west", "left", "right", "forward", "backward",
        "upward", "downward", "straight", "curve", "turn", "rotate", "spin", "around", "across",

        "time", "moment", "instant", "period", "duration", "interval", "past", "present", "future",
        "history", "modern", "ancient", "recent", "current", "former", "previous", "next", "following",
        "early", "late", "beginning", "middle", "end", "start", "finish", "continue", "pause", "stop",

        "action", "activity", "operation", "process", "procedure", "method", "technique", "approach",
        "strategy", "plan", "scheme", "project", "task", "job", "work", "duty", "function", "role",
        "responsibility", "obligation", "commitment", "promise", "agreement", "contract", "deal",

        "communication", "language", "speech", "talk", "conversation", "discussion", "dialogue", "debate",
        "argument", "statement", "comment", "remark", "observation", "opinion", "view", "perspective",
        "attitude", "belief", "thought", "idea", "concept", "notion", "understanding", "meaning",
    ]

    total_num_words = len(words)
    ind1 = torch.randint(total_num_words, (num_words,)).tolist()
    ind2 = torch.randint(len(SEPARATORS), (num_words,)).tolist()
    seq_words = [
        x
        for tup in zip(
            [words[i] for i in ind1],
            [SEPARATORS[i] for i in ind2],
        )
        for x in tup
    ]
    return "".join(seq_words[:-1])


if __name__ == "__main__":
    seed = 31415927
    torch.random.manual_seed(seed)
    for _ in range(5):
        print("\n" + sequence_of_words(20))

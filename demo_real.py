"""
VectorTrainer Real VectorWave Demo — 실제 OpenAI API + Weaviate 연동.

4개의 실무 AI 함수를 @vectorize 데코레이터로 감싸고,
다양한 입력으로 실행 로그를 Weaviate에 적재한 뒤
VectorTrainer 파이프라인(추출→합성→학습)을 실행합니다.

사전 조건:
  - Weaviate 컨테이너 실행 (localhost:8080)
  - .env 파일에 OPENAI_API_KEY 설정
  - pip install -e ".[dev]"

실행:
  python3 demo_real.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

load_dotenv()

console = Console()

# ---------------------------------------------------------------------------
# OpenAI client (직접 호출)
# ---------------------------------------------------------------------------

from openai import OpenAI

_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


# ---------------------------------------------------------------------------
# 실무 AI 함수 4종 (OpenAI API 호출)
# ---------------------------------------------------------------------------


def generate_review_summary(review_text: str) -> str:
    """제품 리뷰를 구조화된 요약으로 변환."""
    client = get_openai_client()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a product review summarizer. "
                    "Return a structured summary: sentiment (POSITIVE/NEGATIVE/NEUTRAL), "
                    "key features mentioned, and a rating out of 5. "
                    "Format: SENTIMENT | [features] | Rating: X/5"
                ),
            },
            {"role": "user", "content": review_text},
        ],
        temperature=0.3,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


def extract_sentiment(review_text: str) -> str:
    """리뷰에서 감정 분석 결과를 JSON으로 추출."""
    client = get_openai_client()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a sentiment analysis engine. "
                    "Extract emotions from the review and return JSON: "
                    '{"overall": "positive/negative/neutral", '
                    '"confidence": 0.0-1.0, '
                    '"emotions": {"joy": 0-100, "anger": 0-100, "sadness": 0-100, "surprise": 0-100}}'
                ),
            },
            {"role": "user", "content": review_text},
        ],
        temperature=0.2,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


def generate_product_recommendation(user_preferences: str, product_category: str) -> str:
    """사용자 선호도 기반 제품 추천 생성."""
    client = get_openai_client()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a product recommendation engine. "
                    "Given user preferences and category, recommend 3 products. "
                    "Format each as: Product Name | Reason | $Price range"
                ),
            },
            {
                "role": "user",
                "content": f"Preferences: {user_preferences}\nCategory: {product_category}",
            },
        ],
        temperature=0.4,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def translate_review(review_text: str, target_language: str) -> str:
    """리뷰를 번역하면서 톤과 뉘앙스를 보존."""
    client = get_openai_client()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    f"Translate the following product review to {target_language}. "
                    "Preserve the tone, sentiment, and any sarcasm. "
                    "Prefix with [TONE: positive/negative/neutral/sarcastic]."
                ),
            },
            {"role": "user", "content": review_text},
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 입력 데이터 세트
# ---------------------------------------------------------------------------

REVIEWS = [
    "This laptop is absolutely incredible! The M3 chip handles everything I throw at it. Best purchase this year.",
    "Great value smartphone. Camera quality is good for the price. Battery lasts all day.",
    "Love this wireless earbuds! Amazing noise cancellation and the sound quality is crystal clear.",
    "The build quality of this monitor is excellent. 4K resolution makes everything look stunning.",
    "Best mechanical keyboard I've ever used. The switches feel amazing and RGB lighting is gorgeous.",
    "This tablet is perfect for digital art. The pen responsiveness is unmatched. Highly recommend.",
    "Good router but setup was confusing. Once configured, WiFi speed is fast and reliable.",
    "The product arrived broken. Terrible packaging and poor quality control. Very disappointing.",
    "Average gaming mouse. Nothing special about the sensor. Overpriced for what you get.",
    "Worst headphones ever. Sound quality is terrible and they broke after a week. Do not buy.",
    "Innovative smart home hub that actually works seamlessly with all my devices. Future of home automation.",
    "The specs say waterproof but it stopped working after a light rain. False advertising at its worst.",
    "Decent external SSD. Transfer speeds match the claimed numbers. Good for backup purposes.",
    "This drone's camera stabilization is revolutionary. Professional-quality aerial footage for hobbyists.",
    "Simple USB hub that just works. Nothing fancy but reliable. Gets the job done.",
    "The new fitness tracker has incredible health monitoring. Sleep tracking changed my habits completely.",
    "Portable charger with amazing capacity. Charges my phone 5 times. Essential travel companion.",
    "This smart watch looks premium but the software is buggy. Needs several updates to be usable.",
    "The VR headset completely exceeded my expectations. Immersive experience that's hard to put down.",
    "Cheap Bluetooth speaker with surprisingly good bass. Perfect for outdoor gatherings.",
]

RECOMMENDATION_INPUTS = [
    ("software engineer, needs portability and power", "laptop"),
    ("student on a budget, social media use", "smartphone"),
    ("audiophile, lossless music, long commute", "headphones"),
    ("graphic designer, color accuracy essential", "monitor"),
    ("competitive gamer, low latency priority", "keyboard"),
    ("digital artist, pressure sensitivity needed", "tablet"),
    ("remote worker, video calls all day", "webcam"),
    ("photographer, large raw files", "external storage"),
    ("fitness enthusiast, outdoor runner", "smartwatch"),
    ("parent, kid-friendly content", "tablet"),
    ("eco-conscious buyer, sustainable materials", "laptop"),
    ("music producer, studio quality", "headphones"),
    ("small business owner, POS system needed", "tablet"),
    ("traveler, lightweight and durable", "laptop"),
    ("streamer, high quality microphone needed", "audio equipment"),
]

TRANSLATION_INPUTS = [
    ("This product changed my life! Absolutely perfect.", "Korean"),
    ("Terrible experience. The product broke on day one.", "Japanese"),
    ("It's okay I guess. Nothing to write home about.", "Spanish"),
    ("Oh sure, this 'premium' product is worth every penny... NOT.", "French"),
    ("Best purchase I've made in years. Highly recommended!", "German"),
    ("Decent product with room for improvement.", "Chinese"),
    ("This is exactly what I needed. Works like a charm.", "Korean"),
    ("Would not recommend. Save your money.", "Japanese"),
    ("Five stars! Exceeded all my expectations!", "Portuguese"),
    ("The quality is questionable but the price is right.", "Italian"),
    ("Revolutionary design meets practical functionality.", "Korean"),
    ("I've been using this for months and it's still perfect.", "Spanish"),
    ("Complete waste of money. Returning immediately.", "German"),
    ("Solid build quality. You get what you pay for.", "French"),
    ("This product is a game changer for productivity.", "Korean"),
]


# ---------------------------------------------------------------------------
# 피드백 쌍 (합성용)
# ---------------------------------------------------------------------------

FEEDBACK_PAIRS = [
    # generate_review_summary
    {
        "function": "generate_review_summary",
        "input_prompt": "Summarize this product review",
        "bad": "Good product.",
        "fixed": "POSITIVE | [build quality, performance, value] | Rating: 4/5 — The product delivers solid performance with good build quality at a reasonable price point.",
    },
    {
        "function": "generate_review_summary",
        "input_prompt": "Summarize this product review",
        "bad": "The product has some good and bad points.",
        "fixed": "NEUTRAL | [camera quality(+), battery(-), design(+)] | Rating: 3/5 — Mixed review highlighting strong camera quality and design but noting battery life concerns.",
    },
    {
        "function": "generate_review_summary",
        "input_prompt": "Summarize this product review",
        "bad": "Not recommended.",
        "fixed": "NEGATIVE | [quality control, durability, customer service] | Rating: 1/5 — Severe quality issues reported with product breaking prematurely and unhelpful customer support.",
    },
    # extract_sentiment
    {
        "function": "extract_sentiment",
        "input_prompt": "Extract sentiment from this review",
        "bad": '{"sentiment": "good"}',
        "fixed": '{"overall": "positive", "confidence": 0.92, "emotions": {"joy": 85, "anger": 0, "sadness": 0, "surprise": 45}}',
    },
    {
        "function": "extract_sentiment",
        "input_prompt": "Extract sentiment from sarcastic review",
        "bad": '{"overall": "positive", "confidence": 0.8}',
        "fixed": '{"overall": "negative", "confidence": 0.75, "emotions": {"joy": 5, "anger": 60, "sadness": 30, "surprise": 20}, "sarcasm_detected": true}',
    },
    # generate_product_recommendation
    {
        "function": "generate_product_recommendation",
        "input_prompt": "Recommend products for a student",
        "bad": "Buy a MacBook.",
        "fixed": "1) Acer Aspire 5 | Budget-friendly with good performance for schoolwork | $400-500\n2) Lenovo IdeaPad 3 | Reliable with great keyboard for note-taking | $350-450\n3) HP Chromebook 14 | Lightweight, long battery, perfect for web-based study | $250-350",
    },
    {
        "function": "generate_product_recommendation",
        "input_prompt": "Recommend headphones for audiophile",
        "bad": "Get AirPods.",
        "fixed": "1) Sennheiser HD 660S2 | Reference-grade open-back with natural soundstage | $400-500\n2) Hifiman Sundara | Planar magnetic with exceptional detail retrieval | $300-350\n3) Beyerdynamic DT 1990 Pro | Studio monitoring with analytical precision | $400-450",
    },
    # translate_review
    {
        "function": "translate_review",
        "input_prompt": "Translate review preserving sarcasm",
        "bad": "이 제품은 좋습니다.",
        "fixed": '[TONE: sarcastic] 아, 네~ 이 "프리미엄" 제품은 정말 돈 값을 하네요... 아닌데요.',
    },
    {
        "function": "translate_review",
        "input_prompt": "Translate review to Korean",
        "bad": "이 제품이 좋습니다.",
        "fixed": "[TONE: positive] 이 제품이 제 인생을 바꿨어요! 완벽 그 자체입니다.",
    },
]


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


def main():
    from vector_trainer.extractor import DensityBasedExtractor
    from vector_trainer.synthesizer import FeedbackDiffAnalyzer, run_synthesis_pipeline
    from vector_trainer.pipeline import OpenAITrainer
    from vector_trainer.dashboard import CLIDashboard
    from vector_trainer.types import FeedbackPair
    from vector_trainer.cost_guard import check_budget, BudgetExceededError

    # ===== 환경 확인 =====
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]OPENAI_API_KEY가 .env에 설정되지 않았습니다.[/red]")
        sys.exit(1)

    console.print(Panel(
        "[bold]VectorTrainer Real Demo[/bold]\n"
        "실제 OpenAI API + VectorWave + Weaviate 연동 데모\n\n"
        "4개 AI 함수 × 다양한 입력 → 실행 로그 적재 → 골든 데이터 추출 → 파인튜닝",
        title="VectorTrainer v0.1.0",
        border_style="blue",
    ))

    output_dir = tempfile.mkdtemp(prefix="vectortrainer_real_")
    console.print(f"\n[dim]Output directory: {output_dir}[/dim]\n")

    # ===== Phase 1: VectorWave 초기화 & 실행 로그 적재 =====
    console.print("[bold blue]== Phase 1: VectorWave 초기화 & 실행 로그 적재 ==[/bold blue]\n")

    from vectorwave import initialize_database, vectorize

    client = initialize_database()
    if not client:
        console.print("[red]Weaviate 연결 실패. docker compose up -d 를 먼저 실행하세요.[/red]")
        sys.exit(1)
    console.print("  [green]Weaviate 연결 성공[/green]\n")

    # @vectorize 데코레이터로 함수 래핑
    vw_summary = vectorize(capture_return_value=True, capture_inputs=True)(generate_review_summary)
    vw_sentiment = vectorize(capture_return_value=True, capture_inputs=True)(extract_sentiment)
    vw_recommend = vectorize(capture_return_value=True, capture_inputs=True)(generate_product_recommendation)
    vw_translate = vectorize(capture_return_value=True, capture_inputs=True)(translate_review)

    total_calls = len(REVIEWS) * 2 + len(RECOMMENDATION_INPUTS) + len(TRANSLATION_INPUTS)
    console.print(f"  총 {total_calls}회 API 호출 예정\n")

    with Progress(console=console) as progress:
        task = progress.add_task("  실행 로그 적재", total=total_calls)

        # generate_review_summary — 20 reviews
        for review in REVIEWS:
            try:
                result = vw_summary(review_text=review)
                progress.update(task, advance=1, description=f"  [summary] {result[:50]}...")
            except Exception as e:
                progress.update(task, advance=1, description=f"  [summary] ERROR: {e}")

        # extract_sentiment — 20 reviews
        for review in REVIEWS:
            try:
                result = vw_sentiment(review_text=review)
                progress.update(task, advance=1, description=f"  [sentiment] {result[:50]}...")
            except Exception as e:
                progress.update(task, advance=1, description=f"  [sentiment] ERROR: {e}")

        # generate_product_recommendation — 15
        for prefs, cat in RECOMMENDATION_INPUTS:
            try:
                result = vw_recommend(user_preferences=prefs, product_category=cat)
                progress.update(task, advance=1, description=f"  [recommend] {result[:50]}...")
            except Exception as e:
                progress.update(task, advance=1, description=f"  [recommend] ERROR: {e}")

        # translate_review — 15
        for review_text, lang in TRANSLATION_INPUTS:
            try:
                result = vw_translate(review_text=review_text, target_language=lang)
                progress.update(task, advance=1, description=f"  [translate] {result[:50]}...")
            except Exception as e:
                progress.update(task, advance=1, description=f"  [translate] ERROR: {e}")

    console.print(f"\n  비동기 인덱싱 대기 중 (5초)...")
    time.sleep(5)

    # 로그 확인
    from vectorwave import search_executions
    for func_name in ["generate_review_summary", "extract_sentiment",
                       "generate_product_recommendation", "translate_review"]:
        try:
            logs = search_executions(limit=50, filters={"function_name": func_name})
            console.print(f"  [green]{func_name}: {len(logs)} 로그[/green]")
        except Exception:
            console.print(f"  [yellow]{func_name}: 로그 조회 실패[/yellow]")

    # ===== Phase 2: 골든 데이터 추출 =====
    console.print("\n[bold cyan]== Phase 2: 골든 데이터 추출 ==[/bold cyan]\n")

    from vectorwave import VectorWaveDatasetManager
    dataset_manager = VectorWaveDatasetManager()
    console.print("  [green]VectorWaveDatasetManager 연결 성공[/green]\n")

    all_jsonl_paths = []
    for func_name in ["generate_review_summary", "extract_sentiment",
                       "generate_product_recommendation", "translate_review"]:
        extractor = DensityBasedExtractor(
            dataset_manager=dataset_manager,
            epsilon=0.3,
            top_k=20,
        )

        steady = extractor.select_steady_state(func_name)
        anomalies = extractor.select_anomalies(func_name)
        console.print(f"  {func_name}: steady={len(steady)}, anomaly={len(anomalies)}")

        jsonl_path = os.path.join(output_dir, f"{func_name}_golden.jsonl")
        extractor.extract_golden_data(func_name, jsonl_path)
        all_jsonl_paths.append(jsonl_path)

        with open(jsonl_path) as f:
            line_count = sum(1 for _ in f)
        console.print(f"    -> {line_count} records -> {jsonl_path}")

    # 통합 JSONL
    merged_path = os.path.join(output_dir, "merged_golden.jsonl")
    with open(merged_path, "w", encoding="utf-8") as out_f:
        for p in all_jsonl_paths:
            with open(p, "r", encoding="utf-8") as in_f:
                out_f.write(in_f.read())

    with open(merged_path) as f:
        total_records = sum(1 for _ in f)
    console.print(f"\n  [green]통합 JSONL: {total_records} records -> {merged_path}[/green]")

    # ===== Phase 3: 피드백 합성 =====
    console.print("\n[bold magenta]== Phase 3: 피드백 합성 ==[/bold magenta]\n")

    feedback_pairs = [
        FeedbackPair(
            input_prompt=fp["input_prompt"],
            bad_output=fp["bad"],
            fixed_output=fp["fixed"],
            context={"function": fp["function"]},
        )
        for fp in FEEDBACK_PAIRS
    ]

    analyzer = FeedbackDiffAnalyzer()
    for i, pair in enumerate(feedback_pairs):
        diff = analyzer.analyze(pair)
        console.print(
            f"  Pair {i+1}: edit_distance={diff['edit_distance']}, "
            f"similarity={diff['similarity_score']:.3f} ({pair.context.get('function', '')})"
        )

    hook_path = run_synthesis_pipeline(pairs=feedback_pairs, output_dir=output_dir)
    console.print(f"\n  [green]Hook Script: {hook_path}[/green]")

    # ===== Phase 4: 비용 검증 =====
    console.print("\n[bold yellow]== Phase 4: 비용 검증 (Cost Guard) ==[/bold yellow]\n")

    from vector_trainer.cost_guard import count_tokens_in_jsonl, estimate_training_cost
    token_count = count_tokens_in_jsonl(merged_path, "gpt-4o-mini-2024-07-18")
    cost_1 = estimate_training_cost(token_count, "gpt-4o-mini-2024-07-18", n_epochs=1)
    cost_3 = estimate_training_cost(token_count, "gpt-4o-mini-2024-07-18", n_epochs=3)

    console.print(f"  토큰 수: {token_count:,}")
    console.print(f"  예상 비용 (1 epoch): ${cost_1:.6f}")
    console.print(f"  예상 비용 (3 epochs): ${cost_3:.6f}")

    try:
        check_budget(merged_path, "gpt-4o-mini-2024-07-18", max_budget_usd=10.0, n_epochs=3)
        console.print("  [green]예산 검증 통과 ($10.00 이내)[/green]")
    except BudgetExceededError as e:
        console.print(f"  [red]예산 초과: {e}[/red]")

    # ===== Phase 5: 결과 요약 =====
    console.print("\n[bold]== 최종 결과 ==[/bold]\n")
    console.print(f"  출력 디렉토리:    {output_dir}")
    console.print(f"  통합 JSONL:       {merged_path} ({total_records} records)")
    console.print(f"  Hook Script:      {hook_path}")
    console.print(f"  토큰 수:          {token_count:,}")
    console.print(f"  예상 비용 (3ep):  ${cost_3:.6f}")

    # 개별 JSONL 경로 출력
    console.print(f"\n  [dim]개별 골든 데이터:[/dim]")
    for p in all_jsonl_paths:
        console.print(f"    {p}")

    console.print()


if __name__ == "__main__":
    main()

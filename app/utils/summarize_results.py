from collections import defaultdict, Counter

def summarize_results(results):
    count = len(results)

    skor_counter = defaultdict(Counter)
    skor_total = defaultdict(int)
    skor_min = defaultdict(lambda: float('inf'))
    skor_max = defaultdict(lambda: float('-inf'))

    sudut_data = defaultdict(float)
    sudut_count = defaultdict(int)

    feedback_counter = Counter()
    rula_summary_counter = Counter()
    reba_summary_counter = Counter()

    for res in results:
        # Hitung skor RULA/REBA
        for key, value in res.items():
            if key.endswith("_score"):
                skor_counter[key][value] += 1
                skor_total[key] += value
                skor_min[key] = min(skor_min[key], value)
                skor_max[key] = max(skor_max[key], value)

        # Hitung sudut
        for sudut_key, sudut_val in res.get("details", {}).items():
            if sudut_key.startswith("sudut_"):
                sudut_data[sudut_key] += sudut_val
                sudut_count[sudut_key] += 1

        # Hitung feedback
        feedback_text = res.get("feedback", "")
        if feedback_text:
            feedback_counter[feedback_text] += 1

        # Hitung summary
        summary_data = res.get("summary", {})
        if "rula_summary" in summary_data:
            rula_summary_counter[summary_data["rula_summary"]] += 1
        if "reba_summary" in summary_data:
            reba_summary_counter[summary_data["reba_summary"]] += 1

    # Hitung statistik skor
    majority_scores = {k: v.most_common(1)[0][0] for k, v in skor_counter.items()}
    average_scores = {k: skor_total[k] / count for k in skor_counter}
    min_scores = {k: skor_min[k] for k in skor_counter}
    max_scores = {k: skor_max[k] for k in skor_counter}

    # Hitung rata-rata sudut
    average_sudut = {k: sudut_data[k] / sudut_count[k] for k in sudut_data}

    # Feedback dan summary paling umum
    feedback_summary = feedback_counter.most_common(1)[0][0] if feedback_counter else "-"
    rula_summary = rula_summary_counter.most_common(1)[0][0] if rula_summary_counter else "-"
    reba_summary = reba_summary_counter.most_common(1)[0][0] if reba_summary_counter else "-"

    # Susun ringkasan teks
    summary_feedback = (
        f"Analisa dari {count} frame:\n\n"
        f"Ringkasan RULA: {rula_summary}\n"
        f"Ringkasan REBA: {reba_summary}\n\n"
        f"Mayoritas Skor:\n" + "\n".join([f"- {k}: {v}" for k, v in majority_scores.items()]) + "\n\n"
        f"Skor Minimum:\n" + "\n".join([f"- {k}: {min_scores[k]}" for k in min_scores]) + "\n\n"
        f"Skor Maksimum:\n" + "\n".join([f"- {k}: {max_scores[k]}" for k in max_scores]) + "\n\n"
        f"Rata-rata Skor:\n" + "\n".join([f"- {k}: {average_scores[k]:.2f}" for k in average_scores]) + "\n\n"
        f"Rata-rata Sudut Tubuh:\n" + "\n".join([f"- {k}: {average_sudut[k]:.2f}" for k in average_sudut]) + "\n\n"
        f"Feedback Terbanyak:\n{feedback_summary}"
    )

    return {
        "feedback": summary_feedback,
        "majority_scores": majority_scores,
        "average_scores": average_scores,
        "min_scores": min_scores,
        "max_scores": max_scores,
        "average_sudut": average_sudut,
        "total_frames": count,
        "most_common_feedback": feedback_summary,
        "rula_summary": rula_summary,
        "reba_summary": reba_summary
    }

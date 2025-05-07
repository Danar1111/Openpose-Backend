from collections import defaultdict, Counter

def summarize_results(results):
    count = len(results)
    
    skor_counter = defaultdict(Counter)
    sudut_data = defaultdict(float)
    sudut_count = defaultdict(int)
    feedback_counter = Counter()

    for res in results:
        for key, value in res.items():
            if key.startswith("skor_"):
                skor_counter[key][value] += 1

            elif isinstance(value, dict):  # Sudut
                for sudut_key, sudut_val in value.items():
                    sudut_data[sudut_key] += sudut_val
                    sudut_count[sudut_key] += 1

        feedback_text = res.get("feedback", "")
        if feedback_text:
            feedback_counter[feedback_text] += 1

    # Ambil skor majority
    majority_scores = {}
    for key, counter in skor_counter.items():
        majority_skor = counter.most_common(1)[0][0]
        majority_scores[key] = majority_skor

    # Hitung rata-rata sudut
    average_sudut = {key: sudut_data[key] / sudut_count[key] for key in sudut_data}

    # Feedback paling sering
    most_common_feedback = feedback_counter.most_common(1)
    feedback_summary = "; ".join([f"{text}" for text, freq in most_common_feedback])

    # Susun feedback akhir
    summary_feedback = (
        f"Hasil analisa {count} frame."
        f" Mayoritas skor per bagian:\n" + "\n".join([f"- {key}: {val}" for key, val in majority_scores.items()]) + "\n\n"
        f" Rata-rata Sudut:\n" + "\n".join([f"- {key}: {val:.2f}" for key, val in average_sudut.items()]) + "\n\n"
        f"\n{feedback_summary}"
    )

    return {
        "feedback": summary_feedback,
        "majority_scores": majority_scores,
        "average_sudut": average_sudut,
        "total_frames": count
    }

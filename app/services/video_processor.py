from collections import defaultdict, Counter

def categorize(score):
    if score == 0:
        return "Ergonomis"
    elif score == 1:
        return "Cukup Ergonomis"
    else:
        return "Kurang Ergonomis"

def summarize_results_majority(results):
    count = len(results)
    
    kategori_counter = defaultdict(Counter)
    sudut_data = defaultdict(float)
    sudut_count = defaultdict(int)
    feedback_counter = Counter()

    for res in results:
        # Hitung kategori per skor
        for key, value in res.items():
            if key.startswith("skor_"):
                kategori = categorize(value)
                kategori_counter[key][kategori] += 1

            elif isinstance(value, dict):  # Sudut
                for sudut_key, sudut_val in value.items():
                    sudut_data[sudut_key] += sudut_val
                    sudut_count[sudut_key] += 1

        # Hitung feedback
        feedback_text = res.get("feedback", "")
        if feedback_text:
            feedback_counter[feedback_text] += 1

    # Ambil mayoritas kategori
    majority_result = {}
    for key, counter in kategori_counter.items():
        majority_kategori = counter.most_common(1)[0][0]
        majority_result[key] = majority_kategori

    # Hitung rata-rata sudut
    average_sudut = {key: sudut_data[key] / sudut_count[key] for key in sudut_data}

    # Feedback paling sering
    most_common_feedback = feedback_counter.most_common(3)
    feedback_summary = "; ".join([f"{text} ({freq}x)" for text, freq in most_common_feedback])

    # Buat feedback akhir
    summary_feedback = (
        f"Hasil analisa {count} frame.\n\n"
        f"Mayoritas kategori per bagian:\n" + "\n".join([f"- {key.replace('skor_', '').replace('_', ' ').title()}: {kategori}" for key, kategori in majority_result.items()]) + "\n\n"
        f"Rata-rata Sudut:\n" + "\n".join([f"- {key}: {val:.2f}" for key, val in average_sudut.items()]) + "\n\n"
        f"Feedback paling sering:\n{feedback_summary}"
    )

    return {
        "feedback": summary_feedback,
        "majority_scores": majority_result,
        "average_sudut": average_sudut,
        "feedback_counter": dict(feedback_counter),
        "total_frames": count
    }

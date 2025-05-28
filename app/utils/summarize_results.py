from collections import defaultdict, Counter

def summarize_results(results):
    count = len(results)
    print(results)
    
    skor_counter = defaultdict(Counter)
    skor_total = defaultdict(int)
    skor_min = defaultdict(lambda: float('inf'))
    skor_max = defaultdict(lambda: float('-inf'))

    sudut_data = defaultdict(float)
    sudut_count = defaultdict(int)

    feedback_counter = Counter()

    for res in results:
        for key, value in res.items():
            if key.startswith("skor_"):
                skor_counter[key][value] += 1
                skor_total[key] += value
                skor_min[key] = min(skor_min[key], value)
                skor_max[key] = max(skor_max[key], value)

            elif key == "details":
                for sudut_key, sudut_val in res["details"].items():
                    if sudut_key.startswith("sudut_"):
                        sudut_data[sudut_key] += sudut_val
                        sudut_count[sudut_key] += 1

        feedback_text = res.get("feedback", "")
        if feedback_text:
            feedback_counter[feedback_text] += 1

    # Ambil skor majority dan statistik lainnya
    majority_scores = {}
    average_scores = {}
    min_scores = {}
    max_scores = {}

    for key, counter in skor_counter.items():
        majority_scores[key] = counter.most_common(1)[0][0]
        average_scores[key] = skor_total[key] / count
        min_scores[key] = skor_min[key]
        max_scores[key] = skor_max[key]

    # Hitung rata-rata sudut
    average_sudut = {key: sudut_data[key] / sudut_count[key] for key in sudut_data}

    # Feedback paling sering
    most_common_feedback = feedback_counter.most_common(1)
    feedback_summary = "; ".join([f"{text}" for text, freq in most_common_feedback])

    # Summary teks
    summary_feedback = (
        f"Hasil analisa {count} frame.\n\n"
        f"Mayoritas skor per bagian:\n" + "\n".join([f"- {k}: {v}" for k, v in majority_scores.items()]) + "\n\n"
        f"Rata-rata skor:\n" + "\n".join([f"- {k}: {average_scores[k]:.2f}" for k in average_scores]) + "\n\n"
        f"Skor maksimum:\n" + "\n".join([f"- {k}: {max_scores[k]}" for k in max_scores]) + "\n\n"
        f"Skor minimum:\n" + "\n".join([f"- {k}: {min_scores[k]}" for k in min_scores]) + "\n\n"
        f"Rata-rata sudut tubuh:\n" + "\n".join([f"- {k}: {average_sudut[k]:.2f}" for k in average_sudut]) + "\n\n"
        f"Feedback paling sering:\n{feedback_summary}"
    )

    return {
        "feedback": summary_feedback,
        "majority_scores": majority_scores,
        "average_scores": average_scores,
        "min_scores": min_scores,
        "max_scores": max_scores,
        "average_sudut": average_sudut,
        "total_frames": count
    }
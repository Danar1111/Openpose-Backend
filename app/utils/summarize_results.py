def summarize_results(results):
    # Anggap semua keys sama
    total_data = {}
    count = len(results)

    # Inisialisasi total
    for key in results[0]:
        if isinstance(results[0][key], (int, float)):
            total_data[key] = 0

    # Akumulasi
    for res in results:
        for key in total_data:
            total_data[key] += res[key]

    # Hitung rata-rata
    average_data = {key: total_data[key] / count for key in total_data}

    # Buat feedback
    feedback = f"Hasil analisa {count} frame. Rata-rata Total REBA: {average_data.get('total_reba', 0):.2f}, RULA: {average_data.get('total_rula', 0):.2f}"

    return {
        "feedback": feedback,
        "average": average_data,
        "total_frames": count
    }

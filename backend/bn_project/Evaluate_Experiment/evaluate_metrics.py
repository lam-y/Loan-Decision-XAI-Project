import os
import json
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# 1. تحديد المسارات
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
json_input_path = os.path.join(CURRENT_DIR, "experiment_200_dataset.json") 
excel_output_path = os.path.join(CURRENT_DIR, "evaluation_metrics_summary.xlsx")

# 2. تحميل بيانات التجربة
print(f"جاري تحميل نتائج التجربة من: {json_input_path}...")
with open(json_input_path, "r", encoding="utf-8") as f:
    experiments = json.load(f)

# 3. إعداد أدوات القياس (ROUGE & BLEU)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
smooth = SmoothingFunction().method1

eval_results = []
r1_list, r2_list, rl_list, bleu_list = [], [], [], []

# 4. حساب المقاييس لكل سيناريو
for item in experiments:
    gt = item['ground_truth']
    gen = item['generated_explanation']
    
    # حساب ROUGE
    scores = scorer.score(gt, gen)
    r1 = round(scores['rouge1'].fmeasure, 4)
    r2 = round(scores['rouge2'].fmeasure, 4)
    rl = round(scores['rougeL'].fmeasure, 4)
    
    # حساب BLEU
    ref_tokens = [gt.lower().split()]
    gen_tokens = gen.lower().split()
    bleu = round(sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smooth), 4)
    
    r1_list.append(r1)
    r2_list.append(r2)
    rl_list.append(rl)
    bleu_list.append(bleu)
    
    # حفظ نتائج كل حالة
    eval_results.append({
        'Scenario_ID': item['scenario_id'],
        'BN_Decision': item['bn_decision'],
        'Prob_Approved_%': item['bn_prob_approved'],
        'ROUGE-1': r1,
        'ROUGE-2': r2,
        'ROUGE-L': rl,
        'BLEU': bleu,
        'LLM_Latency_s': item['llm_latency_s'],
        'Ground_Truth': gt,
        'Generated_Explanation': gen
    })

# 5. تحويل النتائج إلى DataFrame
df_details = pd.DataFrame(eval_results)

# إنشاء جدول الملخص الإحصائي العام (Overall Averages)
summary_data = {
    'Metric': ['ROUGE-1 Average', 'ROUGE-2 Average', 'ROUGE-L Average', 'BLEU Average', 'Avg LLM Latency (s)'],
    'Value': [
        round(sum(r1_list) / len(r1_list), 4),
        round(sum(r2_list) / len(r2_list), 4),
        round(sum(rl_list) / len(rl_list), 4),
        round(sum(bleu_list) / len(bleu_list), 4),
        round(df_details['LLM_Latency_s'].mean(), 2)
    ]
}
df_summary = pd.DataFrame(summary_data)

# 6. الحفظ في ملف Excel بصفحتين (صفحة للملخص وصفحة للتفاصيل)
with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
    df_summary.to_excel(writer, sheet_name='Summary Metrics', index=False)
    df_details.to_excel(writer, sheet_name='Detailed Per-Scenario', index=False)

print("\n" + "="*40)
print(" ملخص النتائج الإحصائية العامة:")
print("="*40)
print(df_summary.to_string(index=False))
print("="*40)
print(f"تم تصدير تقرير التقييم الشامل إلى Excel في:\n  {excel_output_path}")
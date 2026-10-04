import sys
sys.stdout = open(sys.stdout.fileno(), "w", encoding="utf-8")
import matplotlib.pyplot as plt
import matplotlib.table as mpl_table

# Data
col_labels = ['Language', 'Code', 'GSM8K', 'ARC', 'OpenBookQA', 'CommonSenseQA', 'NameIndex', 'MiddleMatch', 'ScriptMixed', 'Total']

cell_text = [
    ['Punjabi', 'pa', '✅250 (trans.)', '✅300 (trans.)', '✅495 (trans.)', '✅288 (trans.)', '✅100 (gen.)', '✅100 (gen.)', '✅20 (gen.)', '7/7 ✅'],
    ['Urdu', 'ur', '✅250 (trans.)', '✅300 (trans.)', '✅5767 (trans.)', '✅288 (trans.)', '✅100 (gen.)', '✅100 (gen.)', '✅20 (gen.)', '7/7 ✅'],
    ['Pashto', 'ps', '✅250 (trans.)', '✅892 (trans.)', '✅513 (trans.)', '✅561 (trans.)', '✅100 (gen.)', '✅100 (gen.)', '✅20 (gen.)', '7/7 ✅'],
    ['Arabic', 'ar', '✅250 (native)', '✅300 (native)', '✅445 (native)', '✅245 (native)', '✅100 (gen.)', '✅100 (gen.)', '✅20 (gen.)', '7/7 ✅'],
    ['Persian', 'fa', '✅1564 (native)', '✅300 (native)', '✅500 (native)', '✅300 (native)', '✅100 (gen.)', '✅100 (gen.)', '✅20 (gen.)', '7/7 ✅'],
    ['Sindhi', 'sd', '✅99 (native)', '✅99 (native)', '✅99 (native)', '✅300 (native)', '✅100 (gen.)', '✅100 (gen.)', '✅20 (gen.)', '7/7 ✅'],
    ['Balochi', 'bal', '✅108 (trans.)', '✅58 (trans.)', '❌BLOCKED', '❌BLOCKED', '✅100 (gen.)', '✅100 (gen.)', '✅20 (gen.)', '5/7 ⚠️'],
]

fig, ax = plt.subplots(figsize=(18, 5))
ax.axis('off')

table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    loc='center',
    cellLoc='center',
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.8)

# Style header
for j in range(len(col_labels)):
    cell = table[(0, j)]
    cell.set_facecolor('#1f2937')
    cell.set_text_props(color='white', weight='bold', fontsize=8)

# Style rows
for i in range(1, len(cell_text) + 1):
    row_color = '#f9fafb' if i % 2 == 0 else 'white'
    for j in range(len(col_labels)):
        cell = table[(i, j)]
        cell.set_facecolor(row_color)

# Highlight blocked Balochi cells (OpenBookQA=5, CommonSenseQA=6, Total=10)
balochi_row = 7  # 0=header, 1-7=data, so Balochi is at index 7
for j in [4, 5]:
    cell = table[(balochi_row, j)]
    cell.set_facecolor('#fee2e2')

cell = table[(balochi_row, 9)]
cell.set_facecolor('#fffbeb')

plt.title('Dataset Completion Status by Language and Task (2026-10-04)',
          fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('dataset_status_table.png', dpi=150, bbox_inches='tight')
print('Saved dataset_status_table.png')

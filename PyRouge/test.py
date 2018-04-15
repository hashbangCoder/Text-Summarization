from PyRouge.pyrouge import Rouge

r = Rouge()

# A simple eample of how rouge can be calculated
print r.rouge_l([[1, 7, 6, 7, 5], [0, 2, 8, 3, 5]], [[1, 2, 3, 4, 5], [3, 9, 5]])

# A more practical example of how it can be used for summary evaluation
system_generated_summary = " The Kyrgyz President pushed through the law requiring the use of ink during the upcoming Parliamentary and Presidential elections In an effort to live up to its reputation in the 1990s as an island of democracy. The use of ink is one part of a general effort to show commitment towards more open elections. improper use of this type of ink can cause additional problems as the elections in Afghanistan showed. The use of ink and readers by itself is not a panacea for election ills."
manual_summmary = " The use of invisible ink and ultraviolet readers in the elections of the Kyrgyz Republic which is a small, mountainous state of the former Soviet republic, causing both worries and guarded optimism among different sectors of the population. Though the actual technology behind the ink is not complicated, the presence of ultraviolet light (of the kind used to verify money) causes the ink to glow with a neon yellow light. But, this use of the new technology has caused a lot of problems. "

print r.rouge_l([system_generated_summary], [manual_summmary])

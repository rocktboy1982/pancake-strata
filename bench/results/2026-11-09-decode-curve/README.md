# Decode speed along a reply, after short and long prompts (MOD_PLAN item 2), 2026-11-09

tools/pancake/decode_curve.py, fast-hip preset, streaming, tok/s per 64-token segment (machine not quiet: compare shapes).

    edit       prompt   2073  ttft 2.50s  decode 126.9  per segment:  153 141 109  69 137 178 128 194   (first request, cold)
    edit-long  prompt  13789  ttft 7.09s  decode 150.8  per segment:  180 181 134  89 163 208 129 221
    edit       prompt   2073  ttft 0.33s  decode 156.3  per segment:  188 178 143  94 173 204 140 208
    edit-long  prompt  13789  ttft 0.26s  decode 155.9  per segment:  184 195 135  91 178 214 131 226
    long       prompt  11731  ttft 4.64s  decode  94.6  per segment:   64  87  94 100 107 103 110 114

- The same edit reply after a 13.8K-token prompt (edit-long, prompt read fresh in run 2) decodes as fast as after the
  2K prompt, segment by segment: the refill of the slots the prompt path borrowed costs nothing visible.
- The "decode after a long prompt" drop of the A/B harness comes from its reply: a prose summary drafts worse
  (acceptance 0.62 vs 0.87 for the edit, engine log) and starts slow (64 tok/s) while the cache adapts to a new domain,
  rising to 114.  That start-up is what the domain/conversation seeds (items 3-4) and k-th-miss admission (item 14) target.
- Item 2 closed: no refill fix needed.

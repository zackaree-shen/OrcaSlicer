# Template-based CN translator for gcode-diff quality reports.
# Every verdict/recommendation string from score/engine.rs, score/suggestions.rs,
# and appearance.rs is enumerated as an (english, chinese) pair.
import re

_RAW = [
    # ===== Cooling =====
    ("No part cooling fan commands -- overhangs and bridges may sag", "\u65e0\u90e8\u4ef6\u51b7\u5374\u98ce\u6247\u6307\u4ee4\u2014\u2014\u60ac\u5782\u548c\u6865\u63a5\u53ef\u80fd\u4e0b\u5760"),
    ("Enable part cooling fan in slicer (M106)", "\u5728\u5207\u7247\u5668\u4e2d\u542f\u7528\u90e8\u4ef6\u51b7\u5374\u98ce\u6247 (M106)"),
    ("No fan commands (expected for {})", "\u65e0\u98ce\u6247\u6307\u4ee4\uff08{}\u9884\u671f\u5982\u6b64\uff09"),
    ("Fan PWM within expected range", "\u98ce\u6247 PWM \u5728\u9884\u671f\u8303\u56f4\u5185"),
    ("Fan PWM low for {}", "{}\u7684\u98ce\u6247 PWM \u504f\u4f4e"),
    ("Increase part cooling fan speed", "\u63d0\u9ad8\u90e8\u4ef6\u51b7\u5374\u98ce\u6247\u8f6c\u901f"),
    ("Fan PWM above expected range (may not be harmful)", "\u98ce\u6247 PWM \u9ad8\u4e8e\u9884\u671f\u8303\u56f4\uff08\u53ef\u80fd\u65e0\u5bb3\uff09"),
    ("Mean fan PWM above {} band; over-cooling may cause warping -- verify per-layer fan strategy (mean averages first-layer offset)", "\u5e73\u5747\u98ce\u6247 PWM \u9ad8\u4e8e{}\u533a\u95f4\uff1b\u8fc7\u5ea6\u51b7\u5374\u53ef\u80fd\u5bfc\u81f4\u7fd8\u66f2\u2014\u2014\u8bf7\u68c0\u67e5\u9010\u5c42\u98ce\u6247\u7b56\u7565\uff08\u9996\u5c42\u504f\u79fb\u88ab\u5747\u503c\u5e73\u5747\u5316\uff09"),
    ("Review slicer fan overrides for {} (disable fan for first layer / overhangs threshold)", "\u68c0\u67e5{}\u7684\u5207\u7247\u5668\u98ce\u6249\u8986\u76d6\u8bbe\u7f6e\uff08\u9996\u5c42\u7981\u7528\u98ce\u6247/\u60ac\u5782\u9608\u503c\uff09"),
    # ===== Retraction =====
    ("No retractions detected (may be vase mode)", "\u672a\u68c0\u6d4b\u5230\u56de\u62bd\uff08\u53ef\u80fd\u662f\u82b1\u74f6\u6a21\u5f0f\uff09"),
    ("Retraction {} too short for {} - may not relieve nozzle pressure", "\u56de\u62bd{}\u5bf9{}\u8fc7\u77ed\u2014\u2014\u53ef\u80fd\u65e0\u6cd5\u91ca\u653e\u55b7\u5634\u538b\u529b"),
    ("Increase retraction to at least {}mm", "\u5c06\u56de\u62bd\u589e\u52a0\u81f3\u81f3\u5c11{}mm"),
    ("Retraction {} excessive for {} - risks grinding and heat creep", "\u56de\u62bd{}\u5bf9{}\u8fc7\u91cf\u2014\u2014\u6709\u7814\u78e8\u8017\u6750\u548c\u70ed\u722c\u5347\u98ce\u9669"),
    ("Reduce retraction below {}mm", "\u5c06\u56de\u62bd\u964d\u4f4e\u81f3{}mm\u4ee5\u4e0b"),
    ("Retraction {}mm low for {}", "\u56de\u62bd{}mm\u5bf9{}\u504f\u4f4e"),
    ("Retraction {}mm high for {}", "\u56de\u62bd{}mm\u5bf9{}\u504f\u9ad8"),
    ("Consider reducing below {}mm", "\u5efa\u8bae\u964d\u4f4e\u81f3{}mm\u4ee5\u4e0b"),
    ("Extremely high retraction density", "\u6781\u9ad8\u7684\u56de\u62bd\u5bc6\u5ea6"),
    ("Slicer profile has incorrect retraction settings", "\u5207\u7247\u5668\u914d\u7f6e\u6587\u4ef6\u7684\u56de\u62bd\u8bbe\u7f6e\u4e0d\u6b63\u786e"),
    ("Elevated retraction density", "\u504f\u9ad8\u7684\u56de\u62bd\u5bc6\u5ea6"),
    ("Check retraction distance and speed", "\u68c0\u67e5\u56de\u62bd\u8ddd\u79bb\u548c\u901f\u5ea6"),
    ("Retraction {} within defaults but extruder type undetermined (avg in 1.5-3.0mm dead zone: OK for long direct-drive or short Bowden)", "\u56de\u62bd{}\u5728\u9ed8\u8ba4\u8303\u56f4\u5185\uff0c\u4f46\u6324\u51fa\u5934\u7c7b\u578b\u65e0\u6cd5\u786e\u5b9a\uff08\u5747\u503c\u57281.5-3.0mm\u6b7b\u533a\uff1a\u9002\u7528\u4e8e\u957f\u76f4\u9a71\u6216\u77edBowden\uff09"),
    ("Confirm extruder type; tune retraction for your specific hotend", "\u786e\u8ba4\u6324\u51fa\u5934\u7c7b\u578b\uff1b\u4e3a\u60a8\u7684\u7279\u5b9a\u70ed\u7aef\u8c03\u6821\u56de\u62bd"),
    ("Retraction OK ({})", "\u56de\u62bd\u6b63\u5e38\uff08{}\uff09"),
    ("; extreme density", "\uff1b\u6781\u7aef\u5bc6\u5ea6"),
    ("; elevated density", "\uff1b\u504f\u9ad8\u5bc6\u5ea6"),
    # ===== Travel =====
    ("Not assessable: negligible movement (no travel or extrusion length recorded)", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u8fd0\u52a8\u91cf\u53ef\u5ffd\u7565\uff08\u672a\u8bb0\u5f55\u7a7a\u9a76\u6216\u6324\u51fa\u957f\u5ea6\uff09"),
    ("Efficient travel ratio: {}%", "\u9ad8\u6548\u7684\u7a7a\u9a76\u6bd4\uff1a{}%"),
    ("Travel ratio: {}% - acceptable", "\u7a7a\u9a76\u6bd4\uff1a{}%\u2014\u2014\u53ef\u63a5\u53d7"),
    ("Travel ratio: {}% - suboptimal path planning", "\u7a7a\u9a76\u6bd4\uff1a{}%\u2014\u2014\u8def\u5f84\u89c4\u5212\u6b20\u4f73"),
    ("Consider optimizing part arrangement or avoid crossing perimeters", "\u8003\u8651\u4f18\u5316\u96f6\u4ef6\u6392\u5217\u6216\u907f\u514d\u7a7f\u8d8a\u8f6e\u5ed3"),
    ("Travel ratio: {}% - excessive, path planning may be broken", "\u7a7a\u9a76\u6bd4\uff1a{}%\u2014\u2014\u8fc7\u9ad8\uff0c\u8def\u5f84\u89c4\u5212\u53ef\u80fd\u6709\u95ee\u9898"),
    # ===== Temperature =====
    ("Not assessable: no temperature commands detected (printer may use a firmware preset -- not verifiable from G-code)", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u672a\u68c0\u6d4b\u5230\u6e29\u5ea6\u6307\u4ee4\uff08\u6253\u5370\u673a\u53ef\u80fd\u4f7f\u7528\u56fa\u4ef6\u9884\u8bbe\u2014\u2014\u65e0\u6cd5\u4ece G-code \u9a8c\u8bc1\uff09"),
    ("Extruder temperature within recommended range", "\u6324\u51fa\u5934\u6e29\u5ea6\u5728\u63a8\u8350\u8303\u56f4\u5185"),
    ("Extruder temperature too low for {}", "{}\u7684\u6324\u51fa\u5934\u6e29\u5ea6\u8fc7\u4f4e"),
    ("Extruder temperature too high for {}", "{}\u7684\u6324\u51fa\u5934\u6e29\u5ea6\u8fc7\u9ad8"),
    ("Increase nozzle temperature", "\u63d0\u9ad8\u55b7\u5634\u6e29\u5ea6"),
    ("Decrease nozzle temperature", "\u964d\u4f4e\u55b7\u5634\u6e29\u5ea6"),
    # ===== Extrusion Uniformity =====
    ("Too few layers ({}) for uniformity analysis; need >= {}", "\u5c42\u6570\u592a\u5c11\uff08{}\uff09\uff0c\u65e0\u6cd5\u8fdb\u884c\u5747\u5300\u6027\u5206\u6790\uff1b\u9700\u8981 >= {}"),
    ("Not assessable: no ;TYPE:Outer wall annotations found (slicer may not emit ;TYPE: comments)", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u672a\u627e\u5230 ;TYPE:\u5916\u5899\u6807\u6ce8\uff08\u5207\u7247\u5668\u53ef\u80fd\u4e0d\u8f93\u51fa ;TYPE: \u6ce8\u91ca\uff09"),
    ("Not enough Outer wall layers ({}) for stable CV; need >= {}", "\u5916\u5899\u5c42\u6570\u4e0d\u8db3\uff08{}\uff09\uff0cCV \u4e0d\u7a33\u5b9a\uff1b\u9700\u8981 >= {}"),
    ("Outer wall extrusion consistent (CV={}%)", "\u5916\u5899\u6324\u51fa\u4e00\u81f4\uff08CV={}%\uff09"),
    ("Outer wall extrusion variation (CV={}%) - may be flow inconsistency", "\u5916\u5899\u6324\u51fa\u53d8\u5316\uff08CV={}%\uff09\u2014\u2014\u53ef\u80fd\u662f\u6d41\u91cf\u4e0d\u4e00\u81f4"),
    ("Check extrusion multiplier stability", "\u68c0\u67e5\u6324\u51fa\u500d\u7387\u7a33\u5b9a\u6027"),
    ("Outer wall extrusion variation (CV={}%) - quality may be affected", "\u5916\u5899\u6324\u51fa\u53d8\u5316\uff08CV={}%\uff09\u2014\u2014\u8d28\u91cf\u53ef\u80fd\u53d7\u5f71\u54cd"),
    # ===== Wall & Shell =====
    ("Not assessable: no slice parameters available", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u65e0\u5207\u7247\u53c2\u6570"),
    ("Not assessable: wall/shell keys not emitted by slicer", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u5207\u7247\u5668\u672a\u8f93\u51fa\u5899\u58c1/\u5916\u58f3\u53c2\u6570"),
    ("Wall/shell layers explicitly set to 0 -- cannot distinguish shell-less/vase body from misconfiguration", "\u5899\u58c1/\u5916\u58f3\u5c42\u6570\u663e\u5f0f\u8bbe\u4e3a0\u2014\u2014\u65e0\u6cd5\u533a\u5206\u65e0\u58f3/\u82b1\u74f6\u4e3b\u4f53\u8fd8\u662f\u914d\u7f6e\u9519\u8bef"),
    ("Wall loops and shell layers adequate", "\u5899\u58c1\u5708\u6570\u548c\u5916\u58f3\u5c42\u6570\u5145\u8db3"),
    ("Wall loops or shell layers below minimum ({}/{}/{} wall, {}/{}/{} top, {}/{}/{} bottom)", "\u5899\u58c1\u5708\u6570\u6216\u5916\u58f3\u5c42\u6570\u4f4e\u4e8e\u6700\u5c0f\u503c\uff08\u5899{}/{}/{}\uff0c\u9876\u5c42{}/{}/{}\uff0c\u5e95\u5c42{}/{}/{}\uff09"),
    ("Increase to at least {} wall loops, {} top / {} bottom shells", "\u81f3\u5c11\u589e\u52a0\u5230{}\u5708\u5899\u58c1\uff0c{}/{}\u9876\u5c42/\u5e95\u5c42\u5916\u58f3"),
    # ===== Print Volume =====
    ("Not assessable: bounding box not computed (no spatial data recorded)", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u672a\u8ba1\u7b97\u5305\u56f4\u76d2\uff08\u65e0\u7a7a\u95f4\u6570\u636e\uff09"),
    ("Model fits within configured bed ({}x{}x{}mm)", "\u6a21\u578b\u5728\u914d\u7f6e\u7684\u70ed\u5e8a\u8303\u56f4\u5185\uff08{}x{}x{}mm\uff09"),
    ("Model exceeds configured bed dimensions", "\u6a21\u578b\u8d85\u51fa\u914d\u7f6e\u7684\u70ed\u5e8a\u5c3a\u5bf8"),
    ("Scale or reposition model to fit within print volume", "\u7f29\u653e\u6216\u91cd\u65b0\u5b9a\u4f4d\u6a21\u578b\u4ee5\u9002\u5e94\u6253\u5370\u4f53\u79ef"),
    # ===== Layer Height =====
    ("Not assessable: layer height not determinable from metadata", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u65e0\u6cd5\u4ece\u5143\u6570\u636e\u786e\u5b9a\u5c42\u9ad8"),
    ("Layer height {}mm in typical range", "\u5c42\u9ad8{}mm \u5728\u5178\u578b\u8303\u56f4\u5185"),
    ("Layer height unusually high ({}) -- detail will be lost", "\u5c42\u9ad8\u5f02\u5e38\u504f\u9ad8\uff08{}\uff09\u2014\u2014\u7ec6\u8282\u4f1a\u4e22\u5931"),
    ("Layer height unusually low ({}) -- print will be slow", "\u5c42\u9ad8\u5f02\u5e38\u504f\u4f4e\uff08{}\uff09\u2014\u2014\u6253\u5370\u4f1a\u5f88\u6162"),
    # ===== Volumetric Flow =====
    ("Not assessable: no volumetric flow data available (no segment flow recorded)", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u65e0\u4f53\u79ef\u6d41\u91cf\u6570\u636e\uff08\u672a\u8bb0\u5f55\u7ebf\u6bb5\u6d41\u91cf\uff09"),
    ("Peak layer flow {} mm^3/s (layer {}), hotend limit ~{} mm^3/s, {}", "\u5cf0\u503c\u5c42\u6d41\u91cf{} mm\u00b3/s\uff08\u7b2c{}\u5c42\uff09\uff0c\u70ed\u7aef\u6781\u9650~{} mm\u00b3/s\uff0c{}"),
    ("Flow exceeds hotend capacity by {}%; reduce speed or increase temperature", "\u6d41\u91cf\u8d85\u51fa\u70ed\u7aef\u80fd\u529b{}%\uff1b\u964d\u4f4e\u901f\u5ea6\u6216\u63d0\u9ad8\u6e29\u5ea6"),
    ("Reduce print speed or layer height; flow {}% over hotend capacity -- expect under-extrusion", "\u964d\u4f4e\u6253\u5370\u901f\u5ea6\u6216\u5c42\u9ad8\uff1b\u6d41\u91cf\u8d85\u51fa\u70ed\u7aef\u80fd\u529b{}%\u2014\u2014\u9884\u8ba1\u4f1a\u51fa\u73b0\u6b20\u6324\u51fa"),
    ("Low print temperature ({}C near material minimum) may reduce achievable max flow; the nozzle-scaled threshold assumes nominal temperature", "\u6253\u5370\u6e29\u5ea6\u504f\u4f4e\uff08{}C \u63a5\u8fd1\u6750\u6599\u4e0b\u9650\uff09\u53ef\u80fd\u964d\u4f4e\u6700\u5927\u6d41\u91cf\uff1b\u55b7\u5634\u7f29\u653e\u9608\u503c\u5047\u8bbe\u6807\u79f0\u6e29\u5ea6"),
    ("nozzle {}mm (from header)", "\u55b7\u5634{}mm\uff08\u6765\u81ea\u6587\u4ef6\u5934\uff09"),
    ("nozzle {}mm (config)", "\u55b7\u5634{}mm\uff08\u914d\u7f6e\uff09"),
    ("no nozzle size parsed -- using 0.4mm baseline (may over-report for larger nozzles)", "\u672a\u89e3\u6790\u55b7\u5634\u5c3a\u5bf8\u2014\u2014\u4f7f\u75280.4mm\u57fa\u7ebf\uff08\u5927\u55b7\u5634\u53ef\u80fd\u9ad8\u4f30\uff09"),
    # ===== Speed Consistency =====
    ("Not assessable: fewer than 3 layers carry speed data", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u5c11\u4e8e3\u5c42\u6709\u901f\u5ea6\u6570\u636e"),
    ("Not assessable: only {} layers carry speed data (need >= 3)", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u4ec5{}\u5c42\u6709\u901f\u5ea6\u6570\u636e\uff08\u9700\u8981 >= 3\uff09"),
    ("Speed consistent across layers (CV={}%)", "\u5404\u5c42\u901f\u5ea6\u4e00\u81f4\uff08CV={}%\uff09"),
    ("Speed variation (CV={}%) - minor, may affect surface", "\u901f\u5ea6\u53d8\u5316\uff08CV={}%\uff09\u2014\u2014\u8f7b\u5fae\uff0c\u53ef\u80fd\u5f71\u54cd\u8868\u9762"),
    ("High speed variation (CV={}%) - frequent accel/decel hurts quality", "\u901f\u5ea6\u53d8\u5316\u8f83\u5927\uff08CV={}%\uff09\u2014\u2014\u9891\u7e41\u52a0\u51cf\u901f\u5f71\u54cd\u8d28\u91cf"),
    # ===== First Layer =====
    ("Not assessable: first layer height not available from metadata", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u5143\u6570\u636e\u4e2d\u65e0\u9996\u5c42\u9ad8\u5ea6"),
    ("Not assessable: first layer height not in metadata and motion data contaminated by start G-code", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u5143\u6570\u636e\u4e2d\u65e0\u9996\u5c42\u9ad8\u5ea6\uff0c\u4e14\u8fd0\u52a8\u6570\u636e\u88ab\u8d77\u59cb G-code \u6c61\u67d3"),
    ("First layer height matches declared ({}/{}mm)", "\u9996\u5c42\u9ad8\u5ea6\u4e0e\u58f0\u660e\u4e00\u81f4\uff08{}/{}mm\uff09"),
    ("First layer height mismatch: declared {}mm, actual ~{}mm", "\u9996\u5c42\u9ad8\u5ea6\u4e0d\u5339\u914d\uff1a\u58f0\u660e{}mm\uff0c\u5b9e\u9645~{}mm"),
    ("First layer height significantly off: declared {}mm, actual ~{}mm", "\u9996\u5c42\u9ad8\u5ea6\u504f\u5dee\u8f83\u5927\uff1a\u58f0\u660e{}mm\uff0c\u5b9e\u9645~{}mm"),
    ("First layer height: {}mm declared, cannot verify from motion data", "\u9996\u5c42\u9ad8\u5ea6\uff1a\u58f0\u660e{}mm\uff0c\u65e0\u6cd5\u4ece\u8fd0\u52a8\u6570\u636e\u9a8c\u8bc1"),
    ("Check bed leveling and Z offset", "\u68c0\u67e5\u70ed\u5e8a\u8c03\u5e73\u548c Z \u504f\u79fb"),
    # ===== Temp Stability =====
    ("Not assessable: fewer than 3 layers carry temperature data (excluding first layer)", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u5c11\u4e8e3\u5c42\u6709\u6e29\u5ea6\u6570\u636e\uff08\u4e0d\u542b\u9996\u5c42\uff09"),
    ("Temperature stable across print (std={}C, first layer excluded){}", "\u6253\u5370\u8fc7\u7a0b\u4e2d\u6e29\u5ea6\u7a33\u5b9a\uff08\u6807\u51c6\u5dee={}C\uff0c\u4e0d\u542b\u9996\u5c42\uff09{}"),
    ("Temperature drift {}C (first layer excluded) - layer bonding may vary", "\u6e29\u5ea6\u6f02\u79fb{}C\uff08\u4e0d\u542b\u9996\u5c42\uff09\u2014\u2014\u5c42\u95f4\u7ed3\u5408\u53ef\u80fd\u4e0d\u5747"),
    ("Large temperature drift {}C (first layer excluded) - risk of delamination", "\u6e29\u5ea6\u6f02\u79fb\u8f83\u5927{}C\uff08\u4e0d\u542b\u9996\u5c42\uff09\u2014\u2014\u6709\u5206\u5c42\u98ce\u9669"),
    (" | pred T_sub={}C < Tg={}C \u2014 cold weld (Springer 2024, R^2=0.980)", " | \u9884\u6d4b T_sub={}C < Tg={}C\u2014\u2014\u51b7\u710a\uff08Springer 2024, R\u00b2=0.980\uff09"),
    (" | pred T_sub={}C > Tg={}C \u2014 OK", " | \u9884\u6d4b T_sub={}C > Tg={}C\u2014\u2014\u6b63\u5e38"),
    # ===== Support Ratio =====
    ("Not assessable: no per-type extrusion data available (no ;TYPE: annotations parsed)", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u65e0\u6309\u7c7b\u578b\u7684\u6324\u51fa\u6570\u636e\uff08\u672a\u89e3\u6790 ;TYPE: \u6807\u6ce8\uff09"),
    ("Minimal support ({}% of extrusion)", "\u652f\u6491\u6781\u5c11\uff08\u5360\u6324\u51fa{}%\uff09"),
    ("Support {}% of extrusion - moderate", "\u652f\u6491\u5360\u6324\u51fa{}%\u2014\u2014\u9002\u4e2d"),
    ("Support {}% of extrusion - consider reducing density", "\u652f\u6491\u5360\u6324\u51fa{}%\u2014\u2014\u5efa\u8bae\u964d\u4f4e\u5bc6\u5ea6"),
    ("Excessive support {}% of extrusion", "\u652f\u6491\u8fc7\u591a\uff0c\u5360\u6324\u51fa{}%"),
    ("Reduce support density or use tree supports", "\u964d\u4f4e\u652f\u6491\u5bc6\u5ea6\u6216\u4f7f\u7528\u6811\u72b6\u652f\u6491"),
    # ===== Corner Speed =====
    ("Smooth speed profile - low ringing risk", "\u901f\u5ea6\u66f2\u7ebf\u5e73\u6ed1\u2014\u2014\u632f\u7eb9\u98ce\u9669\u4f4e"),
    ("{}% of layers have corner stop-and-go - minor ringing risk", "{}%\u7684\u5c42\u6709\u62d0\u89d2\u505c\u987f\u2014\u2014\u8f7b\u5fae\u632f\u7eb9\u98ce\u9669"),
    ("{}% of layers have corner stop-and-go - ringing/ghosting likely", "{}%\u7684\u5c42\u6709\u62d0\u89d2\u505c\u987f\u2014\u2014\u53ef\u80fd\u51fa\u73b0\u632f\u7eb9/\u91cd\u5f71"),
    ("Reduce jerk/acceleration or print speed", "\u964d\u4f4e\u6296\u52a8/\u52a0\u901f\u5ea6\u6216\u6253\u5370\u901f\u5ea6"),
    # ===== Min Layer Time =====
    ("Not assessable: fewer than 3 layers recorded", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u5c11\u4e8e3\u5c42\u6709\u8bb0\u5f55"),
    ("Not assessable: fewer than 3 layers carry timing data", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u5c11\u4e8e3\u5c42\u6709\u65f6\u5e8f\u6570\u636e"),
    ("All layers have sufficient cooling time (min {}s, threshold {})", "\u6240\u6709\u5c42\u6709\u8db3\u591f\u7684\u51b7\u5374\u65f6\u95f4\uff08\u6700\u5c0f{}s\uff0c\u9608\u503c{}\uff09"),
    ("{}% layers below {}, bond {}% at worst layer (de Gennes/Wool t^(1/4))", "{}%\u7684\u5c42\u4f4e\u4e8e{}\uff0c\u6700\u5dee\u5c42\u7ed3\u5408\u5f3a\u5ea6{}%\uff08de Gennes/Wool t^(1/4)\uff09"),
    (" [phys, T_amb~25C]", " [\u7269\u7406\u6a21\u578b\uff0cT_amb~25\u2103]"),
    (" [model trust: below VERIFIED]", " [\u6a21\u578b\u4fe1\u4efb\u5ea6\uff1a\u672a\u8fbe\u5df2\u9a8c\u8bc1]"),
    ("Increase minimum layer time or reduce speed", "\u589e\u52a0\u6700\u5c0f\u5c42\u65f6\u95f4\u6216\u964d\u4f4e\u901f\u5ea6"),
    (" [slicer slow_down_layer_time={}s < material safe minimum {}s -- slicer allows faster printing than bond strength requires]", " [\u5207\u7247\u5668 slow_down_layer_time={}s < \u6750\u6599\u5b89\u5168\u6700\u5c0f\u503c{}s\u2014\u2014\u5207\u7247\u5668\u5141\u8bb8\u7684\u6253\u5370\u901f\u5ea6\u8d85\u8fc7\u7ed3\u5408\u5f3a\u5ea6\u8981\u6c42]"),
    # ===== Layer Consistency =====
    ("Not assessable: slicer did not declare total layer count", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u5207\u7247\u5668\u672a\u58f0\u660e\u603b\u5c42\u6570"),
    ("Layer count matches declared ({} declared, {} detected)", "\u5c42\u6570\u4e0e\u58f0\u660e\u4e00\u81f4\uff08\u58f0\u660e{}\u5c42\uff0c\u68c0\u6d4b\u5230{}\u5c42\uff09"),
    ("Layer count minor mismatch ({} declared, {} detected)", "\u5c42\u6570\u8f7b\u5fae\u4e0d\u5339\u914d\uff08\u58f0\u660e{}\u5c42\uff0c\u68c0\u6d4b\u5230{}\u5c42\uff09"),
    ("Layer count mismatch ({} declared, {} detected) -- possible truncation", "\u5c42\u6570\u4e0d\u5339\u914d\uff08\u58f0\u660e{}\u5c42\uff0c\u68c0\u6d4b\u5230{}\u5c42\uff09\u2014\u2014\u53ef\u80fd\u622a\u65ad"),
    ("Layer count severely mismatched ({} declared, {} detected) -- file may be truncated or corrupt", "\u5c42\u6570\u4e25\u91cd\u4e0d\u5339\u914d\uff08\u58f0\u660e{}\u5c42\uff0c\u68c0\u6d4b\u5230{}\u5c42\uff09\u2014\u2014\u6587\u4ef6\u53ef\u80fd\u622a\u65ad\u6216\u635f\u574f"),
    ("Check G-code file integrity and re-slice if needed", "\u68c0\u67e5 G-code \u6587\u4ef6\u5b8c\u6574\u6027\uff0c\u5fc5\u8981\u65f6\u91cd\u65b0\u5207\u7247"),
    # ===== Multi-Extruder =====
    ("Not assessable: single-extruder print (no tool changes)", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u5355\u6324\u51fa\u5934\u6253\u5370\uff08\u65e0\u6362\u8272\uff09"),
    ("{} tool changes ({}, {}/layer, max burst {}/layer)", "{}\u6b21\u6362\u8272\uff08{}\uff0c{}/\u5c42\uff0c\u6700\u5927\u8fde\u53d1{}/\u5c42\uff09"),
    ("; {} purge blocks ({}% of changes -- poop-purge/AMS style)", "\uff1b{}\u4e2a\u6e05\u4ed3\u5757\uff08\u5360\u6362\u8272{}%\u2014\u2014\u6392\u6599/AMS\u98ce\u683c\uff09"),
    ("; wipe-tower workflow (clean transitions)", "\uff1b\u64e6\u5854\u5de5\u4f5c\u6d41\uff08\u5e72\u51c0\u7684\u6362\u8272\uff09"),
    # ===== Overhang =====
    ("Not assessable: no point-cloud data (pass commands to enable overhang analysis)", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u65e0\u70b9\u4e91\u6570\u636e\uff08\u4f20\u5165\u547d\u4ee4\u4ee5\u542f\u7528\u60ac\u5782\u5206\u6790\uff09"),
    ("Not assessable: too few visible-surface points per layer", "\u65e0\u6cd5\u8bc4\u4f30\uff1a\u6bcf\u5c42\u53ef\u89c1\u8868\u9762\u70b9\u592a\u5c11"),
    (" [slicer support threshold: {}deg -- overhangs above this should have support]", " [\u5207\u7247\u5668\u652f\u6491\u9608\u503c\uff1a{}\u5ea6\u2014\u2014\u8d85\u8fc7\u6b64\u89d2\u5ea6\u7684\u60ac\u5782\u5e94\u6709\u652f\u6491]"),
    ("Reduce overhang angle, add support, or increase cooling", "\u51cf\u5c0f\u60ac\u5782\u89d2\u5ea6\u3001\u6dfb\u52a0\u652f\u6491\u6216\u589e\u5f3a\u51b7\u5374"),
    ("Negligible overhang (no layer has a real overhang region; only isolated prime/wipe points)", "\u60ac\u5782\u53ef\u5ffd\u7565\uff08\u65e0\u5c42\u5177\u6709\u771f\u5b9e\u60ac\u5782\u533a\u57df\uff1b\u4ec5\u6709\u5b64\u7acb\u7684\u64e6\u62ed/\u9884\u6324\u51fa\u70b9\uff09"),
    ("Worst overhang {}deg on layer {} (45deg rule; {}% points unsupported)", "\u6700\u5dee\u60ac\u5782{}\u5ea6\uff0c\u5728\u7b2c{}\u5c42\uff0845\u5ea6\u89c4\u5219\uff1b{}%\u70b9\u672a\u88ab\u652f\u6491\uff09"),
    ("Overhang present (worst layer {}, {}% unsupported, max {}mm) -- angle not scored (layer height unparseable)", "\u5b58\u5728\u60ac\u5782\uff08\u6700\u5dee\u5c42{}\uff0c{}%\u672a\u88ab\u652f\u6491\uff0c\u6700\u5927{}mm\uff09\u2014\u2014\u89d2\u5ea6\u672a\u8bc4\u5206\uff08\u5c42\u9ad8\u65e0\u6cd5\u89e3\u6790\uff09"),
    # ===== Material Anisotropy =====
    ("Z strength {}% of XY ({})", "Z \u65b9\u5411\u5f3a\u5ea6\u4e3a XY \u7684{}%\uff08{}\uff09"),
    # ===== Suggestions =====
    ("Reduces stringing", "\u51cf\u5c11\u62c9\u4e1d"),
    ("Improves layer bonding", "\u6539\u5584\u5c42\u95f4\u7ed3\u5408"),
    ("Increases layer adhesion", "\u589e\u5f3a\u5c42\u95f4\u9644\u7740\u529b"),
    ("Improves dimensional accuracy", "\u63d0\u9ad8\u5c3a\u5bf8\u7cbe\u5ea6"),
    ("Better surface quality", "\u66f4\u597d\u7684\u8868\u9762\u8d28\u91cf"),
    ("Better bed adhesion", "\u66f4\u597d\u7684\u70ed\u5e8a\u9644\u7740\u529b"),
    ("Eliminates over/under-extrusion", "\u6d88\u9664\u8fc7\u6324\u51fa/\u6b20\u6324\u51fa"),
    ("Reduces stringing and ooze", "\u51cf\u5c11\u62c9\u4e1d\u548c\u6e17\u6599"),
    ("Prevents overhang sag", "\u9632\u6b62\u60ac\u5782\u4e0b\u5760"),
    ("Stronger walls", "\u66f4\u5f3a\u7684\u5899\u58c1"),
    ("Add support or reduce overhang angle; increase part cooling", "\u6dfb\u52a0\u652f\u6491\u6216\u51cf\u5c0f\u60ac\u5782\u89d2\u5ea6\uff1b\u589e\u5f3a\u90e8\u4ef6\u51b7\u5374"),
    ("Prevents extruder skipping", "\u9632\u6b62\u6324\u51fa\u5934\u8df3\u9f7f"),
    ("Avoids edge artifacts", "\u907f\u514d\u8fb9\u7f18\u7455\u75b5"),
    ("Shorter travel paths", "\u66f4\u77ed\u7684\u7a7a\u9a76\u8def\u5f84"),
    ("Consistent layer finish", "\u4e00\u81f4\u7684\u5c42\u8868\u9762"),
    ("Structural integrity", "\u7ed3\u6784\u5b8c\u6574\u6027"),
    ("Stable extrusion temperature", "\u7a33\u5b9a\u7684\u6324\u51fa\u6e29\u5ea6"),
    ("Easier support removal", "\u66f4\u5bb9\u6613\u62c6\u9664\u652f\u6491"),
    ("Less bulge at corners", "\u62d0\u89d2\u5904\u51f8\u8d77\u66f4\u5c11"),
    ("Proper cooling per layer", "\u6bcf\u5c42\u5145\u5206\u51b7\u5374"),
    ("Consistent extrusion pressure", "\u4e00\u81f4\u7684\u6324\u51fa\u538b\u529b"),
    ("Valid G-code", "\u6709\u6548\u7684 G-code"),
    ("Reliable tool changes", "\u53ef\u9760\u7684\u6362\u8272"),
    ("Unknown", "\u672a\u77e5"),
]

_W = {
    "nozzle": "\u55b7\u5634", "fan": "\u98ce\u6247", "speed": "\u901f\u5ea6",
    "temperature": "\u6e29\u5ea6", "temp": "\u6e29\u5ea6", "extruder": "\u6324\u51fa\u5934",
    "extrusion": "\u6324\u51fa", "retraction": "\u56de\u62bd", "support": "\u652f\u6491",
    "model": "\u6a21\u578b", "printer": "\u6253\u5370\u673a", "slicer": "\u5207\u7247\u5668",
    "material": "\u6750\u6599", "flow": "\u6d41\u91cf", "layer": "\u5c42",
    "layers": "\u5c42", "bed": "\u70ed\u5e8a", "travel": "\u7a7a\u9a76",
    "wall": "\u5899\u58c1", "shell": "\u5916\u58f3", "cooling": "\u51b7\u5374",
    "overhang": "\u60ac\u5782", "print": "\u6253\u5370", "printing": "\u6253\u5370",
    "bond": "\u7ed3\u5408", "purge": "\u6e05\u4ed3", "tower": "\u5854",
    "transitions": "\u6362\u8272", "score": "\u5206\u6570", "range": "\u8303\u56f4",
    "limit": "\u6781\u9650", "capacity": "\u80fd\u529b", "ratio": "\u6bd4\u7387",
    "density": "\u5bc6\u5ea6", "profile": "\u914d\u7f6e", "settings": "\u8bbe\u7f6e",
    "header": "\u6587\u4ef6\u5934", "config": "\u914d\u7f6e", "baseline": "\u57fa\u7ebf",
    "increase": "\u589e\u52a0", "reduce": "\u964d\u4f4e", "check": "\u68c0\u67e5",
    "enable": "\u542f\u7528", "excessive": "\u8fc7\u91cf", "consistent": "\u4e00\u81f4",
    "stable": "\u7a33\u5b9a", "extreme": "\u6781\u7aef", "elevated": "\u504f\u9ad8",
    "acceptable": "\u53ef\u63a5\u53d7", "moderate": "\u9002\u4e2d", "minimal": "\u6781\u5c11",
    "exceeds": "\u8d85\u51fa", "too": "\u8fc7", "low": "\u504f\u4f4e",
    "high": "\u504f\u9ad8", "short": "\u8fc7\u77ed", "long": "\u8fc7\u957f",
    "slow": "\u6162", "fast": "\u5feb", "large": "\u5927",
    "small": "\u5c0f", "worst": "\u6700\u5dee", "first": "\u9996",
    "outer": "\u5916", "top": "\u9876\u90e8", "bottom": "\u5e95\u90e8",
    "single": "\u5355", "dual": "\u53cc", "direct": "\u76f4\u9a71", "direct-drive": "\u76f4\u9a71", "drive": "\u9a71\u52a8",
    "for": "\u9002\u7528\u4e8e", "within": "\u5728...\u5185", "below": "\u4f4e\u4e8e",
    "above": "\u8d85\u8fc7", "from": "\u6765\u81ea", "using": "\u4f7f\u7528",
    "detected": "\u68c0\u6d4b\u5230", "declared": "\u58f0\u660e", "expected": "\u9884\u671f",
    "may": "\u53ef\u80fd", "should": "\u5e94", "will": "\u5c06",
    "not": "\u672a", "no": "\u65e0", "or": "\u6216", "and": "\u4e14",
    "the": "", "a": "", "an": "", "is": "\u662f", "are": "\u662f",
    "was": "\u662f", "be": "\u662f", "this": "\u6b64",
}


_COMPILED = []
for _en, _cn in sorted(_RAW, key=lambda p: -len(p[0])):
    _pat = re.escape(_en).replace(r'\{\}', r'(.+?)')
    if _pat.endswith(r'(.+?)'):
        _pat = _pat[:-len(r'(.+?)')] + r'(.+)'
    _COMPILED.append((re.compile(_pat, re.DOTALL), _cn))

def translate(text):
    if not text:
        return text
    result = text
    for _ in range(8):
        matched = False
        for regex, cn_tmpl in _COMPILED:
            m = regex.search(result)
            if m:
                groups = [translate(g) for g in m.groups()]
                cn_text = cn_tmpl.format(*groups) if groups else cn_tmpl
                result = result[:m.start()] + cn_text + result[m.end():]
                matched = True
                break
        if not matched:
            break
    result = _word_fallback(result)
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result


def _word_fallback(text):
    if not text:
        return text

    def _repl(m):
        w = m.group(0)
        return _W.get(w.lower(), w)

    return re.sub(r"[A-Za-z]+(?:-[A-Za-z]+)*", _repl, text)



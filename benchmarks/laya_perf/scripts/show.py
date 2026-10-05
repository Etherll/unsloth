import json, sys
for f in sys.argv[1:]:
    r = json.load(open(f))
    print(f.split("/")[-1], "med", round(r["s_step"], 3), "mean", round(r["s_step_mean"], 3), "first", round(r["first_step_s"], 1),
          "tok/s", round(r["tok_s"]), "res", round(r["peak_res_mib"]), "alloc", round(r["peak_alloc_mib"]), [round(t, 2) for t in r["step_times"][1:]])

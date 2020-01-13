from week_3 import gen_sets

# temps, mes, stds, ims = gen_sets.gen_sets("markov", plot_result=True)
# temps, mes, stds, ims = gen_sets.gen_sets("factor", plot_result=True)
# temps, mes, stds, ims = gen_sets.gen_sets("beta", plot_result=True)

mins = gen_sets.find_min_spread("factor")

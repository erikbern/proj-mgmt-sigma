import math
import numpy
import random
import scipy.stats
import seaborn
from matplotlib import pyplot, ticker

"""
The conditional ROI is the max of
* The derivative of the conditional PDF (i.e. just the original PDF divided by 1 - CDF)
* The max slope of the line between any t' > t and the conditional CDF delta, so (CDF(t') - CDF(t)) / (1 - CDF(t)) / (t' - t)

Note that the second one basically degenerates to the first one if t' convertes to t, which seems like a useful sanity check.
"""


pyplot.style.use('ggplot')
GRAY = (0.5, 0.5, 0.5)


def get_data(sigmas=[0.6, 0.8, 1.0, 1.2, 1.4]):
    n = 10000
    log_ts = numpy.linspace(-25, 25, n)
    for sigma in sigmas:
        cs = scipy.stats.norm.cdf(log_ts, scale=sigma)
        ts = numpy.exp(log_ts)
        yield (n, sigma, ts, cs)


def set_xscale():
    pyplot.xlabel("Time (compared to estimate)")
    pyplot.xlim([0, 7.3])
    pyplot.axvline(x=1, color=GRAY, linestyle=':')
    pyplot.text(1, 0, "Initial estimate", horizontalalignment="center", verticalalignment="bottom", color=GRAY)
    pyplot.gca().xaxis.set_major_formatter(ticker.PercentFormatter(1.0))


# Plot CDFs
pyplot.figure(figsize=(12, 8))
for n, sigma, ts, cs in get_data():
    pyplot.plot(ts, cs, label="$ \sigma = %.1f $" % sigma)
    for t in [.5, 1, 2, 4]:
        print("sigma = %.1f t = %3.1f, cdf = %5.2f%%" % (sigma, t, 100*cs[numpy.argmax(ts > t)]))

pyplot.legend()
set_xscale()
pyplot.ylabel("Cumulative density (0-1)")
pyplot.title("Cumulative density functions for different log-normal distributions")
pyplot.savefig("cdf.png")


# Plot conditional wait time
pyplot.figure(figsize=(12, 8))
data = list(get_data([0.6, 1.2]))
for j, hatch, (n, sigma, ts, cs) in zip([0, 1], ["/", "\\"], data):
    lo_qs = cs + (1 - cs) * 0.25
    hi_qs = cs + (1 - cs) * 0.75
    lo = numpy.exp(scipy.stats.norm.ppf(lo_qs, scale=sigma))
    hi = numpy.exp(scipy.stats.norm.ppf(hi_qs, scale=sigma))
    pyplot.fill_between(ts, lo, hi, color="none", hatch=hatch, edgecolor="C%d" % j, label="$ \sigma = %.1f $" % sigma)

pyplot.legend()
set_xscale()
pyplot.gca().yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
pyplot.ylabel("Additional time to finish")
pyplot.ylim([0, 5])
pyplot.title("Additional time to finish given that you spent some time already")
pyplot.savefig("additional_time.png")


# Plot hazard ratios
pyplot.figure(figsize=(12, 8))
for n, sigma, ts, cs in get_data():
    hs = numpy.zeros(n)
    ln_Ss = numpy.log(1 - cs)
    for i in range(n-1):
        hs[i] = -(ln_Ss[i+1] - ln_Ss[i]) / (ts[i+1] - ts[i])
    pyplot.plot(ts, hs, label="$ \sigma = %.1f $" % sigma)

pyplot.legend()
set_xscale()
pyplot.savefig("hazard_rates.png")


def get_conditional_roi(ts, cs):
    vs = numpy.zeros(n)  # conditional ROIs
    eps = 1e-12
    for i in range(n-1):
        slopes = (cs - cs[i]) / (1 + eps - cs[i]) / (ts - ts[i] + eps)
        vs[i] = numpy.max(slopes[i+1:])
    vs /= vs[0]
    return vs


# Plot conditional ROIs
pyplot.figure(figsize=(12, 8))
for n, sigma, ts, cs in get_data():
    vs = get_conditional_roi(ts, cs)

    # Plot
    p = pyplot.plot(ts, vs, label="$ \sigma = %.1f $" % sigma)
    color = p[0].get_color()

pyplot.legend()
pyplot.ylabel("Marginal ROI\n(initial ROI = 1.0)")
pyplot.axhline(y=1, color=GRAY, linestyle=':')
set_xscale()
pyplot.title("Marginal ROI compared to initial ROI, as project goes on")
pyplot.savefig("marginal_roi.png")

# Plot CDF vs time, but mark the point where we give up
pyplot.figure(figsize=(12, 8))
for n, sigma, ts, cs in get_data():
    vs = get_conditional_roi(ts, cs)

    # Plot
    p = pyplot.plot(ts, cs, label="$ \sigma = %.1f $" % sigma)
    color = p[0].get_color()

    # Find the first point i > 0 such that the value is less than the first point
    i = numpy.argmax(vs < vs[0])
    pyplot.plot(ts[i], cs[i], "o", color=color, markersize=10)
    pyplot.annotate(
        "Stopping point",
        xy=(ts[i], cs[i]),
        xytext=(3, 0.5),
        size=20,
        bbox=dict(fc="white"),
        arrowprops=dict(facecolor="black", shrink=0.1)
    )

    print("sigma = %.1f stop t = %5.2f%%, cdf = %5.2f%%" % (sigma, 100.*ts[i], 100.*cs[i]))

pyplot.legend()
set_xscale()
pyplot.ylabel("Cumulative density (0-1)")
pyplot.title("Cumulative density functions for different log-normal distributions\n"
             "with optimal point of abandoning a project in favor of the next one")
pyplot.savefig("cdf_with_stop.png")

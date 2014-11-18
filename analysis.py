# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from Data import *
import math
import Noise_Analysis as na
import numpy as np
import os
from collections import defaultdict
import scipy.stats as stats
import matplotlib.pyplot as plt
import random as nd
import statsmodels.api as sm
%pylab inline
%load_ext autoreload
%autoreload 2

# <headingcell level=1>

# Setup & helper functions

# <markdowncell>

# For initial analyses, we extract the sequence of outcomes for each game in a given condition (i.e. who got the high payoff, if anyone)

# <codecell>

def get_outcome_seq(conflict_level, cond) :
    max_points = 4 if conflict_level == 'high' else 2
    big_list = []
    for r,d,f in os.walk('./{0}_conflict_{1}/completed'.format(conflict_level,cond)):
        for files in f:
            if files.startswith('game_'):
                ts = []
                destination = r + '/' + files
                d = Data(destination).get_data()
                num_games = d[-1][0]
                red_data, blue_data = na.get_red_data(d), na.get_blue_data(d)
                red_points, blue_points = 0, 0                                                                                  
                for i in range(1,num_games+1) :
                    g = na.get_game_data(d,i)
                    if na.get_red_data(g)[0][-2] == na.get_red_data(g)[-1][-2] - max_points :
                        ts.append('R')
                    elif na.get_blue_data(g)[0][-2] == na.get_blue_data(g)[-1][-2] - max_points :
                        ts.append('B')
                    else :
                        ts.append('T')
                big_list.append(ts)
    return big_list

# <markdowncell>

# Given the outcome sequence for a given condition, we can compute the efficiency and fairness for each game

# <codecell>

def get_efficiency_fairness(outcome_seq) :
    efficiency, fairness = [], []
    # Technically doesn't matter what number we plug in 
    # as long as we normalize correctly, but just to be clear...
    if (len(outcome_seq[0]) == 50) :
        win_payoff = 4 # Set winning payoff as 4 in 'high' condition
    else :
        win_payoff = 2 # Set winning payoff as 2 in 'low' condition
    lose_payoff = 1
    for ts in outcome_seq :
        total_red, total_blue = 0,0
        f_red, f_blue = 0,0
        for element in ts :
            if element == 'R' :
                total_red += win_payoff
                total_blue += lose_payoff
                f_red += 1
            elif element == 'B' :
                total_blue += win_payoff
                total_red += lose_payoff
                f_blue += 1
        # Correct so no dividing by zero (i.e. special case: perfectly fair if they tied every time)...
        if (f_red == 0 and f_blue == 0):
            f_red, f_blue = 1, 1
        efficiency.append((total_red + total_blue) 
                          / ((win_payoff+lose_payoff) * float(len(ts))))
        fairness.append(min(f_red, f_blue) / float(max(f_red, f_blue)))
    return efficiency, fairness 

# <headingcell level=1>

# Import data

# <codecell>

h_dyn = get_outcome_seq("high", "dynamic")
h_bal = get_outcome_seq("high", "ballistic")
l_dyn = get_outcome_seq("low", "dynamic")
l_bal = get_outcome_seq("low", "ballistic")

# <markdowncell>

# Check how many dyads we have for each condition

# <codecell>

print "1 vs. 4 dynamic:\t", len(h_dyn)
print "1 vs. 4 ballistic:\t", len(h_bal)
print "1 vs. 2 dynamic:\t", len(l_dyn)
print "1 vs. 2 ballistic:\t", len(l_bal)

# <markdowncell>

# Compute efficiency and fairness

# <codecell>

h_bal_eff, h_bal_fair = get_efficiency_fairness(h_bal)
h_dyn_eff, h_dyn_fair = get_efficiency_fairness(h_dyn)
l_bal_eff, l_bal_fair = get_efficiency_fairness(l_bal)
l_dyn_eff, l_dyn_fair = get_efficiency_fairness(l_dyn)

# <headingcell level=1>

# Efficiency Analysis

# <markdowncell>

# First, we look at the distribution of efficiency scores for each condition

# <codecell>

plt.suptitle("Efficiency histograms", y = 1.1, fontsize = 14)

plt.subplot(221)
plt.hist(h_bal_eff, normed=True, range=(0,1))
#plt.ylim(0,1)
plt.xlim(0,1)
plt.ylim(0,6)
plt.title("1 v. 4 ballistic")
ax = plt.gca()
plt.setp( ax.get_xticklabels(), visible=False)

plt.subplot(222)
plt.hist(h_dyn_eff, normed=True, range=(0,1))
plt.xlim(0,1)
plt.ylim(0,6)
plt.title("1 v. 4 dynamic")
ax = plt.gca()
plt.setp( ax.get_xticklabels(), visible=False)

plt.subplot(223)
plt.xlim(0,1)
plt.ylim(0,6)
plt.hist(l_bal_eff, normed=True, range=(0,1))
plt.title("1 v. 2 ballistic")

plt.subplot(224)
plt.xlim(0,1)
plt.ylim(0,6)
plt.hist(l_dyn_eff, normed=True, range=(0,1))
plt.title("1 v. 2 dynamic")
plt.show()

# <markdowncell>

# Next, we make an interaction plot comparing *mean efficiency* across the four conditions, and conduct a set of non-parametric statistical tests.

# <codecell>

plt.errorbar([0,1], [np.mean(h_bal_eff), np.mean(h_dyn_eff)], 
             yerr = [np.std(h_bal_eff) / math.sqrt(len(h_bal_eff)), 
                     np.std(h_dyn_eff) / math.sqrt(len(h_dyn_eff))], 
             fmt='-o',  label = "1 v. 4")
plt.errorbar([0,1], [np.mean(l_bal_eff), np.mean(l_dyn_eff)], 
             yerr = [np.std(l_bal_eff) / math.sqrt(len(l_bal_eff)), 
                     np.std(l_dyn_eff) / math.sqrt(len(l_dyn_eff))], 
             fmt = ':x', label = "1 v. 2")
plt.xticks([0,1],['ballistic', 'dynamic'])
plt.xlim(-0.5, 1.5)
plt.ylim(0.6,.9)
plt.legend(loc='lower right')
plt.ylabel("efficiency (mean)")

print "means: ", np.mean(h_bal_eff), np.mean(h_dyn_eff), np.mean(l_bal_eff), np.mean(l_dyn_eff)
print "kruscal-wallis w/ (K, p) =", stats.kruskal(h_dyn_eff,l_dyn_eff,h_bal_eff,l_bal_eff)
print "mann-whitney b/w h_bal and h_dyn w/ (U, p) =", stats.mannwhitneyu(h_bal_eff, h_dyn_eff)
print "mann-whitney b/w l_bal and l_dyn w/ (U, p) =", stats.mannwhitneyu(l_bal_eff, l_dyn_eff)
print "mann-whitney b/w l_bal and h_bal w/ (U, p) =", stats.mannwhitneyu(l_bal_eff, h_bal_eff)
print "mann-whitney b/w l_dyn and h_dyn w/ (U, p) =", stats.mannwhitneyu(l_dyn_eff, h_dyn_eff)

# <markdowncell>

# We find a main effect of the "ballistic-dynamic" manipulation but no effect of the payoff manipulation.
# 
# Since the kruskal-wallis and mann-whitney tests compare *ranks*, not *means*, the previous plot is technically misleading. We need to report *mean rank*, which shows the same effect but has a less interpretable $y$-axis.

# <codecell>

r = stats.rankdata(h_bal_eff + h_dyn_eff + l_bal_eff + l_dyn_eff)
tick1 = len(h_bal_eff)
tick2 = tick1 + len(h_dyn_eff)
tick3 = tick2 + len(l_bal_eff)
h_bal_ranks = (r[:tick1]) 
h_dyn_ranks = (r[tick1 : tick2])
l_bal_ranks = (r[tick2 : tick3])
l_dyn_ranks = (r[tick3 :])
plt.errorbar([0,1], [np.mean(h_bal_ranks), np.mean(h_dyn_ranks)], 
             yerr = [np.std(h_bal_ranks) / math.sqrt(len(h_bal_ranks)), 
                     np.std(h_dyn_ranks) / math.sqrt(len(h_dyn_ranks))], 
             fmt='-o',  label = "1 v. 4")
plt.errorbar([0,1], [np.mean(l_bal_ranks), np.mean(l_dyn_ranks)], 
             yerr = [np.std(l_bal_ranks) / math.sqrt(len(l_bal_ranks)), 
                     np.std(l_dyn_ranks) / math.sqrt(len(l_dyn_ranks))], 
             fmt = '-x', label = "1 v. 2")
plt.xticks([0,1],['ballistic', 'dynamic'])
plt.xlim(-0.5, 1.5)
#plt.ylim(0.6,.9)
plt.legend(loc='upper left')
plt.ylabel("efficiency (mean rank)")

# <headingcell level=1>

# Fairness Analysis

# <markdowncell>

# Note that the distribution of fairness scores is bimodal

# <codecell>

plt.suptitle("Fairness histograms", y = 1.1, fontsize = 14)

plt.subplot(221)
plt.xlim(0,1)
plt.ylim(0,4.5)
plt.hist(h_bal_fair, normed=True, range=(0,1))
plt.title("1 v. 4 ballistic")
ax = plt.gca()
plt.setp( ax.get_xticklabels(),  visible=False)

plt.subplot(222)
plt.xlim(0,1)
plt.ylim(0,4.5)
plt.hist(h_dyn_fair, normed=True, range=(0,1))
plt.title("1 v. 4 dynamic")
ax = plt.gca()
plt.setp( ax.get_xticklabels(),  visible=False)

plt.subplot(223)
plt.xlim(0,1)
plt.ylim(0,4.5)
plt.hist(l_bal_fair, normed=True, range=(0,1))
plt.title("1 v. 2 ballistic")

plt.subplot(224)
plt.xlim(0,1)
plt.ylim(0,4.5)
plt.hist(l_dyn_fair, normed=True, range=(0,1))
plt.title("1 v. 2 dynamic")
plt.show()

# <markdowncell>

# Plot means for visualization, and compare using kruskal-wallis and post-hoc mann-whitney tests

# <codecell>

plt.errorbar([0,1], [np.mean(h_bal_fair), np.mean(h_dyn_fair)], 
             yerr = [np.std(h_bal_fair) / math.sqrt(len(h_bal_fair)), 
                     np.std(h_dyn_fair) / math.sqrt(len(h_dyn_fair))], 
             fmt='-o',  label = "1 v. 4")
plt.errorbar([0,1], [np.mean(l_bal_fair), np.mean(l_dyn_fair)], 
             yerr = [np.std(l_bal_fair) / math.sqrt(len(l_bal_fair)), 
                     np.std(l_dyn_fair) / math.sqrt(len(l_dyn_fair))], 
             fmt = '-x', label = "1 v. 2")
plt.xticks([0,1],['ballistic', 'dynamic'])
plt.xlim(-0.5, 1.5)

print "means:", np.mean(h_bal_fair), np.mean(h_dyn_fair), np.mean(l_bal_fair), np.mean(l_dyn_fair)

plt.legend(loc='best')
plt.ylabel("fairness (mean)")
print "Kruskal Wallis w/ (K, p) =", stats.mstats.kruskalwallis(h_bal_fair,l_bal_fair,h_dyn_fair, l_dyn_fair)
print "mann-whitney: h_bal vs. h_dyn  w/ (U, p) =", stats.mannwhitneyu(h_bal_fair, h_dyn_fair)
print "mann-whitney: h_bal vs. l_dyn w/ (U, p) =", stats.mannwhitneyu(h_bal_fair, l_dyn_fair)
print "mann-whitney: l_bal vs. l_dyn w/ (U,p) =", stats.mannwhitneyu(l_bal_fair, l_dyn_fair)
print "mann-whitney: l_bal vs. h_bal w/ (U, p) =", stats.mannwhitneyu(h_bal_fair, l_bal_fair)

# <markdowncell>

# Again, we technically want to look at mean *rank* rather than the raw means of fairness scores, but we find the same results:

# <codecell>

r = stats.rankdata(h_bal_fair + h_dyn_fair + l_bal_fair + l_dyn_fair)
tick1 = len(h_bal_fair)
tick2 = tick1 + len(h_dyn_fair)
tick3 = tick2 + len(l_bal_fair)
h_bal_ranks = (r[:tick1]) 
h_dyn_ranks = (r[tick1 : tick2])
l_bal_ranks = (r[tick2 : tick3])
l_dyn_ranks = (r[tick3 :])
plt.errorbar([0,1], [np.mean(h_bal_ranks), np.mean(h_dyn_ranks)], 
             yerr = [np.std(h_bal_ranks) / math.sqrt(len(h_bal_ranks)),
                     np.std(h_dyn_ranks) / math.sqrt(len(h_dyn_ranks))],
             fmt = '-o', label = "1 v. 4")
plt.errorbar([0,1], [np.mean(l_bal_ranks), np.mean(l_dyn_ranks)], 
             yerr = [np.std(l_bal_ranks) / math.sqrt(len(l_bal_ranks)),
                     np.std(l_dyn_ranks) / math.sqrt(len(l_dyn_ranks))],
             fmt = '-x', label = "1 v. 2")
plt.xticks([0,1],['ballistic', 'dynamic'])
plt.xlim(-0.5, 1.5)
plt.legend(loc='best')
plt.ylabel("fairness (mean rank)")

# <headingcell level=1>

# Stability Analysis

# <markdowncell>

# Especially in ballistic conditions, people use the random assignment of the payoffs to coordinate. One player will always go top and the other will always go bottom. The stability of this pattern isn't captured by the outcomes, so we need to check the direction sequence as well.

# <codecell>

def get_direction_seq(conflict_level, cond) :
    big_list = []
    max_points = 4 if conflict_level == 'high' else 2
    for r,d,f in os.walk('./{0}_conflict_{1}/completed'.format(conflict_level,cond)):
        for files in f:
            if files.startswith('game_'):
                ts = []
                destination = r + '/' + files
                d = Data(destination).get_data()
                num_games = d[-1][0]
                red_data, blue_data = na.get_red_data(d), na.get_blue_data(d)
                for i in range(1,num_games+1) :
                    g = na.get_game_data(d,i)
                    # If red wins... Count it if they're on top, otherwise note that blue went top
                    if (na.get_red_data(g)[0][-2] == (na.get_red_data(g)[-1][-2] - max_points)) :
                        if (na.get_red_data(g)[0][2]) == 'top' :
                            ts.append('R')
                        else :
                            ts.append('B')
                    # If blue wins... Count it if they're on top, otherwise note that red went top
                    elif (na.get_blue_data(g)[0][-2] == (na.get_blue_data(g)[-1][-2] - max_points)) :
                        if (na.get_blue_data(g)[0][2]) == 'top' :
                            ts.append('B')
                        else :
                            ts.append('R')
                    else :
                        ts.append('T')
                big_list.append(ts)
    return big_list

# <codecell>

h_bal_dir = get_direction_seq("high", "ballistic")
h_dyn_dir = get_direction_seq("high", "dynamic")
l_bal_dir = get_direction_seq("low", "ballistic")
l_dyn_dir = get_direction_seq("low", "dynamic")

# <markdowncell>

# `get_surprise_ts` first builds up a conditional probability distribution giving the likelihood of observing one outcome given the previous $m$ outcomes. Once we have this distribution, it computes Shannon's surprisal for each round of the game:

# <codecell>

def get_surprise_ts (ts, num_back = 2) :
    d = defaultdict(lambda: defaultdict(int))
    surp = []
    # Build conditional distribution
    for i in range(len(ts) - num_back - 1) :
        substring = ts[i:i+num_back + 1]
        d[''.join(substring[:num_back])][substring[-1]] += 1
    # Compute surprisal (-log(p)) for each time step
    for i in range(len(ts) - num_back - 1) :
        substring = ts[i:i+num_back + 1]
        relevant_d = d[''.join(substring[:num_back])]
        # Add 1/k for virtual counts
        surp.append(-np.log2((relevant_d[substring[-1]] + 1/float(3))/float((sum(relevant_d.values()) + 1))))
    return surp

# <markdowncell>

# `get_big_surp` is a wrapper function for `get_surprise_ts`. It builds surprisal time series for both the outcome encoding and the direction encoding, then appends the one that is lower overall to the big list of surprisal time series for all dyads in a given condition.

# <codecell>

def get_big_surp (outcome_seqs, direction_seqs, num_back = 2) :
    big_surp = []
    dir_counter = 0
    for ts1, ts2 in zip(outcome_seqs, direction_seqs) :
        surp1 = get_surprise_ts(ts1, num_back)
        surp2 = get_surprise_ts(ts2, num_back)
        # Take the surprises of the most stable encoding (I'm being generous to ur strategies)
        if np.median(surp1) < np.median(surp2) :
            big_surp.append(surp1) 
        else :
            dir_counter += 1
            big_surp.append(surp2)
    return big_surp, dir_counter

# <markdowncell>

# We calculate the big list of surprisals for each condition:

# <codecell>

num_back = 2
l_dyn_surp = [image for mi in get_big_surp(l_dyn, l_dyn_dir, num_back)[0] 
              for image in mi]
l_bal_surp = [image for mi in get_big_surp(l_bal, l_bal_dir, num_back)[0] 
              for image in mi]
h_dyn_surp = [image for mi in get_big_surp(h_dyn, h_dyn_dir, num_back)[0] 
              for image in mi]
h_bal_surp = [image for mi in get_big_surp(h_bal, h_bal_dir, num_back)[0] 
              for image in mi]

# <headingcell level=3>

# Result: Direction encoding was more stable in Ballistic games than Dynamic games

# <markdowncell>

# One minor question of interest concerns which conditions used the 'direction' encoding and which conditions used the 'outcome' coding. In other words, which conditions generally adopted which convention.
# 
# We test this using a chi-squared contingency test on the counts of how many dyads in each condition adopted each convention. 

# <codecell>

n = len(l_dyn)
l_dyn_freq = get_big_surp(l_dyn, l_dyn_dir)[1]
l_dyn_entry = [l_dyn_freq, n - l_dyn_freq] 

n = len(l_bal)
l_bal_freq = get_big_surp(l_bal, l_bal_dir)[1]
l_bal_entry = [l_bal_freq, n - l_bal_freq]

n = len(h_dyn)
h_dyn_freq = get_big_surp(h_dyn, h_dyn_dir)[1]
h_dyn_entry = [h_dyn_freq, n - h_dyn_freq] 

n = len(h_bal)
h_dyn_freq = get_big_surp(h_bal, h_bal_dir)[1]
h_bal_entry = [h_dyn_freq, n - h_dyn_freq]

low_pooled1, low_pooled2 = l_dyn_entry[0] + l_bal_entry[0], l_dyn_entry[1] + l_bal_entry[1]
high_pooled1, high_pooled2 = h_dyn_entry[0] + h_bal_entry[0], h_dyn_entry[1] + h_bal_entry[1]

#print stats.chi2_contingency([[low_pooled1, low_pooled2],[high_pooled1, high_pooled2]])
#print [[dyn_pooled1, dyn_pooled2],[bal_pooled1, bal_pooled2]]
res = stats.chi2_contingency([[l_dyn_entry, 
                               l_bal_entry],# High Dynamic is more likely to be outcome based than expected
                              [h_dyn_entry, 
                               h_bal_entry]])
print "chi^2 stat =", res[0], "-> p = ", res[1]
print()
expected = res[-1]# High Ballistic is more likely to be outcome based than expected
print "If negative, it means that the observed frequency is smaller than expected"
print np.array([[l_dyn_entry, 
        l_bal_entry], 
       [h_dyn_entry,
        h_bal_entry]]) - expected
print "key:\n", np.array([[["1v2 dyn dir", "1v2 dyn out"],
        ["1v2 bal dir", "1v2 bal out"]], 
       [["1v4 dyn dir", "1v4 dyn out"],
        ["1v4 bal dir", "1v4 bal out"]]])

# <markdowncell>

# We find that the full 2x2x2 matrix of is significantly different from the matrix expected if  independent $\chi^2(4) = 17.18, p = .002.
# 
# We can also plot the binomial probability of settling into a turn-taking equilibrium in each condition:

# <codecell>

n = len(l_dyn)
l_dyn_p = get_big_surp(l_dyn, l_dyn_dir)[1] / float(n)
l_dyn_sd = math.sqrt(l_dyn_p * (1 - l_dyn_p) / float(n))
n = len(l_bal)
l_bal_p = get_big_surp(l_bal, l_bal_dir)[1] / float(n)
l_bal_sd = math.sqrt(l_bal_p * (1 - l_bal_p) / float(n))
n = len(h_dyn)
h_dyn_p = get_big_surp(h_dyn, h_dyn_dir)[1] / float(n)
h_dyn_sd = math.sqrt(h_dyn_p * (1 - h_dyn_p) / float(n))
n = len(h_bal)
h_bal_p = get_big_surp(h_bal, h_bal_dir)[1] / float(n)
h_bal_sd = math.sqrt(h_bal_p * (1 - h_bal_p) / float(n))
plt.errorbar([0,1], [np.mean(h_bal_p), np.mean(h_dyn_p)], 
             yerr = [h_bal_sd, h_dyn_sd],
             fmt='-o',  label = "1 v. 4")
plt.errorbar([0,1], [np.mean(l_bal_p), np.mean(l_dyn_p)], 
             yerr = [l_bal_sd, l_dyn_sd],
             fmt = '-x', label = "1 v. 2")
plt.xticks([0,1],['ballistic', 'dynamic'])
plt.xlim(-0.5, 1.5)
#plt.ylim(.5,1)
plt.legend(loc='best')
plt.ylabel("p(direction encoding most stable)")

# <headingcell level=3>

# Stability histograms

# <codecell>

highest_x = 1
highest_y = 5
plt.subplot(221)
plt.xlim(0,highest_x)
plt.ylim(0,highest_y)
plt.hist(h_bal_surp, normed=True, bins = 20,range = (0, highest_x))
plt.title("1 v. 4 ballistic")
ax = plt.gca()
plt.setp( ax.get_xticklabels(), visible=False)

plt.subplot(222)
plt.xlim(0,highest_x)
plt.hist(h_dyn_surp, normed=True,bins = 20,range = (0, highest_x))
plt.ylim(0,highest_y)
plt.title("1 v. 4 dynamic")
ax = plt.gca()
plt.setp( ax.get_xticklabels(), visible=False)

plt.subplot(223)
plt.xlim(0,highest_x)
plt.ylim(0,highest_y)
plt.hist(l_bal_surp, normed=True,bins = 20, range = (0, highest_x))
plt.title("1 v. 2 ballistic")

plt.subplot(224)
plt.xlim(0,highest_x)
plt.ylim(0,highest_y)
plt.hist(l_dyn_surp, normed=True,bins = 20,range = (0, highest_x))
plt.title("1 v. 2 dynamic")
plt.show()

# <markdowncell>

# Now we construct the interaction plot for surprisal and conduct our kruskal-wallis and mann-whitney tests

# <codecell>

plt.errorbar([0,1], [np.mean(h_bal_surp), np.mean(h_dyn_surp)], 
             yerr = [np.std(h_bal_surp) / math.sqrt(len(h_bal_surp)), 
                     np.std(h_dyn_surp) / math.sqrt(len(h_dyn_surp))], 
             fmt='-o',  label = "1 v. 4")
plt.errorbar([0,1], [np.mean(l_bal_surp), np.mean(l_dyn_surp)], 
             yerr = [np.std(l_bal_surp) / math.sqrt(len(l_bal_surp)), 
                     np.std(l_dyn_surp) / math.sqrt(len(l_dyn_surp))],
                     fmt = ':x', label = "1 v. 2")
plt.xticks([0,1],['ballistic', 'dynamic'])
plt.xlim(-0.5, 1.5)
#plt.ylim(.1,.5)
plt.legend(loc='best')
plt.ylabel("stability (mean surprise)")
plt.show()
print(np.mean(h_bal_surp), np.mean(h_dyn_surp), np.mean(l_bal_surp), np.mean(l_dyn_surp))

print "Kruskal Wallis: w/ (K, p) =", stats.mstats.kruskalwallis(h_bal_surp,l_bal_surp,h_dyn_surp, l_dyn_surp)
print "1 v. 4 ballistic & 1 v. 4 dynamic: mann-whitney (U, p) =", stats.mannwhitneyu(h_bal_surp, h_dyn_surp)
print "1 v. 2 ballistic & 1 v. 2 dynamic: mann-whitney (U, p) =", stats.mannwhitneyu(l_bal_surp, l_dyn_surp)
print "1 v. 4 ballistic & 1 v. 2 dynamic: mann-whitney (U, p) =", stats.mannwhitneyu(h_bal_surp, l_dyn_surp)
print "1 v. 4 ballistic & 1 v. 2 ballistic: mann-whitney (U, p) =", stats.mannwhitneyu(h_bal_surp, l_bal_surp)

# <markdowncell>

# We see that there is an interaction between payoff and environment, with low stakes leading to a *less* stable outcome in the 'dynamic' condition than the 'ballistic' condition, and high stakes leading to a *more* stable outcome in the 'dynamic' condition than the 'ballistic' condition.
# 
# The same interaction is depicted when using mean rank

# <codecell>

r = stats.rankdata(h_bal_surp + h_dyn_surp + l_bal_surp + l_dyn_surp)
tick1 = len(h_bal_surp)
tick2 = tick1 + len(h_dyn_surp)
tick3 = tick2 + len(l_bal_surp)
h_bal_ranks = (r[:tick1]) 
h_dyn_ranks = (r[tick1 : tick2])
l_bal_ranks = (r[tick2 : tick3])
l_dyn_ranks = (r[tick3 :])
plt.errorbar([0,1], [np.mean(h_bal_ranks), np.mean(h_dyn_ranks)], 
             yerr = [np.std(h_bal_ranks) / math.sqrt(len(h_bal_ranks)),
                     np.std(h_dyn_ranks) / math.sqrt(len(h_dyn_ranks))],
             fmt = '-o', label = "1 v. 4")
plt.errorbar([0,1], [np.mean(l_bal_ranks), np.mean(l_dyn_ranks)], 
             yerr = [np.std(l_bal_ranks) / math.sqrt(len(l_bal_ranks)),
                     np.std(l_dyn_ranks) / math.sqrt(len(l_dyn_ranks))],
             fmt = '-x', label = "1 v. 2")
plt.xticks([0,1],['ballistic', 'dynamic'])
plt.xlim(-0.5, 1.5)
plt.title("Stability")
plt.legend(loc='best')
plt.ylabel("stability (mean rank)")

# <headingcell level=3>

# Generate Fig. 2

# <codecell>

plt.subplot(131)
plt.errorbar([0,1], [np.mean(h_bal_eff), np.mean(h_dyn_eff)], 
             yerr = [np.std(h_bal_eff) / math.sqrt(len(h_bal_eff)), 
                     np.std(h_dyn_eff) / math.sqrt(len(h_dyn_eff))], 
             fmt='-o',  label = "High")
plt.errorbar([0,1], [np.mean(l_bal_eff), np.mean(l_dyn_eff)], 
             yerr = [np.std(l_bal_eff) / math.sqrt(len(l_bal_eff)), 
                     np.std(l_dyn_eff) / math.sqrt(len(l_dyn_eff))], 
             fmt = ':x', label = "Low")
matplotlib.rcParams.update({'font.size': 20})
plt.xticks([0,1],['ballistic', 'dynamic'])
plt.xlim(-0.5, 1.5)
plt.ylim(0.6,.9)
plt.ylabel(r"$\overline{\rho_1 + \rho_2}$")
plt.title("Efficiency")


plt.subplot(132)
plt.errorbar([0,1], [np.mean(h_bal_fair), np.mean(h_dyn_fair)], 
             yerr = [np.std(h_bal_fair) / math.sqrt(len(h_bal_fair)), 
                     np.std(h_dyn_fair) / math.sqrt(len(h_dyn_fair))], 
             fmt='-o',  label = "High")
plt.errorbar([0,1], [np.mean(l_bal_fair), np.mean(l_dyn_fair)], 
             yerr = [np.std(l_bal_fair) / math.sqrt(len(l_bal_fair)), 
                     np.std(l_dyn_fair) / math.sqrt(len(l_dyn_fair))], 
             fmt = ':x', label = "Low")
plt.xticks([0,1],['ballistic', 'dynamic'])
plt.xlim(-0.5, 1.5)
plt.legend(loc='upper left')
plt.ylabel(r"$\overline{\min(\rho_1', \rho_2') / \max(\rho_1', \rho_2')}$")
plt.title("Fairness")

plt.subplot(133)
plt.errorbar([0,1], [np.mean(h_bal_surp), np.mean(h_dyn_surp)], 
             yerr = [np.std(h_bal_surp) / math.sqrt(len(h_bal_surp)), 
                     np.std(h_dyn_surp) / math.sqrt(len(h_dyn_surp))], 
             fmt='-o',  label = "High")
plt.errorbar([0,1], [np.mean(l_bal_surp), np.mean(l_dyn_surp)], 
             yerr = [np.std(l_bal_surp) / math.sqrt(len(l_bal_surp)), 
                     np.std(l_dyn_surp) / math.sqrt(len(l_dyn_surp))],
                     fmt = ':x', label = "Low")
plt.xticks([0,1],['ballistic', 'dynamic'])
plt.xlim(-0.5, 1.5)
#plt.legend(loc='best')
plt.ylabel(r"surprisal (bits)")
plt.title("Stability")
matplotlib.rcParams.update({'font.size': 25})
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(32,7)

# <headingcell level=3>

# Generate Supp. Figure 2

# <codecell>

plt.subplot(131)

r = stats.rankdata(h_bal_eff + h_dyn_eff + l_bal_eff + l_dyn_eff)
tick1 = len(h_bal_eff)
tick2 = tick1 + len(h_dyn_eff)
tick3 = tick2 + len(l_bal_eff)
h_bal_ranks = (r[:tick1]) 
h_dyn_ranks = (r[tick1 : tick2])
l_bal_ranks = (r[tick2 : tick3])
l_dyn_ranks = (r[tick3 :])
plt.errorbar([0,1], [np.mean(h_bal_ranks), np.mean(h_dyn_ranks)], 
             yerr = [np.std(h_bal_ranks) / math.sqrt(len(h_bal_ranks)), 
                     np.std(h_dyn_ranks) / math.sqrt(len(h_dyn_ranks))], 
             fmt='-o',  label = "high")
plt.errorbar([0,1], [np.mean(l_bal_ranks), np.mean(l_dyn_ranks)], 
             yerr = [np.std(l_bal_ranks) / math.sqrt(len(l_bal_ranks)), 
                     np.std(l_dyn_ranks) / math.sqrt(len(l_dyn_ranks))], 
             fmt = ':x', label = "low")
plt.xticks([0,1],['ballistic', 'dynamic'])
plt.xlim(-0.5, 1.5)
#plt.ylim(0.6,.9)
plt.title("Efficiency")
plt.ylabel("efficiency (mean rank)")

print np.mean(l_bal_ranks), np.mean(l_dyn_ranks), np.mean(h_bal_ranks), np.mean(h_dyn_ranks)
print "kruscal-wallis significant w/ p =", stats.kruskal(h_dyn_eff,l_dyn_eff,h_bal_eff,l_bal_eff)
print "mann-whitney shows significant diff b/w h_bal and h_dyn w/ p =", stats.mannwhitneyu(h_bal_eff, h_dyn_eff), len(h_bal_eff), len(h_dyn_eff)
print "mann-whitney shows significant diff b/w l_bal and l_dyn w/ p =", stats.mannwhitneyu(l_bal_eff, l_dyn_eff), len(l_bal_eff), len(l_dyn_eff)
print "mann-whitney shows significant diff b/w l_bal and h_bal w/ p =", stats.mannwhitneyu(l_bal_eff, h_bal_eff)
print "mann-whitney shows significant diff b/w l_dyn and h_dyn w/ p =", stats.mannwhitneyu(l_dyn_eff, h_dyn_eff)

plt.subplot(132)
r = stats.rankdata(h_bal_fair + h_dyn_fair + l_bal_fair + l_dyn_fair)
tick1 = len(h_bal_fair)
tick2 = tick1 + len(h_dyn_fair)
tick3 = tick2 + len(l_bal_fair)
h_bal_ranks = (r[:tick1]) 
h_dyn_ranks = (r[tick1 : tick2])
l_bal_ranks = (r[tick2 : tick3])
l_dyn_ranks = (r[tick3 :])
plt.errorbar([0,1], [np.mean(h_bal_ranks), np.mean(h_dyn_ranks)], 
             yerr = [np.std(h_bal_ranks) / math.sqrt(len(h_bal_ranks)),
                     np.std(h_dyn_ranks) / math.sqrt(len(h_dyn_ranks))],
             fmt = '-o', label = "high")
plt.errorbar([0,1], [np.mean(l_bal_ranks), np.mean(l_dyn_ranks)], 
             yerr = [np.std(l_bal_ranks) / math.sqrt(len(l_bal_ranks)),
                     np.std(l_dyn_ranks) / math.sqrt(len(l_dyn_ranks))],
             fmt = ':x', label = "low")
plt.xticks([0,1],['ballistic', 'dynamic'])
plt.xlim(-0.5, 1.5)
plt.legend(loc='best')
plt.title("Fairness")
plt.ylabel("fairness (mean rank)")


plt.subplot(133)
r = stats.rankdata(h_bal_surp + h_dyn_surp + l_bal_surp + l_dyn_surp)
tick1 = len(h_bal_surp)
tick2 = tick1 + len(h_dyn_surp)
tick3 = tick2 + len(l_bal_surp)
h_bal_ranks = (r[:tick1]) 
h_dyn_ranks = (r[tick1 : tick2])
l_bal_ranks = (r[tick2 : tick3])
l_dyn_ranks = (r[tick3 :])
plt.errorbar([0,1], [np.mean(h_bal_ranks), np.mean(h_dyn_ranks)], 
             yerr = [np.std(h_bal_ranks) / math.sqrt(len(h_bal_ranks)),
                     np.std(h_dyn_ranks) / math.sqrt(len(h_dyn_ranks))],
             fmt = '-o', label = "high")
plt.errorbar([0,1], [np.mean(l_bal_ranks), np.mean(l_dyn_ranks)], 
             yerr = [np.std(l_bal_ranks) / math.sqrt(len(l_bal_ranks)),
                     np.std(l_dyn_ranks) / math.sqrt(len(l_dyn_ranks))],
             fmt = ':x', label = "low")
plt.xticks([0,1],['ballistic', 'dynamic'])
plt.xlim(-0.5, 1.5)
plt.title("Stability")
plt.ylabel("surprisal (mean rank)")

matplotlib.rcParams.update({'font.size': 25})
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(32,7)

# <headingcell level=3>

# Generate different stability look-back for Supplementary

# <codecell>

num_back = 3
l_dyn_surp = [image for mi in get_big_surp(l_dyn, l_dyn_dir, num_back)[0] for image in mi]
l_bal_surp = [image for mi in get_big_surp(l_bal, l_bal_dir, num_back)[0] for image in mi]
h_dyn_surp = [image for mi in get_big_surp(h_dyn, h_dyn_dir, num_back)[0] for image in mi]
h_bal_surp = [image for mi in get_big_surp(h_bal, h_bal_dir, num_back)[0] for image in mi]

plt.subplot(131)
r = stats.rankdata(h_bal_surp + h_dyn_surp + l_bal_surp + l_dyn_surp)
tick1 = len(h_bal_surp)
tick2 = tick1 + len(h_dyn_surp)
tick3 = tick2 + len(l_bal_surp)
h_bal_ranks = (r[:tick1]) 
h_dyn_ranks = (r[tick1 : tick2])
l_bal_ranks = (r[tick2 : tick3])
l_dyn_ranks = (r[tick3 :])
plt.errorbar([0,1], [np.mean(h_bal_ranks), np.mean(h_dyn_ranks)], 
             yerr = [np.std(h_bal_ranks) / math.sqrt(len(h_bal_ranks)),
                     np.std(h_dyn_ranks) / math.sqrt(len(h_dyn_ranks))],
             fmt = '-o', label = "high")
plt.errorbar([0,1], [np.mean(l_bal_ranks), np.mean(l_dyn_ranks)], 
             yerr = [np.std(l_bal_ranks) / math.sqrt(len(l_bal_ranks)),
                     np.std(l_dyn_ranks) / math.sqrt(len(l_dyn_ranks))],
             fmt = ':x', label = "low")
plt.xticks([0,1],['ballistic', 'dynamic'])
plt.xlim(-0.5, 1.5)
plt.ylim(5000, 6100)
plt.title("$m = 3$")
plt.ylabel("surprisal (mean rank)")

num_back = 4
l_dyn_surp = [image for mi in get_big_surp(l_dyn, l_dyn_dir, num_back)[0] for image in mi]
l_bal_surp = [image for mi in get_big_surp(l_bal, l_bal_dir, num_back)[0] for image in mi]
h_dyn_surp = [image for mi in get_big_surp(h_dyn, h_dyn_dir, num_back)[0] for image in mi]
h_bal_surp = [image for mi in get_big_surp(h_bal, h_bal_dir, num_back)[0] for image in mi]


plt.subplot(132)
r = stats.rankdata(h_bal_surp + h_dyn_surp + l_bal_surp + l_dyn_surp)
tick1 = len(h_bal_surp)
tick2 = tick1 + len(h_dyn_surp)
tick3 = tick2 + len(l_bal_surp)
h_bal_ranks = (r[:tick1]) 
h_dyn_ranks = (r[tick1 : tick2])
l_bal_ranks = (r[tick2 : tick3])
l_dyn_ranks = (r[tick3 :])
plt.errorbar([0,1], [np.mean(h_bal_ranks), np.mean(h_dyn_ranks)], 
             yerr = [np.std(h_bal_ranks) / math.sqrt(len(h_bal_ranks)),
                     np.std(h_dyn_ranks) / math.sqrt(len(h_dyn_ranks))],
             fmt = '-o', label = "high")
plt.errorbar([0,1], [np.mean(l_bal_ranks), np.mean(l_dyn_ranks)], 
             yerr = [np.std(l_bal_ranks) / math.sqrt(len(l_bal_ranks)),
                     np.std(l_dyn_ranks) / math.sqrt(len(l_dyn_ranks))],
             fmt = ':x', label = "low")
plt.xticks([0,1],['ballistic', 'dynamic'])
plt.xlim(-0.5, 1.5)
plt.ylim(5000, 6100)
plt.title("$m = 4$")
plt.ylabel("surprisal (mean rank)")

num_back = 5
l_dyn_surp = [image for mi in get_big_surp(l_dyn, l_dyn_dir, num_back)[0] for image in mi]
l_bal_surp = [image for mi in get_big_surp(l_bal, l_bal_dir, num_back)[0] for image in mi]
h_dyn_surp = [image for mi in get_big_surp(h_dyn, h_dyn_dir, num_back)[0] for image in mi]
h_bal_surp = [image for mi in get_big_surp(h_bal, h_bal_dir, num_back)[0] for image in mi]


plt.subplot(133)
r = stats.rankdata(h_bal_surp + h_dyn_surp + l_bal_surp + l_dyn_surp)
tick1 = len(h_bal_surp)
tick2 = tick1 + len(h_dyn_surp)
tick3 = tick2 + len(l_bal_surp)
h_bal_ranks = (r[:tick1]) 
h_dyn_ranks = (r[tick1 : tick2])
l_bal_ranks = (r[tick2 : tick3])
l_dyn_ranks = (r[tick3 :])
plt.errorbar([0,1], [np.mean(h_bal_ranks), np.mean(h_dyn_ranks)], 
             yerr = [np.std(h_bal_ranks) / math.sqrt(len(h_bal_ranks)),
                     np.std(h_dyn_ranks) / math.sqrt(len(h_dyn_ranks))],
             fmt = '-o', label = "high")
plt.errorbar([0,1], [np.mean(l_bal_ranks), np.mean(l_dyn_ranks)], 
             yerr = [np.std(l_bal_ranks) / math.sqrt(len(l_bal_ranks)),
                     np.std(l_dyn_ranks) / math.sqrt(len(l_dyn_ranks))],
             fmt = ':x', label = "low")
plt.xticks([0,1],['ballistic', 'dynamic'])
plt.xlim(-0.5, 1.5)
plt.ylim(5000, 6100)
plt.title("$m = 5$")
plt.ylabel("surprisal (mean rank)")

matplotlib.rcParams.update({'font.size': 25})
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(32,7)

# <headingcell level=3>

# Generate pipeline figure

# <codecell>

def plot_sequence (seq, title = "") :
    new_l1 = [t for t in seq]
    for i,s in enumerate(seq):
        if s == 'T' :
            new_l1[i]='black'
        elif s == 'B' :
            new_l1[i] = 'blue'
        elif s == 'R' :
            new_l1[i] = 'red'
    plt.bar(range(len(seq)), np.ones(len(seq)), width = 1, color = new_l1)
    plt.title(title)
    plt.xlabel("Game Number")
    frame1 = plt.gca()
    frame1.axes.get_yaxis().set_visible(False)
    plt.show()

# <codecell>

dir_seq32 = get_direction_seq("high", "dynamic")[32]
out_seq32 = get_outcome_seq("high", "dynamic")[32]
dir_seq43 = get_direction_seq("high", "dynamic")[43]
out_seq43 = get_outcome_seq("high", "dynamic")[43]
plot_sequence(dir_seq32, title = "game 8ef3e directions")
plot_sequence(out_seq32, title = "game 8ef3e outcomes")
plot_sequence(dir_seq43, title = "game b2a9e directions")
plot_sequence(out_seq43, title = "game b2a9e outcomes")

# <codecell>

num_back = 3
print np.mean(get_surprise_ts(out_seq43, num_back=num_back)), np.mean(get_surprise_ts(dir_seq43, num_back=num_back))
plot(get_surprise_ts(out_seq32, num_back=num_back))
plot(get_surprise_ts(out_seq43, num_back=num_back))
plt.show()
plt.clf()
plt.hist(get_surprise_ts(out_seq32), range = (0, 3), histtype = 'step')
plt.hist(get_surprise_ts(out_seq43), range = (0, 3), histtype = 'step')
#h_dyn[32]
#h_dyn[43]

# <headingcell level=3>

# Generate Supp. Figure 6

# <codecell>

plt.hist(l_dyn_surp, range = (0, 4), bins = 20, cumulative = True, normed = True, histtype = 'step', label = 'low + dynamic', linestyle = 'dashed')
plt.hist(l_bal_surp, range = (0, 4), bins = 20,cumulative = True, normed = True, histtype = 'step', label = 'low + ballistic')
plt.hist(h_bal_surp, range = (0, 4), bins = 20,cumulative = True, normed = True, histtype = 'step', label = 'high + ballistic')
plt.hist(h_dyn_surp, range = (0, 4), bins = 20,cumulative = True, normed = True, histtype = 'step', label = 'high + dynamic', color = 'black', ls = 'dotted')

plt.legend(loc='lower right')
plt.xlabel("surprisal (bits)")
plt.ylabel("P(surprisal)")
plt.ylim(0,1)

plt.show()

# <headingcell level=1>

# Peeloff Time Analysis

# <markdowncell>

# First, we classify participants' heading at each point in time as going toward the top or going toward the bottom.

# <codecell>

import numpy.ma as ma
def going_toward_bottom (a) :
    return np.ma.getmaskarray(ma.masked_inside(a, 91, 269))
def going_toward_top (a) :
    return np.ma.getmaskarray(ma.masked_less(a % 271, 90))

# <markdowncell>

# We then track the point at which the lower switches from pursuing the high payoff target to pursuing the low payoff target

# <codecell>

def peeloff_helper (d, i) :
    g = na.get_game_data(d,i)
    red_data, blue_data = na.get_red_data(g), na.get_blue_data(g)
    blue_change = blue_data[-1][-2] - blue_data[0][-2]
    red_change = red_data[-1][-2] - red_data[0][-2]
    if blue_change >= 2 :
        winner_data, loser_data = blue_data, red_data 
        loser_starting = 270
    elif red_change >= 2 :
        winner_data, loser_data = red_data, blue_data
        loser_starting = 90
    else : # They tied, so noone peeled off 
        return 1
    loser_away_mask = (going_toward_top(loser_data['heading']) if loser_data[0][2] == 'bottom' 
                       else (going_toward_bottom(loser_data['heading'])))
    loser_toward_mask = (going_toward_bottom(loser_data['heading']) if loser_data[0][2] == 'bottom' 
                        else (going_toward_top(loser_data['heading'])))
    # If they haven't clicked anything for half the game (or wait so long that they go toward it and lose) don't count it
    no_action_data = loser_data[loser_data['heading'] == loser_starting]
    if (len(no_action_data) > 13) or (len(loser_data[loser_away_mask]) == 0) :
        peeloff = np.nan
    # if they never go toward the high payoff, count as t = 0
    elif len(loser_data[loser_toward_mask]) == 0 :
        peeloff = 0
    # Otherwise, calculate peel-off percentage
    else :   
        winner_mask = winner_data['pointsearned'] > winner_data[0][-2]
        peeloff = loser_data[loser_away_mask][0][1] / float(winner_data[winner_mask][0][1])
    return peeloff

# <markdowncell>

# This is a wrapper function for the helper above, which performs this analysis for every game in the given condition.

# <codecell>

def get_peeloff_times(conflict_level) :
    peeloff_times = np.array([])
    num_games = 50 if conflict_level == 'high' else 60
    for r,d,f in os.walk('./{0}_conflict_dynamic/completed'.format(conflict_level)):
        for files in f:
            if files.startswith('game_'):
                local_ts = []
                destination = r + '/' + files
                d = Data(destination).get_data()
                num_games = d[-1][0]
                for i in range(1,num_games+1) :
                    local_ts.append(peeloff_helper(d, i))
                peeloff_times = np.append(peeloff_times, local_ts)
    return peeloff_times.reshape((len(peeloff_times) / num_games, num_games))

# <markdowncell>

# Compute the peeloff times (pot)

# <codecell>

# testing peeloff function
h_pot = get_peeloff_times('high')
l_pot = get_peeloff_times('low')

# <markdowncell>

# Plot the distribution of peeloff times for each condition and test whether they're the same

# <codecell>

plt.xlim(0,1)
cleaned_l_pot = l_pot[~np.isnan(l_pot)]
cleaned_h_pot = h_pot[~np.isnan(h_pot)]
plt.hist(cleaned_l_pot, bins = 20, range = (0,1))
plt.ylim(0,2300)
plt.title("1 v. 2 dynamic peeloff times")
plt.show()
plt.clf()
plt.xlim(0,1)
plt.ylim(0,2300)
plt.title("1 v. 4 dynamic peeloff times")
plt.hist(cleaned_h_pot, bins = 20, range = (0,1))
plt.show()
print stats.mannwhitneyu(cleaned_l_pot, cleaned_h_pot)
r = stats.rankdata(np.append(cleaned_l_pot, cleaned_h_pot))
print np.mean(r[:len(cleaned_l_pot)]), np.mean(r[len(cleaned_l_pot):])

# <markdowncell>

# Use this function to get the predicted values according to a lowess regression

# <codecell>

import scipy.interpolate as interp
def get_lowess_ts (outcomes, frac) :
    lowess = sm.nonparametric.lowess
    x = range(len(outcomes[0]))
    y = stats.nanmean(outcomes, axis = 0)
    z = lowess(y, x, return_sorted = True, is_sorted = True, frac = frac)
    return z.T[1]

# <markdowncell>

# Use to bootstrap lowess regressions

# <codecell>

choice = random.choice
def bootstrap_lowess (outcomes) :
    counter = 0
    l = []
    for t in outcomes.T :
        # bootstrap new sample
        values = [choice(t) for _ in xrange(len(t))]
        l.append(values)
    return get_lowess_ts(np.array(l).T, 1./3)

def get_bounds (outcomes) :
    big_a = np.zeros((1000,len(outcomes[0])))
    upper_int = []
    lower_int = []
    for i in range(1000) :
        new_curve = bootstrap_lowess (outcomes)
        big_a[i] = new_curve
    for t in big_a.T :
        upper_int.append(sorted(t)[841])
        lower_int.append(sorted(t)[159])
    return (lower_int, upper_int)

# <headingcell level=3>

# Make Fig.3

# <markdowncell>

# Precompute +1SD and -1SD envelopes (gives an interval of what we COULD have observed if we'd constructed our local regression from a slightly different sample)

# <codecell>

l_lower, l_upper = get_bounds(l_pot) 
h_lower, h_upper = get_bounds(h_pot) 

# <markdowncell>

# Plot the nonparametric regression curves for each condition, as well as the +/- 1SD envelope around each curve

# <codecell>

frac = 1./3
z1 = get_lowess_ts(l_pot, frac)
z2 = get_lowess_ts(h_pot, frac)

fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.set_xlabel("Round # (Low)")
ax1.plot(l_lower, 'k-', lw = 1)
ax1.plot(l_upper, 'k-', lw = 1)
low_line, = ax1.plot(z1,'k-', lw = 3, label = 'low')
plt.ylabel("peel off time")
ax1.vlines(9, .15,.55)
ax1.vlines(15, .15, .55)

ax2 = ax1.twiny()
ax2.set_xlabel("Round # (High)")
high_line, = ax2.plot(z2, 'b-', lw = 3, label = 'high')
ax2.plot(h_lower, 'b:')
ax2.plot(h_upper, 'b:')
plt.yticks(np.linspace(.15,.55,num = 9), ["15%", "20%", "25%", "30%", "35%", "40%", "45%", "50%", "55%"])
plt.ylim(.15,.55)
fig.legend((high_line, low_line), ("High", "Low"), loc = (.5,.55) )

# <markdowncell>

# Note that conflict (measured by peel-off time) is *greater* in the 'high' condition than the 'low' condition in the initial window, but this relationship flips in the later window. We now test this observation directly on the data:

# <codecell>

# Gotta filter out nans
l_pot_beg = np.array([item for sublist in [t[:9] for t in l_pot] for item in sublist])
l_pot_beg = list(l_pot_beg[~numpy.isnan(l_pot_beg)])

l_pot_end = np.array([item for sublist in [t[15:] for t in l_pot] for item in sublist])
l_pot_end = list(l_pot_end[~numpy.isnan(l_pot_end)])

h_pot_beg = np.array([item for sublist in [t[:7] for t in h_pot] for item in sublist])
h_pot_beg = list(h_pot_beg[~numpy.isnan(h_pot_beg)])

h_pot_end = np.array([item for sublist in [t[13:] for t in h_pot] for item in sublist])
h_pot_end = list(h_pot_end[~numpy.isnan(h_pot_end)])

plt.errorbar([0,1], [np.mean(h_pot_beg), np.mean(h_pot_end)], 
             yerr = [np.std(h_pot_beg) / math.sqrt(len(h_pot_beg)),
                     np.std(h_pot_end) / math.sqrt(len(h_pot_end))],
             fmt = '-o', label = "High")
plt.errorbar([0,1], [np.mean(l_pot_beg), np.mean(l_pot_end)], 
             yerr = [np.std(l_pot_beg) / math.sqrt(len(l_pot_beg)),
                     np.std(l_pot_end) / math.sqrt(len(l_pot_end))],
             fmt = 'k:', label = "Low")
plt.xticks([0,1],['beginning', 'end'])
plt.xlim(-0.5, 1.5)
#plt.title("(b)")
plt.legend(loc='best')
plt.ylabel("peel off time")
plt.ylim(.15,.5)
plt.yticks([.15,.20,.25,.30,.35,.40,.45,.50],["15%", "20%", "25%", "30%", "35%", "40%", "45%", "50%"])
print "Kruskal Wallis significant w/ p =", stats.mstats.kruskalwallis(l_pot_beg,l_pot_end,h_pot_beg, h_pot_end)
print "Mann-whitney @ Beginning", stats.mannwhitneyu(l_pot_beg, h_pot_beg)
print "n_1 = ", size(l_pot_beg), "n_2 = ", size(h_pot_beg)
print "Mann-whitney @ End", stats.mannwhitneyu(l_pot_end, h_pot_end)
print "n_1 = ", size(l_pot_end), "n_2 = ", size(h_pot_end)
print "Mann-whitney @ High", stats.mannwhitneyu(h_pot_beg, h_pot_end)
print "Mann-whitney @ Low", stats.mannwhitneyu(l_pot_beg, l_pot_end)
matplotlib.rcParams.update({'font.size': 22})
fig = matplotlib.pyplot.gcf()
#fig.set_size_inches(32,7)

# <headingcell level=3>

# Generate Supp Table 3 (what if we chose different windows around the crossover?)

# <codecell>

l_cross_over = 12
h_cross_over = 10
percentage_list = np.linspace(.06, .16, 6)
s_tab3 = np.zeros((6,7))
for i in range(6) :
    percentage = percentage_list[i]
    l_pot_left = l_cross_over - math.floor(60*percentage)
    l_pot_right = l_cross_over + math.floor(60*percentage)
    h_pot_left = h_cross_over - math.floor(50*percentage)
    h_pot_right = h_cross_over + math.floor(50*percentage)

    l_pot_beg = np.array([item for sublist in [t[:l_pot_left] for t in l_pot] for item in sublist])
    l_pot_beg = list(l_pot_beg[~numpy.isnan(l_pot_beg)])
    l_beg_mean = round(np.mean(l_pot_beg),2)
    
    l_pot_end = np.array([item for sublist in [t[l_pot_right:] for t in l_pot] for item in sublist])
    l_pot_end = list(l_pot_end[~numpy.isnan(l_pot_end)])
    l_end_mean = round(np.mean(l_pot_end),2)
    
    h_pot_beg = np.array([item for sublist in [t[:h_pot_left] for t in h_pot] for item in sublist])
    h_pot_beg = list(h_pot_beg[~numpy.isnan(h_pot_beg)])
    h_beg_mean = round(np.mean(h_pot_beg),2)

    h_pot_end = np.array([item for sublist in [t[h_pot_right:] for t in h_pot] for item in sublist])
    h_pot_end = list(h_pot_end[~numpy.isnan(h_pot_end)])
    h_end_mean = round(np.mean(h_pot_end),2)

    kw_p = round(stats.mstats.kruskalwallis(l_pot_beg,l_pot_end,h_pot_beg, h_pot_end)[1],3)
    pval_beg = round(stats.mannwhitneyu(l_pot_beg, h_pot_beg)[1], 3)
    pval_end = round(stats.mannwhitneyu(l_pot_end, h_pot_end)[1], 3)
    s_tab3[i] = [kw_p, l_beg_mean, h_beg_mean, pval_beg, l_end_mean, h_end_mean, pval_end]

print(s_tab3)

# <codecell>

# Gotta filter out nans
l_pot_beg = np.array([item for sublist in [t[:8] for t in l_pot] for item in sublist])
l_pot_beg = list(l_pot_beg[~numpy.isnan(l_pot_beg)])

l_pot_end = [item for sublist in [t[18:] for t in l_pot] for item in sublist]

h_pot_beg = np.array([item for sublist in [t[:7] for t in h_pot] for item in sublist])
h_pot_beg = list(h_pot_beg[~numpy.isnan(h_pot_beg)])

h_pot_end = [item for sublist in [t[10:] for t in h_pot] for item in sublist]
r = stats.rankdata(l_pot_beg + l_pot_end + h_pot_beg + h_pot_end)
tick1 = len(l_pot_beg)
tick2 = tick1 + len(l_pot_end)
tick3 = tick2 + len(h_pot_beg)
l_beg_ranks = (r[:tick1]) 
l_end_ranks = (r[tick1 : tick2])
h_beg_ranks = (r[tick2 : tick3])
h_end_ranks = (r[tick3 :])
print len(l_pot_beg)
print len(l_beg_ranks)
plt.errorbar([0,1], [np.mean(h_beg_ranks), np.mean(h_end_ranks)], 
             yerr = [np.std(h_beg_ranks) / math.sqrt(len(h_beg_ranks)),
                     np.std(h_end_ranks) / math.sqrt(len(h_end_ranks))],
             fmt = '-o', label = "1 v. 4")
plt.errorbar([0,1], [np.mean(l_beg_ranks), np.mean(l_end_ranks)], 
             yerr = [np.std(l_beg_ranks) / math.sqrt(len(l_beg_ranks)),
                     np.std(l_end_ranks) / math.sqrt(len(l_end_ranks))],
             fmt = '-x', label = "1 v. 2")
plt.xticks([0,1],['beginning', 'end'])
plt.xlim(-0.5, 1.5)
plt.title("Stability")
plt.legend(loc='best')
plt.ylabel("peel off time (mean rank)")
print "Kruskal Wallis significant w/ p =", stats.mstats.kruskalwallis(l_pot_beg,l_pot_end,h_pot_beg, h_pot_end)
print "Mann-whitney @ Beginning", stats.mannwhitneyu(l_pot_beg, h_pot_beg)
print "Mann-whitney @ End", stats.mannwhitneyu(l_pot_end, h_pot_end)

#print "1 v. 4 ballistic & 1 v. 4 dynamic: mann-whitney significant with p =", stats.mannwhitneyu(h_bal_surp, h_dyn_surp)
#print "1 v. 2 ballistic & 1 v. 2 dynamic: mann-whitney not significant with p =", stats.mannwhitneyu(l_bal_surp, l_dyn_surp)
#print "1 v. 4 ballistic & 1 v. 2 dynamic: mann-whitney significant with p =", stats.mannwhitneyu(h_bal_surp, l_dyn_surp)
#print "1 v. 4 ballistic & 1 v. 2 ballistic: mann-whitney not significant with p =", stats.mannwhitneyu(h_bal_surp, l_bal_surp)
print np.nanmean(l_beg_ranks), np.nanmean(l_end_ranks)
print np.nanmean(h_beg_ranks), np.mean(h_end_ranks)

# <headingcell level=3>

# Doesn't look like any individual correlations

# <codecell>

l_pot_beg = np.array([item for sublist in [t[:9] for t in l_pot] for item in sublist])
l_pot_beg = list(l_pot_beg[~numpy.isnan(l_pot_beg)])

l_pot_end = np.array([item for sublist in [t[15:] for t in l_pot] for item in sublist])
l_pot_end = list(l_pot_end[~numpy.isnan(l_pot_end)])

h_pot_beg = np.array([item for sublist in [t[:7] for t in h_pot] for item in sublist])
h_pot_beg = list(h_pot_beg[~numpy.isnan(h_pot_beg)])

h_pot_end = np.array([item for sublist in [t[13:] for t in h_pot] for item in sublist])
h_pot_end = list(h_pot_end[~numpy.isnan(h_pot_end)])

# <codecell>

l_beg_means = np.ma.masked_invalid(np.array([np.nanmean(v[:9]) for v in l_pot]))
l_beg_means = np.ma.masked_equal(l_beg_means, 0)
l_z = stats.mstats.zscore(l_beg_means)
l_end_means = np.array([np.nanmean(v[15:]) for v in l_pot])

h_beg_means = np.ma.masked_invalid(np.array([np.nanmean(v[:7]) for v in h_pot]))
h_beg_means = np.ma.masked_equal(h_beg_means, 0)
h_z = stats.mstats.zscore(h_beg_means)
h_end_means = np.array([np.nanmean(v[13:]) for v in h_pot])

l_avg_diff = mean(l_end_means / (l_beg_means+.000001))
l_real_diff = (l_end_means / (l_beg_means+.000001))
plt.hist(l_real_diff)
plt.show()
l_residual = stats.mstats.zscore(l_real_diff - l_avg_diff)
l_m, l_b, l_r_value, l_p_value, std_err = stats.linregress(l_z.filled(fill_value = nanmean(l_z)),
                             l_residual.filled(fill_value = nanmean(l_residual)))

h_avg_diff = mean(h_end_means / (h_beg_means + .000001))
print(h_avg_diff)
h_real_diff = h_end_means / (h_beg_means + .000001)
plt.hist(h_real_diff)
plt.show()
h_residual = stats.mstats.zscore(h_real_diff - h_avg_diff)
h_m, h_b, h_r_value, h_p_value, std_err = stats.linregress(h_z.filled(fill_value = nanmean(h_z)), 
                             h_residual.filled(fill_value = nanmean(h_residual)))
print(l_p_value), (len(l_z))
print(h_p_value),(len(h_z))
plt.subplot(121)
plt.scatter(l_z, l_residual)
plt.text(-2.5,2,"r = {0}, p < .001".format(round(l_r_value, 2)))
plt.plot(l_z, l_m * l_z + l_b, 'b-')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.title("Low condition")
plt.xlabel("avg. beginning POT (z scored)")
plt.ylabel("% decreased (residual)")

plt.subplot(122)
plt.scatter(h_z, h_residual)
plt.text(-2.5,2,"r = {0}, p < .001".format(round(h_r_value, 2)))
plt.ylim(-3,3)
plt.plot(h_z, h_m * h_z + h_b, 'b-')
plt.title("High condition")
plt.xlabel("avg. beginning POT (z scored)")
plt.ylabel("% decreased (residual)")

#plt.show()
matplotlib.rcParams.update({'font.size': 25})
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(20,7)
#plt.ylabel("avg POT t = $25:35$")
#plt.title("1v2 peel off times correlation")

# <codecell>

jitter = 0
x1 = [np.nanmean(dyad[:3]) + random.uniform(-jitter, jitter)  for dyad in h_pot]
x2 = [np.nanmean(dyad[15:40]) + random.uniform(-jitter, jitter) for dyad in h_pot]
maskedx1 = np.ma.array(x1, mask=np.isnan(x1))
maskedx2 = np.ma.array(x2, mask=np.isnan(x2))
print np.ma.cov(maskedx1,maskedx2,rowvar=False,allow_masked=True)
plt.scatter(x1,x2)
plt.plot(range(2), range(2))
plt.xlabel("avg POT $t = 0:2$")
plt.ylabel("avg POT $t = 25:35$")
plt.title("1v4 peel off times correlation")

# <codecell>

jitter = 0
x1 = [np.nanmean(dyad[:3]) + random.uniform(-jitter, jitter)  for dyad in l_pot]
x2 = [np.nanmean(dyad[15:40]) + random.uniform(-jitter, jitter) for dyad in l_pot]
random.shuffle(x1)
maskedx1 = np.ma.array(x1, mask=np.isnan(x1))
maskedx2 = np.ma.array(x2, mask=np.isnan(x2))
print np.ma.cov(maskedx1,maskedx2,rowvar=False,allow_masked=True)
plt.scatter(x1,x2)
plt.plot(range(2), range(2))
plt.xlabel("avg POT $t = 0:2$")
plt.ylabel("avg POT $t = 25:35$")
plt.title("1v4 peel off times correlation")

# <codecell>



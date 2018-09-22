"""
Module for transforming interval data from load_gen.py into bound constraints for contracting Solar + Storage EMSA

Created on May 18, 2018
S+S Infrastructure Development
Advanced Microgrid Solutions, Inc
Written By: Kan Ito
kani@advmicrogrid.com

"""

import scipy.stats

from sklearn.linear_model import LogisticRegression
import sys
from gen_stats import *
from sklearn import preprocessing
from armadasummary import *
matplotlib.use('Agg')    # Necessary when running in a virtualenv, must be before initializing pyplot.
matplotlib.rcParams['path.simplify'] = False
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')    # Other options: seaborn-dark-palette, fivethirtyeight, seaborn-whitegrid
plt.ioff()    # Necessary when running in a virtualenv.
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
from statsmodels.stats import weightstats
import re
from sklearn.metrics import *

def z_test_wrapper(normal_tags, pos, neg):
    # Run Z - Testing on Normally Distributed Tags
    bound_low = []
    bound_up = []
    bound_tags = []
    p_values = []
    tags_no_sig = []
    p_values_all = []
    tags_all = []
    for tag in normal_tags:
        p_value = z_test_two_sample(pos[tag], neg[tag], tag)
        p_values_all.append(p_value)
        tags_all.append(tag)
        # Statistical Significance Test, if yes, obtain confidence intervals
        if p_value < sig_level:
            p_values.append(p_value)
            CI_pos = scipy.stats.norm.interval(0.95, loc=pos[tag].mean(), scale=pos[tag].std())
            CI_neg = scipy.stats.norm.interval(0.95, loc=neg[tag].mean(), scale=neg[tag].std())
            low, high = get_ci_range(CI_pos, CI_neg, tag)
            # If CI splits in two
            if isinstance(low, list):
                logging.info("double CI")
                bound_low.append(low[0])
                bound_up.append(high[0])
                bound_tags.append(tag)
                bound_low.append(low[1])
                bound_up.append(high[1])
                bound_tags.append(tag)
            elif low == None:
                logging.info("Confidence interval pair do not yield good bounding structure for: {}".format(tag))
            else:
                bound_low.append(low)
                bound_up.append(high)
                bound_tags.append(tag)
        else:
            # Savings to Bound Later
            tags_no_sig.append(tag)
    norm_test_results = pd.DataFrame(
        {
            "tag": tags_all,
            "p_value": p_values_all
        }
    )
    bounds_df_norm = pd.DataFrame(
        {
            "bound_low": bound_low,
            "bound_up": bound_up,
            "tag": bound_tags,
            "significance level": p_values
        }
    )
    bounds_collapsed_norm = convert_monthly_bounds(bounds_df_norm)

    return norm_test_results, bounds_df_norm, bounds_collapsed_norm, tags_no_sig

def t_test_wrapper(non_normal_tags, pos, neg):
    # Run T - Testing on non-Normal Samples
    bound_low_t = []
    bound_up_t = []
    bound_tags_t = []
    p_values_t = []
    tags_all_t = []
    p_values_all_t = []
    for tag in non_normal_tags:
        p_value_t = t_test_two_sample(pos[tag], neg[tag], tag)
        p_values_all_t.append(p_value_t)
        tags_all_t.append(tag)
        # Get confidence interval
        if p_value_t < sig_level:
            p_values_t.append(p_value_t)
            CI_pos = scipy.stats.t.interval(0.95, df=pos[tag].count() - 1, loc=pos[tag].mean(), scale=pos[tag].std())
            CI_neg = scipy.stats.t.interval(0.95, df=neg[tag].count() - 1, loc=neg[tag].mean(), scale=neg[tag].std())

            low, high = get_ci_range(CI_pos, CI_neg, tag)
            # If CI splits in two
            if isinstance(low, list):
                logging.info("double CI")
                bound_low_t.append(low[0])
                bound_up_t.append(high[0])
                bound_tags_t.append(tag)
                bound_low_t.append(low[1])
                bound_up_t.append(high[1])
                bound_tags_t.append(tag)
            elif low == None:
                logging.info("Confidence interval pair do not yield good bounding structure for: {}".format(tag))
            else:
                bound_low_t.append(low)
                bound_up_t.append(high)
                bound_tags_t.append(tag)
        else:
            tags_no_sig.append(tag)

    t_test_results = pd.DataFrame(
        {
            "tag": tags_all_t,
            "p_value": p_values_all_t
        }
    )
    bounds_df_t = pd.DataFrame(
        {
            "bound_low": bound_low_t,
            "bound_up": bound_up_t,
            "tag": bound_tags_t,
            "significance level": p_values_t
        }
    )
    bounds_collapsed_t = convert_monthly_bounds(bounds_df_t)

    return t_test_results, bounds_df_t, bounds_collapsed_t, tags_no_sig

def non_statistical_bound(tags_no_sig, pos, neg):
    bound_low_nonstat = []
    bound_up_nonstat = []
    bound_tags_nonstat = []
    unbounded_tags = []

    for tag in tags_no_sig:
        pos_interval = (pos[tag].quantile(.25), pos[tag].quantile(.75))
        neg_interval = (neg[tag].quantile(.25), neg[tag].quantile(.75))
        low, high = get_ci_range(pos_interval, neg_interval, tag)
        # If CI splits in two
        if isinstance(low, list):
            logging.info("double CI")
            bound_low_nonstat.append(low[0])
            bound_up_nonstat.append(high[0])
            bound_tags_nonstat.append(tag)
            bound_low_nonstat.append(low[1])
            bound_up_nonstat.append(high[1])
            bound_tags_nonstat.append(tag)
        elif low == None:
            logging.info("Problem creating NON Statistical Bounds for: {}".format(tag))
            unbounded_tags.append(tag)
        else:
            bound_low_nonstat.append(low)
            bound_up_nonstat.append(high)
            bound_tags_nonstat.append(tag)

    bounds_df_nonstat = pd.DataFrame(
        {
            "bound_low": bound_low_nonstat,
            "bound_up": bound_up_nonstat,
            "tag": bound_tags_nonstat,
        }
    )
    unbounded_tags_df = pd.DataFrame(
        {
            "tag": unbounded_tags
        }
    )
    bounds_collapsed_nonstat = convert_monthly_bounds(bounds_df_nonstat)

    return bounds_df_nonstat, bounds_df_nonstat, bounds_collapsed_nonstat, unbounded_tags_df

def calc_conf_matrix_perf_v1(bound_df, baseline_df, bounding_measures, master_data_df):
    # initialize
    y_true = []
    y_pred = []
    # For each model run
    for index, model_run in master_data_df.iterrows():
        for col in master_data_df.columns:
            if col[0:20] == 'Utility Import Value':
                temp_measure = re.search("_(.*)", col).group(0)[1:]
                # Check if bounded measure
                if temp_measure in bound_df['measure'].tolist():
                    # Hit MGCS?
                    # compute month boundary, convert to derived from baseline
                    temp_low = bound_df[bound_df['measure'] == temp_measure]['bound_low_pct_diff'].values[0] / 100
                    low = input_base_stats_df[input_base_stats_df['tag'] == col]['Baseline'].values[0] * (1 - temp_low)
                    temp_up = bound_df[bound_df['measure'] == temp_measure]['bound_up_pct_diff'].values[0] / 100
                    up = input_base_stats_df[input_base_stats_df['tag'] == col]['Baseline'].values[0] * (1 + temp_up)
                    # Hit MGCS
                    if model_run['PF_totalsavings'] >= savings_threshold:
                        if model_run[col] <= up and model_run[col] > low:
                            y_true.append(1)
                            y_pred.append(1)
                        else:
                            y_true.append(1)
                            y_pred.append(0)
                    # Did not hit MGCS
                    else:
                        if model_run[col] <= up and model_run[col] > low:
                            y_true.append(0)
                            y_pred.append(1)
                        else:
                            y_true.append(0)
                            y_pred.append(0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    logging.info("True Neg {}, False Pos {} False Neg {}, True Pos {}".format(tn,fp,fn,tp))
    acc = accuracy_score(y_true, y_pred)  # Accuracy Score
    prec = precision_score(y_true, y_pred)  # Precision Score
    err = hamming_loss(y_true, y_pred)  # Error Rate / Hamming Loss
    sn = recall_score(y_true, y_pred)  # Sensitivity / Recall Score
    roc = roc_auc_score(y_true, y_pred)  # Receiver Operating Characteristic
    sp = float(tn) / (tn + fp)  # specificity
    fpr = float(fp) / (tn + fp)  # false positive rate
    logging.info("Error Rate: {}".format(err))
    logging.info("Accuracy: {}".format(acc))
    logging.info("Sensitivity: {}".format(sn))
    logging.info("Specificity: {}".format(sp))
    logging.info("Precision: {}".format(prec))
    logging.info("Receiver Operator Char: {}".format(roc))
    logging.info("False Positive Rate: {}".format(fpr))
    bound_calc = {
        "Performance Attribute": ["TP", "FN", "FP", "TN", "N", "P", "Error Rate", "Accuracy", "Sensitivity",
                                  "Specificity", "Precision", "False Positive Rate"],
        "value": [tp, fn, fp, tn, (tn + fp), (tp + fn), err, acc, sn, sp, prec, fpr]
    }
    bound_calc_df = pd.DataFrame(bound_calc)
    return bound_calc_df

def calc_conf_matrix_perf_v2(bound_df, baseline_df, bounding_measures, master_data_df):
    #initialize
    y_true = []
    y_pred = []
    TN = False
    for index,model_run in master_data_df.iterrows():
        for col in master_data_df.columns:
            if col[0:20] == 'Utility Import Value':
                temp_measure = re.search("_(.*)", col).group(0)[1:]
                # Check if bounded measures
                if temp_measure in bound_df['measure'].tolist():
                    # compute month boundary, convert to derived from baseline
                    temp_low = bound_df[bound_df['measure'] == temp_measure]['bound_low_pct_diff'].values[0] / 100
                    low = input_base_stats_df[input_base_stats_df['tag'] == col]['Baseline'].values[0] * (
                        1 - temp_low)
                    temp_up = bound_df[bound_df['measure'] == temp_measure]['bound_up_pct_diff'].values[0] / 100
                    up = input_base_stats_df[input_base_stats_df['tag'] == col]['Baseline'].values[0] * (
                        1 + temp_up)
                    # Did not hit MGCS
                    if model_run['PF_totalsavings'] <= savings_threshold:
                        # True Negative
                        if model_run[col] >= up and model_run[col] > low:
                            TN = True
        if model_run['PF_totalsavings'] <= savings_threshold:
            if TN:
                y_true.append(0)
                y_pred.append(0)
                TN = False
            else:
                y_true.append(0)
                y_pred.append(1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    logging.info("True Neg {}, False Pos {} False Neg {}, True Pos {}".format(tn, fp, fn, tp))
    sp = float(tn) / (tn + fp)  # specificity
    fpr = float(fp) / (tn + fp)  # false positive rate
    logging.info("Specificity: {}".format(sp))
    logging.info("False Positive Rate: {}".format(fpr))
    bound_calc = {
        "Performance Attribute": ["TP", "FN", "FP", "TN", "N", "P",
                                  "Specificity", "False Positive Rate"],
        "value": [tp, fn, fp, tn, (tn + fp), (tp + fn), sp,  fpr]
    }
    bound_calc_df = pd.DataFrame(bound_calc)
    return bound_calc_df

def convert_monthly_bounds(monthly_bounds_df):
    # Flatten Monthly into Annual Bound, losing month specificity here
    annual_low_list = []
    annual_up_list = []
    measure_list = []
    month_count = []
    monthly_bounds_df['measure'] = monthly_bounds_df['tag'].str.extract(pat=r"_(.*)")
    for measure in monthly_bounds_df['measure'].unique():
        filtered = monthly_bounds_df[monthly_bounds_df['measure'] == measure]
        annual_low_list.append(filtered['bound_low'].min())  # taking the extremes of up to 12 months
        annual_up_list.append(filtered['bound_up'].max())  # taking the extremes of up to 12 months
        month_count.append(filtered['bound_low'].count())
        measure_list.append(measure)
    bounds_annual = pd.DataFrame(
        {
            "bound_low": annual_low_list,
            "bound_up": annual_up_list,
            "measure": measure_list,
            "month_count": month_count
        }
    )
    return bounds_annual

def get_ci_range(CI_pos, CI_neg, tag):
    # B: Breach
    """This Function handles transforming a pair of confidence intervals into a bound(s)
    This requires some QA... Conditionals to find where bounds lie in the spectrum """

    # Scenario 1: separate
    if CI_pos[1] > CI_neg[0] or CI_neg[1] > CI_pos[0]:
        low, high = [CI_neg[0], CI_neg[1]]

    # Scenario 2: B right hand side
    elif CI_pos[0] > CI_neg[0] and CI_pos[1] > CI_neg[1]:
        low, high = [CI_neg[0], CI_pos[0]]

    # Scenario 3: B left hand side
    elif CI_pos[0] < CI_neg[0] and CI_pos[1] < CI_neg[1]:
        low, high = [CI_pos[1], CI_neg[1]]

    # Scenario 4: splits in two B in middle
    # will return a list instead of float64
    elif CI_pos[0] > CI_neg[0] and CI_pos[1] < CI_neg[1]:
        logging.info("WARNING: SPLIT Boundaries for: {}".format(tag))
        low = []
        high = []
        # left side
        low.append(CI_neg[0])
        high.append(CI_pos[0])
        # right side
        low.append(CI_pos[1])
        high.append(CI_neg[1])

    # Scenario 5: NB right
    elif CI_pos[0] < CI_neg[0] and CI_neg[0] < CI_pos[1]:
        low, high = [CI_neg[0], CI_pos[1]]

    # Scenario 6: NB left
    elif CI_pos[1] > CI_neg[1] and CI_neg[1] > CI_pos[0]:
        low, high = [CI_neg[0], CI_neg[1]]

    # Scenario 7: NB in middle
    elif CI_neg[0] > CI_pos[0] and CI_neg[1] < CI_pos[1]:
        low, high = [CI_neg[0], CI_neg[1]]

    else:
        # CHANGE THIS; CURRENTLY PLACEHOLDER
        return None, None
    return low, high

def get_analysis_results(analysis_id):
    path = str(ams_analytics_path) + "/web_model/"+str(analysis_id)+"_summary.xlsx"
    armada_results = pd.read_excel(path)
    armada_results = armada_results[
        ['analysisTitle', 'portfolioanalysisId', 'portfolioname', 'portfolioportfolioId', 'preBatteryTotal',
         'postBatteryTotal']]
    armada_results['PF_totalsavings'] = armada_results['preBatteryTotal'] - armada_results['postBatteryTotal']
    armada_results = armada_results.set_index('portfolioportfolioId')
    armada_results['savings_unacceptable'] = armada_results['PF_totalsavings'] < savings_threshold
    return armada_results

def visualize_analysis_results(armada_results,tag,analysis_id):
    fig = plt.figure()
    fig.set_figheight(4)
    fig.set_figwidth(12)
    histogram = plt.subplot(1, 2, 1)
    histogram.hist(armada_results)
    norm_prob_plot = plt.subplot(1, 2, 2)
    scipy.stats.probplot(armada_results, plot=norm_prob_plot)
    plt.ylabel(tag)
    plt.title(str(tag)+" With mean " + str(armada_results.mean()))
    plt.savefig("Analysis _{}_Visuals_{}.png".format(analysis_id,tag), bbox_inches='tight')
    plt.close()
    return

def split_between_savings_threshold(savings_threshold,armada_results):
    positives = armada_results['PF_totalsavings'] < savings_threshold
    positives = armada_results[positives]
    negatives = armada_results['PF_totalsavings'] >= savings_threshold
    negatives = armada_results[negatives]
    return positives, negatives

def get_interval_attributes(analysis_id):
    all_stats_df = pd.read_excel(test_bounds_dict['load_sheets']['load_sheet_standard_test1'])
    all_tags = all_stats_df.columns
    tags = []
    for tag in all_tags:
        if tag[0:20] == "Utility Import Value":
            tags.append(tag)
    return tags, all_stats_df

def normal_test(data,tag,sig_level):
    k2, p = scipy.stats.normaltest(data)
    return p

def z_test_two_sample(sample1, sample2, tag):
    z_score, p_value = weightstats.ztest(sample1, sample2)
    return p_value

def t_test_two_sample(sample1, sample2, tag):
    t_score, p_value, sigma = weightstats.ttest_ind(sample1, sample2, alternative='two-sided')
    return p_value

def log_reg(x, data_tags,y):
    # FLOATING POINT PROBLEM: multiply everything by 100, then int()
    x = (x[data_tags]*100).apply(np.int64)
    y = (y*100).apply(np.int64)
    lr = LogisticRegression()
    lr.fit(x, y)
    # logging.info("Running Logistic Regression on: {}".format(x.columns))
   # logging.info("Log Reg Coefficients: {}".format(lr.coef_))
   # logging.info("Intercept: {}".format(lr.intercept_))
    return lr

class LoadBoundType(object):
    """ An LoadBoundType object contains information related to the measure properties of a Load Bound.
        Specifically this class exists to ease prototyping of additional formats of bound modules.
        There are numerous formats of contract bounds, each including different statistical measures, duration of bound,
        bound metric (ex. % or raw number), etc. Rather than storing these format properties in the LoadBound object, the
        LoadBound object inherits from a LoadBoundType that stores this format data.
        """

    INTERVAL_DICT = {
        "monthly": 30,
        "annual": 365
    }

    BOUND_DETAILS_DICT = {
        "default": {
            'measures': ['sum', 'load_factor', 'max', 'hours_below_p3', 'hours_below_p5', 'hours_below_p10', 'hours_below_p20'],
            'bound_key_unit': 'percent_difference'
        },
        "option1_hard_bound": {
            #'skew', 'mad', 'kurt','total_kwh',
            'measures': ['sum', 'load_factor', 'max', 'hours_below_p3', 'hours_below_p5', 'hours_below_p10', 'hours_below_p20'],
            'bound_key_unit': 'raw_difference'
        }
    }

    def __init__(self, name=None, freq=None, type='default'):
        self.name = name
        self.type = type
        self.interval = self.INTERVAL_DICT[freq]
        self.freq = freq
        self.bound_key_unit = self.BOUND_DETAILS_DICT[type]['bound_key_unit']
        self.bound_measures = self.BOUND_DETAILS_DICT[type]['measures']


class LoadBound(object):

    TEST_DICT = {
        "test1": {
            'load_bound_type': 'default'
        },
        "test2": {
            'load_bound_type': 'default'
        }
    }

    def __init__(self, name=None, load_bound_type=None, batch_to_set_bound=None, lower_savings_bound=None):
        self.name = name
        self.load_bound_type = load_bound_type
        self.batch_to_set_bound = batch_to_set_bound
        self.lower_savings_bound = lower_savings_bound
        self.all_bound_stats_df = None
        self.bound = None

    def gen_bound_features(self, base_stats_df):

        merged_df = copy.deepcopy(self.batch_to_set_bound.portfolio_runs_load_merged_df)
        bounded_merged_all_df = merged_df[merged_df.totalSavings > self.batch_to_set_bound.lower_savings_bound]
        bound_max_df = pd.DataFrame()
        bound_min_df = pd.DataFrame()
        bound_base_df = pd.DataFrame()

        for measure in self.load_bound_type.bound_measures:
            bound_max_df = bound_max_df.append(pd.DataFrame(bounded_merged_all_df[bounded_merged_all_df.filter(like=measure).columns].max(), columns=['bound_stat_max']))
            bound_min_df = bound_min_df.append(pd.DataFrame(bounded_merged_all_df[bounded_merged_all_df.filter(like=measure).columns].min(), columns=['bound_stat_min']))
            bound_base_df = bound_base_df.append(pd.DataFrame(base_stats_df[base_stats_df.index.str.contains(measure)]))

        bound_base_df.rename(columns={bound_base_df.columns.values[0]:'bound_stat_base'}, inplace=True)

        all_bound_stats_df = pd.concat([bound_max_df, bound_min_df, bound_base_df], axis=1)

        all_bound_stats_df['top_bound_raw_difference'] = all_bound_stats_df['bound_stat_max'] - all_bound_stats_df['bound_stat_base']
        all_bound_stats_df['top_bound_percent_difference'] = all_bound_stats_df['top_bound_raw_difference'] / abs(all_bound_stats_df['bound_stat_base'])
        all_bound_stats_df['bottom_bound_raw_difference'] = all_bound_stats_df['bound_stat_min'] - all_bound_stats_df[
            'bound_stat_base']
        all_bound_stats_df['bottom_bound_percent_difference'] = all_bound_stats_df['bottom_bound_raw_difference'] / abs(all_bound_stats_df['bound_stat_base'])        
        self.all_bound_stats_df = all_bound_stats_df
        self.all_bound_stats_df.to_excel(str(analysis_id) + "_Prototype_I_"+str(self.name)+'_all_bound_stats.xlsx')
        return all_bound_stats_df

    def set_bounds(self):

        bound = {
            'measures':{},
            'bound_key_unit': self.load_bound_type.bound_key_unit
        }
        for measure in self.load_bound_type.bound_measures:
            measure_df = pd.DataFrame(self.all_bound_stats_df[self.all_bound_stats_df.index.str.contains(measure)])
            measure_df = measure_df[np.abs(measure_df - measure_df.mean()) <= (2*measure_df.std())]
            measure_df = measure_df.dropna(axis=0,how='any')
            if measure == 'load_factor':
                measure_df = measure_df[measure_df['bound_stat_base']>=0]

            bound['measures'][measure] = [measure_df['bottom_bound_{}'.format(self.load_bound_type.bound_key_unit)].median(), measure_df['top_bound_{}'.format(self.load_bound_type.bound_key_unit)].median()]

        self.bound = bound
        logging.info("Prototype I Bounds: {}".format(self.bound))
        return bound

    def output_bounds(self):
        output_bounds_df = pd.DataFrame(self.bound)
        output_bounds_df.to_excel(str(analysis_id)+'_Prototype_I_final_bounds.xlsx'.format(self.name))

def reformat_df(df):
    """ Function takes in dataframe of raw interval data output by Armada and returns a interval dataset re-indexed by Timestamp
        with a month column added.
        df                   | string, name of product vendor
    """
    if type(df) is gen_stats.LoadDataset:
        df = df.dataset
    # Re-index interval data by Timestamp.
    df = df.set_index('Interval End')
    df.index.name = "Timestamp"
    df['Interval End'] = df.index
    df.index = pd.to_datetime(df.index)

    # Create a month column
    df['month'] = df.index.month

    return df

def gen_base_stats(baseline_data, load_bound):
    """ Function takes in baseline interval data and calculates monthly statistics given this data. Returns a flattened table of load statistics.
        Uses Load Bound to determine frequency by which load stats should be generated #TODO note whether or not this load bound logic should be here or contained elsewhere

        baseline_data        | string, name of product vendor
        load_bound           | string, name of specific battery technology_id,
    """

    baseline_data_df = reformat_df(baseline_data[0])

    cum_stat_df = pd.DataFrame()

    for month in baseline_data_df.month.unique():
        base_data_chunk = baseline_data_df[baseline_data_df.month == month]
        base_data_stats = calc_single_load_stats(base_data_chunk, load_bound.load_bound_type.bound_measures,
                                                 base_data_chunk).transpose()
        base_data_stats.rename(columns={'Power.Grid kW': 'Utility Import Value {}'.format(month)}, inplace=True)

        cum_stat_df = cum_stat_df.append(base_data_stats)

    return cum_stat_df

def flatten_monthly_stats_flipped(df):
    """ Function takes in dataframe of interval data statistics, then flattens and transposes it. Returns transposed, flattened dataframe of statistics
        df                   | string, name of product vendor
    """
    # Flattened statistics dataframe
    summarized_stats = df.stack().reset_index()
    summarized_stats['combined_col_name'] = summarized_stats['level_1'] + "_" + summarized_stats['level_0']
    summarized_stats = summarized_stats.set_index('combined_col_name')
    summarized_stats = summarized_stats.drop('level_0', 1)
    summarized_stats = summarized_stats.drop('level_1', 1)

    # Transposing statistics dataframe
    summarized_stats = summarized_stats.transpose()

    # Adding in additional naming columns to identify source of statiistics
    # summarized_stats['load_name'] = name
    # summarized_stats['sens_type'] = sens_type

    return summarized_stats

def test_bound_batch(baseline_data, test_list, test_batch, load_bound):
    """ Function takes in dataframe of interval data statistics, then flattens and transposes it. Returns transposed, flattened dataframe of statistics
        baseline_data          | string, name of product vendor
        test_list              | string, name of product vendor
        test_batch             | string, name of product vendor
        load_bound             | string, name of product vendor
    """

    baseline_data_df = reformat_df(baseline_data[0])

    test_output_df = pd.DataFrame()

    for i, test_data in enumerate(test_list):

        logging.info("This dataset is called {}".format(test_data.name))
        test_data_df = reformat_df(test_data)

        if load_bound.load_bound_type.freq == "monthly":
            stat_box_df = pd.DataFrame()

            for month in baseline_data_df.month.unique():
                month_stat_df = pd.DataFrame()

                base_data_chunk = baseline_data_df[baseline_data_df.month == month]
                base_data_stats = calc_single_load_stats(base_data_chunk, load_bound.load_bound_type.bound_measures,
                                                         base_data_chunk).transpose()
                base_data_stats.rename(columns={'Power.Grid kW': 'Utility Import Value Base {}'.format(month)},
                                       inplace=True)

                month_stat_df = month_stat_df.append(base_data_stats)

                test_data_chunk = test_data_df[test_data_df.month == month]
                test_data_stats = calc_single_load_stats(test_data_chunk, load_bound.load_bound_type.bound_measures,
                                                         base_data_chunk).transpose()
                test_data_stats.rename(columns={'Power.Grid kW': 'Utility Import Value Test {}'.format(month)},
                                       inplace=True)
                month_stat_df = pd.concat([month_stat_df, test_data_stats], axis=1)

                month_stat_df['raw_difference {}'.format(month)] = month_stat_df[
                                                                       'Utility Import Value Base {}'.format(
                                                                           month)] - month_stat_df[
                                                                       'Utility Import Value Test {}'.format(month)]
                month_stat_df['percent_difference {}'.format(month)] = month_stat_df[
                                                                           'raw_difference {}'.format(month)] / \
                                                                       month_stat_df[
                                                                           'Utility Import Value Base {}'.format(
                                                                               month)]

                stat_box_df = stat_box_df.append(month_stat_df.transpose())

            bound_key_unit = load_bound.load_bound_type.bound_key_unit

            bound_box_df = stat_box_df[stat_box_df.index.str.contains(bound_key_unit)]

            for key in load_bound.bound['measures'].keys():
                bound_box_df['top_bound {}'.format(key)] = np.where(
                    bound_box_df[key] <= load_bound.bound['measures'][key][1], True, False)
                bound_box_df['bottom_bound {}'.format(key)] = np.where(
                    bound_box_df[key] >= load_bound.bound['measures'][key][0], True, False)

                if key == 'sum':
                    bound_box_df['valid {}'.format(key)] = bound_box_df['bottom_bound {}'.format(key)]
                elif key == 'load_factor':
                    bound_box_df['valid {}'.format(key)] = bound_box_df['top_bound {}'.format(key)]
                else:
                    bound_box_df['valid {}'.format(key)] = bound_box_df['top_bound {}'.format(key)] & bound_box_df[
                        'bottom_bound {}'.format(key)]

            bound_box_df['base_dataset'] = baseline_data[0].name
            bound_box_df['test_dataset'] = test_data.name
            bound_box_df = bound_box_df.reset_index()

        elif load_bound.load_bound_type == "custom":
            logging.info("This is a custom interval")

        test_output_df = test_output_df.append(bound_box_df)
        merged_bound_box_df = pd.merge(test_output_df.reset_index(), test_batch.portfolio_runs_load_merged_df[
            ['load_name', 'portfolioname', 'totalSavings']], left_on='test_dataset',
                                       right_on='load_name')
        merged_bound_box_df['lower_savings_bound'] = load_bound.lower_savings_bound
        merged_bound_box_df['greater_than_savings_bound'] = np.where(
            merged_bound_box_df['totalSavings'] >= load_bound.lower_savings_bound, True, False)

    test_output_df.to_csv('test_{}_results_bounded.csv'.format(test_batch.name))
    merged_bound_box_df.to_csv('testresults_{}_csv'.format(test_batch.name))



"""
        PROTOTYPE I INPUTS
        
        savings_threshold     | int,    Perfect Foresight Cutoff to Bound off of
        run_prototype_I       | bool,   True to run prototype I
        bound_type            | string, 2 options: 'default' or 'option1_hard_bound' 
                                        for percentage diff bounds or raw difference 
                                        bounds respectively
        base_path             | string, directory for baseline interval file
        baseline_datasets_list| list of strings,
                                        list of files for baseline interval data
        bound_gen_batch       | int,    primary analysis ID from Armada to bound containing low probability events
        precontract_batch     | int,    analysis ID that contains only standard probability events
        test1_standard_batch  | int,    analysis ID to test bounding off of
        ams_site_id           | string, project site 4 letter code
        
        PROTOTYPE II INPUTS
        
        bounding_measures     | list of strings, 
                                        measures/tags used to bound, all options 
                                        are columns in "...allstats.xlsx"
        run_prototype_II      | bool,   True to run prototype II
        savings_threshold     | int,    Perfect Foresight Cutoff to Bound off of
        analysis_id           | int,    Analysis ID from Armada 
        sig_level             | float,  significance level used for hypothesis testing
        all_stats_file_path   | string, stores interval data characteristics generated in load_gen.py
                                        stored ending in "all_stats.xlsd" format, this is file path directory
        file                  | string, filename of above
        chars_before_load_name| int,    char count of model runs up to description of load scenario
        
"""

################## Prototype I INPUTS  ##################

# Prototype I INPUTS
savings_threshold = 238427
# savings_threshold = 208427
run_prototype_I = True
bound_type = 'option1_hard_bound'
base_path = LOCAL_DROPBOX + '/Load Data/South Coast Air Quality Mgmt. Dist. (SCAQMD)/Distribution Analysis/_ProjectSites/Data_Review'
baseline_datasets_list = ['SCAQ1001_WebETL.csv']
bound_gen_batch = 6205
precontract_batch = 6193
test1_standard_batch = 6205
ams_site_id = "SCAQ"

################## Prototype II INPUTS  ##################
bounding_measures = ['max','load_factor','total_kwh']
run_prototype_II = True
ams_analytics_path = "/Users/kanito/AMS-Analytics"
savings_threshold = 238427
analysis_id = 6205
sig_level = 0.05
all_stats_file_path = LOCAL_DROPBOX + "/Load Data/South Coast Air Quality Mgmt. Dist. (SCAQMD)/_ETL/_LoadGen_SCAQ1001_Office_nonexport_20180709_113913"
file = "SCAQ_ETL_test1_low_south_coast_prob_allstats_20180907.xlsx"
chars_before_load_name = 104



test_base_path = all_stats_file_path
test_num = "test1_low"

# Prime Load Data Sheets
test_bounds_dict = {
    "ams_site_id": ams_site_id,
    "load_sheets": {
        "load_sheet_precontract": '{}/{}'.format(test_base_path,
                                                      file),
        "load_sheet_bound_gen": '{}/{}'.format(test_base_path,
                                                    file),
        "load_sheet_standard_test1": '{}/{}'.format(test_base_path,
                                                      file),
    },
    "SensitivityBatches": {
        "bound_gen_batch": bound_gen_batch,
        "precontract_batch": precontract_batch,
        "test1_standard_batch": test1_standard_batch,
    },
    "load_sheet_to_SensitivityBatches_mapping": {
        "precontract_batch": "load_sheet_precontract",
        "bound_gen_batch": "load_sheet_bound_gen",
        "test1_standard_batch": "load_sheet_standard_test1",

    },
    "test_datasets_list": [
        "SCAQ1001_low_BLDec_LFinc_0-024_p_0-5_Smooth_3_export_20180606.csv".format(test_num),
        "SCAQ1001_low_BLEE_ScaleDown_0-255_LFinc_0-116_p_0-3_Smooth_4_export_20180606.csv".format(test_num),
        "SCAQ1001_standard_BLDec_LFinc_0-123_p_0-5_Smooth_4_export_20180606.csv".format(test_num),
        "SCAQ1001_standard_BLEE_ScaleDown_0-123_LFinc_0-0105_p_0-45_Smooth_1_export_20180606.csv".format(test_num),
        "SCAQ1001_standard_BLInc_ScaleUp_0-043_export_20180606.csv".format(test_num),
        "SCAQ1001_standard_BLOff_Offshift_5_Smooth_6_export_20180606.csv".format(test_num),
        "SCAQ1001_standard_BLOff_Offshift_13_Smooth_1_export_20180606.csv".format(test_num),
    ]
}

if __name__ == '__main__':
    if run_prototype_I:
        logging.info("Running Prototype I")

        load_sheet_standard_test1 = pd.read_excel(test_bounds_dict['load_sheets']['load_sheet_standard_test1'])
        test1_standard_batch = SensitivityBatch(chars_before_load_name=chars_before_load_name,name="{}_test1_standard_batch".format(ams_site_id), benchmark_run_id=None,
                                            portfolio_runs_list=[
                                               str(test_bounds_dict['SensitivityBatches']['test1_standard_batch'])])
        test1_standard_batch.gen_portfolio_runs_load_merged_df(
            eval(test_bounds_dict['load_sheet_to_SensitivityBatches_mapping']['test1_standard_batch']))

        ## Generate Bounds
        ##Initialize underlying data that create load bound
        load_sheet_precontract = pd.read_excel(test_bounds_dict['load_sheets']['load_sheet_precontract'])
        precontract_batch = SensitivityBatch(chars_before_load_name=chars_before_load_name,name="{}_bound_gen_standard".format(ams_site_id), benchmark_run_id=None,
                                           portfolio_runs_list=[str(test_bounds_dict['SensitivityBatches']["precontract_batch"])],
                                             lower_savings_bound=savings_threshold)

        precontract_batch.gen_portfolio_runs_load_merged_df(
            eval(test_bounds_dict["load_sheet_to_SensitivityBatches_mapping"]['precontract_batch']))


        load_sheet_bound_gen = pd.read_excel(test_bounds_dict['load_sheets']['load_sheet_bound_gen'])
        bound_gen_batch = SensitivityBatch(chars_before_load_name=chars_before_load_name,name="{}_bound_gen_low".format(ams_site_id), benchmark_run_id=None,
                                           portfolio_runs_list=[
                                               str(test_bounds_dict['SensitivityBatches']['bound_gen_batch'])])
        bound_gen_batch.gen_portfolio_runs_load_merged_df(
            eval(test_bounds_dict['load_sheet_to_SensitivityBatches_mapping']['bound_gen_batch']))

        ##Initialize Load Bound Types
        # TODO - just two bound types currently, could pre-initialize a library bound types, or initialize just the ones needed on the fly
        monthly_load_bound_type_default = LoadBoundType(freq="monthly", type=bound_type)
        monthly_load_bound_type_option1 = LoadBoundType(freq="monthly", type=bound_type)

        ##Initialize actual Load Bounds
        test_bound_1 = LoadBound(name=ams_site_id+test_num, load_bound_type=monthly_load_bound_type_default,
                                 batch_to_set_bound=bound_gen_batch,
                                 lower_savings_bound=precontract_batch.lower_savings_bound)

        test_bound_1.lower_savings_bound = savings_threshold
        test_bound_1.batch_to_set_bound.lower_savings_bound = savings_threshold
        baseline_dataset = get_pickled_load_interval_data(base_path, baseline_datasets_list)

        ## Output & set bounds
        # gen_base_stats needs a load bound in order to create the base_stats to the interval stored by load bound
        # TODO within the gen_base_stats function, need to decide if this is the best implementation
        baseline_stats_df = gen_base_stats(baseline_dataset, test_bound_1)
        baseline_stats_df.to_excel('baseline_stats_df.xlsx')
        input_base_stats_df = flatten_monthly_stats_flipped(baseline_stats_df).transpose()
        input_base_stats_df.to_excel('inputbasestats.xlsx')

        # Need input base statistics in order to generate bound numbers. Bound numbers are generated from statistics
        test_bound_1.gen_bound_features(input_base_stats_df)
        test_bound_1.set_bounds()
        test_bound_1.output_bounds()

        ## Testing ##
        """
        batch_for_testing = test1_standard_batch
        test_datasets = get_pickled_load_interval_data(base_path, test_bounds_dict['test_datasets_list'])
        batch_for_testing.portfolio_runs_load_merged_df.to_excel('{}_bounded.xlsx'.format(batch_for_testing.name))
        test_bound_batch(baseline_dataset, test_datasets, batch_for_testing, test_bound_1)
        """

    # Prototype II Module
    if run_prototype_II:

        logging.info("Running Prototype II")

        # Retrieve results from swagger_armada_api.py
        armada_results = get_analysis_results(analysis_id)

        # Visualize Savings Distribution
        visualize_analysis_results(armada_results['PF_totalsavings'], 'PF_totalsavings', analysis_id)

        # Retrieve Interval Attributes aka. all stats Data
        tags, interval_attributes = get_interval_attributes(analysis_id)

        # Consolidate into one DF
        armada_results['load_name'] = armada_results['portfolioname'].str[chars_before_load_name:]
        try:
            master_data_df = pd.merge(armada_results, interval_attributes, left_on='load_name',
                                 right_on='load_name')
            master_data_df.to_excel(str(analysis_id) +'_master_data.xlsx')
        except:
            logging.info( "concat of armada summary and interval attributes was unsuccessful. Check chars_before_load_name input")

        # Check for Normalcy in Tags: Normal Testing
        normal_tags = []
        non_normal_tags = []
        for tag in tags:
            p_value = normal_test(master_data_df[tag], tag, sig_level)
            if p_value > sig_level:
                normal_tags.append(tag)
            else:
                non_normal_tags.append(tag)

        # Split Between Positives (non-acceptable) and Negatives (acceptable)
        pos, neg = split_between_savings_threshold(savings_threshold, master_data_df)

        # Bounding via Z-Testing
        norm_test_results, bounds_df_norm, bounds_collapsed_norm, tags_no_sig= z_test_wrapper(normal_tags, pos, neg)
        norm_test_results.to_excel(str(analysis_id) + '_Z_Test_Results.xlsx')

        # Bounding via T-Testing
        t_test_results, bounds_df_t, bounds_collapsed_t, tags_no_sig = t_test_wrapper(non_normal_tags, pos, neg)
        t_test_results.to_excel(str(analysis_id) +'_T_Test_Results.xlsx')

        # Bounding via Non-Statistical
        bounds_df_nonstat, bounds_df_nonstat, bounds_collapsed_nonstat, unbounded_tags_df = non_statistical_bound(tags_no_sig, pos, neg)
        bounds_df_nonstat.to_excel(str(analysis_id) +'_Nonstat_Results.xlsx')
        unbounded_tags_df.to_excel(str(analysis_id) + '_Unbounded_tags.xlsx')

        # Convert to percentage against baseline load
        input_base_stats_df = pd.read_excel('inputbasestats.xlsx')
        input_base_stats_df.columns = [['tag','Baseline']]
        input_base_stats_df_annual = input_base_stats_df.copy()
        input_base_stats_df_annual['measure'] = input_base_stats_df_annual['tag'].str.extract(pat=r"_(.*)")

        # Output Monthly Bounding Results
        bounds_df_t['Statistical Test Type'] = 'T Test'
        bounds_df_norm['Statistical Test Type'] = 'Z Test'
        bounds_df_nonstat['Statistical Test Type'] = 'Nonstat'
        p_2_bds_mo = bounds_df_t.append(bounds_df_norm)
        p_2_bds_mo = p_2_bds_mo.append(bounds_df_nonstat)
        p_2_bds_mo = p_2_bds_mo.sort_index(by=['measure', 'tag'])
        p_2_bds_mo = pd.merge(p_2_bds_mo, input_base_stats_df, on='tag', sort=False)
        p_2_bds_mo.columns = [['Statistical Test Type','bound_low_raw','bound_up_raw','measure','p_value','tag','Baseline']]
        p_2_bds_mo['bound_low_pct_diff'] = (p_2_bds_mo['Baseline'] - p_2_bds_mo['bound_low_raw']) / p_2_bds_mo['Baseline'] * 100
        p_2_bds_mo['bound_up_pct_diff'] = (p_2_bds_mo['Baseline'] - p_2_bds_mo['bound_up_raw']) / p_2_bds_mo['Baseline'] * -100
        p_2_bds_mo.to_excel(str(analysis_id) + "_Prototype_II_bounds_monthly.xlsx")

        # Output Annual Bounding Results via largest percentage difference
        # Flatten Monthly into Annual Bound, losing month specificity here
        annual_low_list = []
        annual_up_list = []
        measure_list = []
        month_count = []
        for measure in p_2_bds_mo['measure'].unique():
            filtered = p_2_bds_mo[p_2_bds_mo['measure'] == measure]
            annual_low_list.append(filtered['bound_low_pct_diff'].max())  # taking the extremes of up to 12 months
            annual_up_list.append(filtered['bound_up_pct_diff'].max())  # taking the extremes of up to 12 months
            month_count.append(filtered['bound_low_pct_diff'].count())
            measure_list.append(measure)
        p_2_bds_yr_v2 = pd.DataFrame(
            {
                "bound_low_pct_diff": annual_low_list,
                "bound_up_pct_diff": annual_up_list,
                "measure": measure_list,
                "month_count": month_count
            }
        )
        p_2_bds_yr_v2.to_excel(str(analysis_id)+'_Prototype_II_bounds_annual_V2.xlsx')

        # Output Annual Bounding Results via largest raw bounds
        bounds_collapsed_t['Statistical Test Type'] = 'T Test'
        bounds_collapsed_norm['Statistical Test Type'] = 'Z Test'
        bounds_collapsed_nonstat['Statistical Test Type'] = 'Nonstat'
        p_2_bds_yr_v1 = bounds_collapsed_t.append(bounds_collapsed_norm)
        p_2_bds_yr_v1 = p_2_bds_yr_v1.append(bounds_collapsed_nonstat)
        p_2_bds_yr_v1 = p_2_bds_yr_v1.sort_index(by=['measure','month_count'])
        p_2_bds_yr_v1.to_excel(str(analysis_id)+"_Prototype_II_bounds_annual_V1.xlsx")

        # Performance Calculator V1
        logging.info( "Calculating Bound Performance  V1")
        bound_df = p_2_bds_yr_v2.where(p_2_bds_yr_v2.measure == bounding_measures[0])
        for measure in bounding_measures[1:]:
            bound_df.append(p_2_bds_yr_v2.where(p_2_bds_yr_v2.measure == measure))

        bound_calc_df = calc_conf_matrix_perf_v1(bound_df, input_base_stats_df, bounding_measures, master_data_df)
        bound_calc_df.to_excel(str(analysis_id) + "_Prototype_II_bounds_performance_V1.xlsx")
        logging.info("File "+str(analysis_id) + "_Prototype_II_bounds_performance_V1.xlsx created")

        # Performance Calculator V2
        logging.info( "Calculating Bound Performace V2")
        bound_df = p_2_bds_yr_v2.where(p_2_bds_yr_v2.measure == bounding_measures[0])
        for measure in bounding_measures[1:]:
            bound_df.append(p_2_bds_yr_v2.where(p_2_bds_yr_v2.measure == measure))
        bound_calc_df = calc_conf_matrix_perf_v2(bound_df, input_base_stats_df, bounding_measures, master_data_df)
        bound_calc_df.to_excel(str(analysis_id) + "_Prototype_II_bounds_performance_V2.xlsx")
        logging.info("File " + str(analysis_id) + "_Prototype_II_bounds_performance_V2.xlsx created")

        # Normalize Sample Data
        master_data_normalized = preprocessing.normalize(master_data_df[tags], norm='l2')
        master_data_normalized = pd.DataFrame(master_data_normalized, columns=tags)
        """
        get_armada_results.visualize_results(master_data_normalized['load_factor'], "normalized load_factor", analysis_id)
        get_armada_results.visualize_results(master_data['load_factor'], "load_factor", analysis_id)
        get_armada_results.visualize_results(master_data['max_kw'], "max_kw", analysis_id)
        """

        # Scale Sample Data for Logistic Regression
        data_scaler = preprocessing.MinMaxScaler()
        master_data_scaled = data_scaler.fit_transform(master_data_df[tags])
        master_data_scaled = pd.DataFrame(master_data_scaled, columns=tags)
        """
        get_armada_results.visualize_results(master_data_scaled['load_factor'], "scaled load_factor", analysis_id)
        get_armada_results.visualize_results(master_data_scaled['max_kw'], "scaled max_kw", analysis_id)
        """

        # Run Logistic Regression on Scaled Data
        logging.info("Running Logistic Regression on Metrics")
        log_reg(master_data_scaled, tags, master_data_df['savings_unacceptable'])
        log_reg(master_data_scaled, normal_tags, master_data_df['savings_unacceptable'])
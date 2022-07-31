from textwrap import wrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns


def missing_data(
    df, subset=[], title="Proportion of missing values per column", ax=None
):
    if len(subset):
        df = df[subset]

    df = df.isna().sum(axis=0) / df.shape[0]
    df = df[df > 0].sort_values().to_frame(name="Missing values (%)")

    ax = sns.barplot(data=df * 100, x="Missing values (%)", y=df.index, ax=None)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.set_xlim(0, 100)
    ax.set(ylabel="Column", title=title)


def sns_multi_time_series(df, x, y, hue, xlabel="", ylabel="", title="", col_wrap=3):
    g = sns.relplot(
        data=df,
        x=x,
        y=y,
        col=hue,
        hue=hue,
        kind="line",
        palette="colorblind",
        linewidth=4,
        zorder=5,
        col_wrap=col_wrap,
        height=3,
        aspect=1.35,
        legend=False,
    )

    for year, ax in g.axes_dict.items():
        # Add the title as an annotation within the plot
        ax.text(0, 1.05, year, transform=ax.transAxes, fontweight="bold")

        # Plot every year's time series in the background
        sns.lineplot(
            data=df,
            x=x,
            y=y,
            units=hue,
            estimator=None,
            legend=False,
            color=".7",
            linewidth=1,
            ax=ax,
        )

    # Tweak the supporting aspects of the plot
    g.set_titles("")
    g.set_axis_labels(xlabel, ylabel)
    g.tight_layout()
    g.fig.subplots_adjust(top=0.88)
    g.fig.suptitle(title)

    return g


def band_proportions_periodic(df: pd.DataFrame, col, title="", period="M", subset=[]):
    quarter_sums = (
        df.groupby([df.AdmissionDateTime.dt.to_period(period), col])
        .size()
        .sum(level=0)
        .rolling(4)
        .sum()
    )

    df = (
        df.groupby([col, df.AdmissionDateTime.dt.to_period(period)])
        .size()
        .groupby(level=0, group_keys=False)
        .rolling(4)
        .sum()
    )
    df.index = df.index.droplevel(0)

    df = df.div(quarter_sums).to_frame("Proportion").reset_index()
    df.Proportion *= 100
    df.AdmissionDateTime = df.AdmissionDateTime.dt.to_timestamp()
    df = df[df[col] != "nan"]
    if len(subset):
        df = df[df[col].isin(subset)]

    return sns_multi_time_series(
        df,
        "AdmissionDateTime",
        "Proportion",
        col,
        xlabel="Admission",
        ylabel="Proportion (%)",
        title=title,
    )


def single_boxplot(df, col):
    fig, ax = plt.subplots(figsize=(12, 2))
    sns.boxplot(data=df[col], orient="h", ax=ax)
    ax.set(title=col)


def topn_icd_in_year(df, icd10, datestart, dateend, topn=5, title=None):
    merged = df[df.AdmissionDateTime.between(datestart, dateend)]
    # Get proportions per opcode during the chosen year
    g = merged.groupby([merged.Group_Code, merged.MainICD10_3_Code]).SpellSerial.count()
    g = g / g.sum()

    # Get the top groups for the year
    top_groups = g.sum(level=0).nlargest(topn).to_frame().reset_index()
    top_groups["MainICD10_3_Code"] = top_groups.Group_Code

    g = (
        g.loc[top_groups.Group_Code]
        .groupby(level=0, group_keys=False)
        .nlargest(topn)
        .reset_index()
    )
    g = pd.concat([top_groups, g])
    g.SpellSerial *= 100

    g = g.merge(
        icd10[["Group_Code", "Group_Desc"]], right_on="Group_Code", left_on="Group_Code"
    )
    ax = sns.barplot(
        data=g.sort_values(["Group_Code", "SpellSerial"], ascending=False),
        x="SpellSerial",
        y="MainICD10_3_Code",
        hue="Group_Desc",
        dodge=False,
    )
    ax.set(
        xlabel="Annual Admissions (%)",
        ylabel="ICD-10",
        title=title
        or f"Top {topn} ICD-10 code groups by proportion of admissions during {datestart} - {dateend}",
    )
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), borderaxespad=0.0)


def median_los_per_band(df, col, period="M", subset=[], title=""):
    df = (
        df.groupby([col, df.AdmissionDateTime.dt.to_period(period)])
        .TotalLOS.median()
        .groupby(level=0, group_keys=False)
        .rolling(4)
        .mean()
    )
    df.index = df.index.droplevel(0)
    df = df.reset_index()
    df.AdmissionDateTime = df.AdmissionDateTime.dt.to_timestamp()
    df = df[df.TotalLOS.notna()]
    if len(subset):
        df = df[df[col].isin(subset)]

    return sns_multi_time_series(
        df,
        "AdmissionDateTime",
        "TotalLOS",
        col,
        xlabel="Admission",
        ylabel="Median LOS",
        title=title,
    )


sci_subset = [
    "Group_Code",
    "AandEPresentingComplaint",
    "AandEMainDiagnosis",
    "AandEArrivalTime",
    "AandEDepartureTime",
    "AandELocation",
    "AandEPatientGroupDescription",
    "NewsCreatedWhen",
    "NewsTouchedWhen",
    "NewsAuthoredDtm",
    "c_Respiration_rate",
    "c_NEWS_resp_rate_score",
    "c_O2_device_or_air",
    "c_NEWS_device_air_score",
    "c_O2_saturation",
    "c_Oxygen_flow_rate",
    "c_NEWS_O2_sat_score",
    "c_Temperature",
    "c_NEWS_temperature_score",
    "c_Patient_Position",
    "c_BP_Systolic",
    "c_BP_Diastolic",
    "c_NEWS_BP_score",
    "c_Heart_rate",
    "c_NEWS_heart_rate_score",
    "c_Level_of_consciousness",
    "c_NEWS_level_of_con_score",
    "c_Pain",
    "c_Nausea",
    "c_Vomiting_since_last_round",
    "c_NEWS_score",
    "CareHome",
    "DiedDuringStay",
    "DiedWithin30Days",
    "DischargeDestinationDescription",
    "Haemoglobin",
    "Urea(serum)",
    "Sodium(serum)",
    "Potassium(serum)",
    "Creatinine",
    "PatientTemperatureVenous",
    "pCO2(POC)Venous",
    "pCO2(Tempcorrected)(POC)Venous",
    "PH(POC)Venous",
    "PH(Tempcorrected)(POC)Venous",
    "pO2(POC)Venous",
    "pO2(Tempcorrected)(POC)Venous",
    "IstCFSDate",
    "AllCFS",
    "LastCFSDate",
    "SEQ",
    "PatientNoSeq",
    "AllCFS.1",
    "AllDatesofCFSReadings",
    "NoofminsbetweenAllCFS&Admission",
    "CFSReadingsBefore(B)After(A)Addmission",
    "NoMinsBeforeadmission",
    "WordingBeforeAdmission",
    "CFSBeforeadmission",
    "CFSAfterAdmission",
    "Noofminsafteradmission",
    "WordingAfterAdmission",
    "SpellSerial",
    "PatientNumber",
    "AdmissionDateTime",
    "ElectiveAdmission",
    "PatientType",
    "IntendedManagement",
    "AdmissionMethodDescription",
    "AdmissionSpecialty",
    "LastSpecialty",
    "DischargeDateTime",
    "TotalLOS",
    "Over7Days",
    "Over14Days",
    "AdmitWard",
]


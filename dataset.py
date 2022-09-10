import string

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.base import BaseEstimator, TransformerMixin


class SCIData(pd.DataFrame):
    """Represents the SCI dataset and related methods to augment or filter it"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def quickload(file, table="table"):
        """Loads the specified HDF5 file into Pandas directly.
        :param str file: The (local) file path
        :param str table: The name of the table to load from the HDF5. By default, 'table' is used.
        :return: DataFrame
        """
        return pd.read_hdf(file, table)

    @classmethod
    def load(cls, file, table="table"):
        """Loads the specified HDF5 file into Pandas.
        :param str file: The (local) file path
        :param str table: The name of the table to load from the HDF5. By default, 'table' is used.
        :return: Instance of SCIData with the DataFrame loaded.
        """
        return cls(data=pd.read_hdf(file, table))

    @classmethod
    def fromxy(cls, X, y):
        return cls(pd.concat([X, pd.Series(y, name="y")], axis=1))

    def save(self, filename="data/sci_processed.h5"):
        self.to_hdf(filename, "table")
        return self

    def derive_all(self):
        """Runs all methods to derive handcrafted features from the raw dataset
        :return: New SCIData instance with the new features added
        """
        return (
            self.derive_covid_strict()
            # .derive_mortality()
            .derive_readmission()
            # .derive_critical_care()
            # .derive_news_risk()
        )

    def clean_all(self):
        return (
            self.clean_blood_pressure()
            .clean_consciousness()
            .clean_device_air()
            .clean_heart_rate()
            .clean_breathing_device()
            .clean_O2_flow_rate()
            .clean_O2_saturation()
            .clean_respiration_rate()
            .clean_temperature()
            .clean_ae_text()
            .clean_ae_patient_group()
            .clean_news()
        )

    def augment_ccs(self, infile="data/ccs.h5", ccs=None, sentinel=259, onehot=False):
        """Joins SCI and the CCS reference table fuzzily. Applies to ALL coded diagnoses, not just MainICD10
        Codes that can't be matched exactly are matched with their 3-codes
        :param infile: The HDF5 file to load the CCS matching table from
        :param ccs: Pre-loaded CCS match table dataframe
        :param sentinel: What to fill un-matched values with. Default is the code for unclassified/residual codes
        :param onehot: Whether to onehot encode the groupings
        :returns: New SCIData instance with augmented with CCS codes and descriptions
        """
        r = self.copy()
        if ccs is None:
            ccs = pd.read_hdf(infile, "codes")

        icd = r[SCICols.diagnoses]
        no_dot = (
            icd.apply(lambda col: col.str.replace(".", "", regex=False))
            .stack()
            .rename("icd10")
            .to_frame()
        )

        perfect = no_dot.join(ccs, on="icd10").CCSGroup.unstack()
        approx = (
            no_dot.apply(lambda col: col.str[:_])
            .join(ccs, on="icd10")
            .CCSGroup.unstack()
            for _ in (3, 4)
        )

        for df in approx:
            mask = df.notna() & perfect.isna() & icd.notna()
            perfect[mask] = df[mask]

        remaining = perfect.isna() & icd.notna()
        perfect[remaining] = sentinel

        perfect.columns = SCICols.diagnoses
        r[perfect.columns] = perfect

        if onehot:
            r = r.encode_ccs_onehot()

        return SCIData(r)

    def _regroup_ccs(self, df, col, onehot=False):
        """Joins SCI and a given grouping table for CCS. Matches ICD-10 diagnoses with CCS if this has not already been done
        :param df: The CCS grouping table (SHMI or HSMR)
        :param col: The column from the matching table to keep
        :param onehot: Whether to onehot encode the result
        """
        r = self.augment_ccs()
        cols = SCICols.diagnoses
        joined = (
            r[cols].stack().rename("ccs").to_frame().join(df, on="ccs")[col].unstack()
        )
        r[cols] = joined

        if onehot:
            r = r.encode_ccs_onehot()

        return SCIData(r)

    def augment_shmi(self, infile="data/ccs.h5", shmi=None, onehot=False):
        """Joins SCI and the SHMI matching table.
        :param infile: The HDF5 file to load the matching table from
        :param ccs: Pre-loaded SHMI match table dataframe
        :param onehot: Whether to onehot encode the groupings
        :returns: New SCIData instance with augmented with SHMI diagnosis groups and descriptions
        """
        if shmi is None:
            shmi = pd.read_hdf(infile, "shmi")

        return self._regroup_ccs(shmi, "SHMIGroup", onehot=onehot)

    def augment_hsmr(self, infile="data/ccs.h5", hsmr=None, onehot=False):
        """Joins SCI and the HSMR matching table
        :param infile: The HDF5 file to load the matching table from
        :param hsmr: Pre-loaded HSMR match table dataframe
        :returns: New SCIData instance with augmented with HSMR aggregate groups and descriptions
        """
        if hsmr is None:
            hsmr = pd.read_hdf(infile, "hsmr")

        return self._regroup_ccs(hsmr, "AggregateGroup", onehot=onehot)

    def augment_icd10(
        self, infile="data/icd10.h5", icd10=None, keep=None, onehot=False, drop_old=True
    ):
        """Joins SCI and the ICD-10 chapter & group table.
        :param infile: The HDF5 file to load the table from
        :param icd10: Pre-loaded ICD-10 table dataframe
        :param keep: Columns to keep from the new grouping. If None, will keep them all
        :param onehot: Whether to onehot encode the result
        :param drop_old: Whether to drop the original coded diagnoses
        :returns: New SCIData instance with augmented with ICD-10 groups and chapters
        """
        if icd10 is None:
            icd10 = pd.read_hdf(infile, "ICD10_3_Codes")

        r = self.join(icd10, on=self.MainICD10.str[:3])

        if keep is not None:
            r = r.drop(set(SCICols.icd10_grouping) - set(keep), axis=1)

        if drop_old:
            r = r.drop(SCICols.diagnoses, axis=1, errors="ignore")

        if onehot:
            return SCIData(r).encode_onehot(keep, "ICD10", drop_old)

        return SCIData(r)

    def augment_icd10_group(self, onehot=False, drop_old=True):
        return self.augment_icd10(keep=["Group_Code"], onehot=onehot, drop_old=drop_old)

    def augment_icd10_chapter(self, onehot=False, drop_old=True):
        return self.augment_icd10(keep=["Chapter_No"], onehot=onehot, drop_old=drop_old)

    def augment_icd10_3code(self, onehot=False, drop_old=True):
        return self.augment_icd10(
            keep=["MainICD10_3_Code"], onehot=onehot, drop_old=drop_old
        )

    def augment_icd10_descriptions(self, infile="data/icd10.h5", icd10=None):
        """Joins SCI and the ICD-10 code table.
        :param infile: The HDF5 file to load the table from
        :param hsmr: Pre-loaded ICD-10 table dataframe
        :returns: New SCIData instance with augmented with ICD-10 full descriptions
        """
        if icd10 is None:
            icd10 = pd.read_hdf(infile, "ICD10_Codes")

        return SCIData(self.join(icd10, on="MainICD10"))

    def derive_covid(
        self, covid_codes=["U07.1", "J12.8", "B97.2"], start_date="2020-01-01",
    ):
        """Derives a binary feature indicating whether the patient had COVID-19, based on their coded diagnoses.
        :param covid_codes: The ICD-10 codes to look for.
        :return: New SCIData instance with the new feature added
        """
        year_mask = self.AdmissionDateTime >= start_date
        covid_mask = self[SCICols.diagnoses].isin(covid_codes).any(axis=1)

        r = self.copy()
        r["Covid"] = year_mask & covid_mask
        return SCIData(data=r)

    def derive_covid_strict(self, covid_codes=["U07.1", "J12.8", "B97.2"]):
        """Derives a binary feature indicating whether the patient had COVID-19 based on the entire ICD10 code combination.
        :return: New SCIData instance with the new feature added
        """
        covid_mask = self[SCICols.diagnoses].apply(frozenset, axis=1) >= frozenset(
            covid_codes
        )

        r = self.copy()
        r["Covid"] = covid_mask
        return SCIData(data=r)

    def derive_readmission(
        self,
        bins=[0, 1, 2, 7, 14, 30, 60],
        labels=["24 Hrs", "48 Hrs", "1 Week", "2 Weeks", "1 Month", "2 Months"],
        readmission_thresh=30,
    ):
        """Determines the timespan since the patient's last admission. Bins readmissions into bands.
        :param bins: Bin edges for readmission bands
        :param labels: String labels for the returned bands
        :return: New SCIData instance with the new features added
        """
        bins = [pd.Timedelta(days=_) for _ in bins]

        r = self.copy()
        grouped = r.sort_values(["PatientNumber", "AdmissionDateTime"]).groupby(
            "PatientNumber"
        )
        r["ReadmissionTimespan"] = grouped.AdmissionDateTime.diff().dropna()
        r["ReadmissionBand"] = (
            pd.cut(r.ReadmissionTimespan, bins, labels=labels)
            .astype(str)
            .replace("nan", np.nan)
            .fillna("N/A")
        )
        r["Readmission"] = r.ReadmissionTimespan < pd.Timedelta(days=readmission_thresh)

        timespan_reverse = grouped.AdmissionDateTime.diff(-1).dropna()
        r["Readmitted"] = (
            timespan_reverse > pd.Timedelta(days=-readmission_thresh)
        ).fillna(False)
        r["ReadmittedTimespan"] = timespan_reverse * (-1)

        return SCIData(data=r)

    def derive_mortality(self, within=1, col_name="DiedWithinThreshold"):
        """Determines the patients' mortality outcome.
        :param within: Time since admission to consider a death. E.g., 1.0 means died within 24 hours, otherwise lived past 24 hours
        :return: New SCIData instance with the new feature added
        """
        r = self.copy()

        m = r[["DiedDuringStay", "DiedWithin30Days", col_name]].copy()
        m["DiedDuringStay"] = m["DiedDuringStay"] & (~m[col_name])
        m["DidNotDie"] = ~m.any(axis=1)

        r["Mortality"] = m.dot(m.columns)
        return SCIData(r)

    def derive_death_within(self, within=1, col_name="DiedWithinThreshold"):
        """Determines the patients' mortality outcome.
        :param within: Time since admission to consider a death. E.g., 1.0 means died within 24 hours, otherwise lived past 24 hours
        :return: New SCIData instance with the new feature added
        """
        r = self.copy()
        r[col_name] = r.DiedDuringStay & (r.TotalLOS <= within)

        return SCIData(r)

    def derive_critical_care(
        self, critical_wards=["CCU", "HH1M"], within=1, col_name="CriticalCare",
    ):
        """Determines admission to critical care at any point during the spell as indicated by admission to specified wards
        :param critical_wards: The wards to search for. By default, ['CCU', 'HH1M']
        :param within: Threshold of maximum LOS to consider events for. Critical care admission that occurs after this value won't be counted.
        :return: New SCIData instance with the new features added
        """
        r = self.copy()
        m = r[SCICols.wards].isin(critical_wards)
        column_where_critical_appeared = m.idxmax(axis=1).where(m.any(1)).dropna()
        los_on_critical_admission = (
            r[SCICols.ward_los]
            .stack()
            .groupby(level=0)
            .cumsum()
            .loc[
                list(
                    zip(
                        column_where_critical_appeared.index,
                        column_where_critical_appeared + "LOS",
                    )
                )
            ]
        )
        los_on_critical_admission.index = los_on_critical_admission.index.droplevel(1)

        # r["CriticalCare"] = m.any(axis=1)
        r[col_name] = los_on_critical_admission <= within
        r[col_name].fillna(False, inplace=True)

        return SCIData(r)

    def derive_news_risk(self):
        """Derives the NEWS clinical risk based on the pre-defined triggers
        :return: New SCIData instance with the new feature added
        """
        r = self.copy()

        mask = r.c_NEWS_score.notna()
        r.loc[mask, "c_NEWS_risk"] = r.loc[mask, "c_NEWS_score"].apply(NEWS.risk_scale)
        return SCIData(r)

    def filter_vague_diagnoses(self):
        """Filters out ICD10 codes marked as vague per Hassani15. Remaining diagnoses are shifted to the right.
        :return: New SCIData instance with the diagnoses filtered
        """
        ranges = [
            ("R", 00, 56),
            ("R", 58, 64),
            ("R", 66, 94),
            ("V", 1, 98),
            ("Z", 00, 13),
            ("Z", 20, 29),
            ("Z", 32, 36),
            ("Z", 39, 39),
            ("Z", 43, 48),
            ("Z", 52, 99),
        ]

        vague = frozenset(
            f"{letter}{str(num).zfill(2)}"
            for letter, start, stop in ranges
            for num in range(start, stop + 1)
        )

        r = self[SCICols.diagnoses].stack()
        r[r.str[:3].isin(vague)] = np.nan
        r = justify(r.unstack())

        result = self.copy()
        result[SCICols.diagnoses] = r

        return SCIData(data=result)

    def clean_news(self):
        """Re-computes the NEWS score for cases where it is missing (but we have all the components) or it disagrees with the components on-record"""
        brand_news = self[SCICols.news_data_scored].sum(axis=1)

        has_params = self[SCICols.news_data_scored].notna().all(axis=1)
        has_news = self.c_NEWS_score.notna()
        mask = (has_params & has_news & (self.c_NEWS_score != brand_news)) | (
            has_params & ~has_news
        )

        r = self.copy()
        r.loc[mask, "c_NEWS_score"] = brand_news[mask]

        return SCIData(r)

    def clean_breathing_device(self):
        col = "c_Breathing_device"
        r = self.copy()

        mask = r[col].notna() & r[col].str.lower().str.startswith("other; nhf")
        r.loc[mask, col] = "NHF"

        mask = ~(r[col].isin(r[col].value_counts().head(16).index)) & r[col].notna()
        r.loc[mask, col] = "Other"

        return SCIData(r)

    def clean_O2_saturation(self, outlier_threshold_std=3):
        sat, score, assisted, device = (
            "c_O2_saturation",
            "c_NEWS_O2_sat_score",
            "c_Assisted_breathing",
            "c_Breathing_device",
        )
        r = self.copy()

        # Assume negatives were erroneous and make them positive
        r.loc[r[sat] < 0, sat] *= -1

        # Assume zeros are dead
        r.loc[r[sat] == 0, sat] = np.nan

        # Assume single-digits were erroneous and shift to the left (e.g. 9.0 -> 90)
        r.loc[r[sat] <= 10, sat] *= 10

        # Set threshold for outliers to be under 3 std of the mean OR anything >100 (as it is impossible)
        threshold = r[sat].mean() - 3 * r[sat].std()
        outlier = (r[sat] > 100) | (r[sat] <= threshold)

        # Set criterion for SpO2_2 scale to be presence of COPD in the coded diagnoses
        copd = (
            r[SCICols.diagnoses]
            .isin(["J44.0", "J44.1", "J44.8", "J44.9", "J44.X"])
            .any(axis=1)
        )

        # Scale 2 will be:
        scale2 = (
            (r[SCICols.operations] == "E85.2").any(
                axis=1
            )  # Non-invasive ventilation as a procedure code
            | (
                r[device] == "NIV - NIV"
            )  # COPD and assisted breathing with non-invasive ventilation
            | (
                copd & (r[device].str.startswith("V"))
            )  # COPD (per diagnoses) and Venturi assisted breathing
            | (copd & r[sat] < 88)  # COPD and sats under 88
        )

        # Outliers not on assisted breathing or with E85.2 and O2 Sat < 3 will be assigned to the midpoint of Scale 1
        r.loc[outlier & ~scale2, sat] = np.nan

        # Set outliers with copd and O2 Sat score = 0 to 92
        mask2 = outlier & scale2 & (r[score] == 0)
        r.loc[mask2, sat] = 92

        # Delete the remaining outliers
        r.loc[outlier & scale2 & ~mask2, sat] = np.nan

        # Missing: Populate missing O2 saturation from score if available
        missing_sat_mask = r[sat].isna() & r[score].notna()
        missing_sat_scale1, missing_sat_scale2 = (
            ~scale2 & missing_sat_mask,
            scale2 & (r[score] == 0) & missing_sat_mask,
        )

        r.loc[missing_sat_scale1, sat] = r.loc[missing_sat_scale1, score].apply(
            NEWS.SpO2_1_Scale_reverse
        )
        r.loc[missing_sat_scale2, sat] = 92

        # Missing: Populate missing O2 score from saturation if available
        missing_score_mask = r[sat].notna() & r[score].isna()
        missing_score_scale1, missing_score_scale2 = (
            ~scale2 & missing_score_mask,
            scale2 & missing_score_mask,
        )

        r.loc[missing_score_scale1, score] = r.loc[missing_score_scale1, sat].apply(
            NEWS.SpO2_1_Scale
        )
        r.loc[missing_score_scale2, score] = r.loc[missing_score_scale2].apply(
            lambda row: NEWS.SpO2_2_Scale(row[sat], row[assisted]), axis=1
        )

        return SCIData(r)

    def clean_blood_pressure(self, outlier_threshold_std=3):
        sys, dia, score = "c_BP_Systolic", "c_BP_Diastolic", "c_NEWS_BP_score"
        r = self.copy()

        for col in [sys, dia]:
            # Assume negatives were erroneous and make them positive
            r.loc[r[col] < 0, col] *= -1

        # Delete values outside cutoff range
        outlier_dia = (r[dia] <= 20) | (r[dia] >= 200)
        r.loc[outlier_dia, dia] = np.nan

        outlier_sys = (r[sys] <= 40) | (r[sys] >= 300) | (r[sys] <= r[dia] + 5)
        r.loc[outlier_sys, sys] = np.nan

        # Fill missing values using score and vice versa
        missing_score_mask, missing_sys_mask = (
            r[score].isna() & r[sys].notna(),
            r[score].notna() & r[sys].isna(),
        )

        r.loc[missing_sys_mask, sys] = r.loc[missing_sys_mask, score].apply(
            NEWS.systolic_scale_reverse
        )
        r.loc[missing_score_mask, score] = r.loc[missing_score_mask, sys].apply(
            NEWS.systolic_scale
        )

        return SCIData(r)

    def clean_respiration_rate(self):
        rate, score = "c_Respiration_rate", "c_NEWS_resp_rate_score"
        r = self.copy()

        # Assume negatives are erroneous and make them positive
        r.loc[r[rate] < 0, rate] *= -1

        # Assume triple digits are erroneous and shift them to the left
        r.loc[r[rate] >= 100, rate] //= 10

        outliers = (r[rate] >= 80) | (r[rate] <= 5)
        r.loc[outliers, rate] = np.nan

        missing_score_mask, missing_rate_mask = (
            r[score].isna() & r[rate].notna(),
            r[rate].isna() & r[score].notna(),
        )

        r.loc[missing_score_mask, score] = r.loc[missing_score_mask, rate].apply(
            NEWS.respiration_scale
        )
        r.loc[missing_rate_mask, rate] = r.loc[missing_rate_mask, score].apply(
            NEWS.respiration_scale_reverse
        )

        return SCIData(r)

    def clean_device_air(self):
        score, value = "c_NEWS_device_air_score", "c_Assisted_breathing"
        r = self.copy()

        # Some erroneous 1.0 values (impossible per the scale)
        r.loc[r[score] == 1.0, score] = np.nan

        missing_score_mask, missing_value_mask = (
            r[value].notna() & r[score].isna(),
            r[score].notna() & r[value].isna(),
        )

        r.loc[missing_score_mask, score] = r.loc[missing_score_mask, value].map(
            {True: 2.0, False: 0.0}
        )

        # Assume the score is more correct
        r[value] = r[score].map({2.0: True, 0.0: False})

        return SCIData(r)

    def clean_temperature(self):
        value, score = "c_Temperature", "c_NEWS_temperature_score"
        r = self.copy()

        r.loc[r[value] < 0, value] *= 1

        outliers = (r[value] > 45) | (r[value] < 25)
        r.loc[outliers, value] = np.nan

        missing_value_mask, missing_score_mask = (
            r[value].isna() & r[score].notna(),
            r[score].isna() & r[value].notna(),
        )

        r.loc[missing_value_mask, value] = r.loc[missing_value_mask, score].apply(
            NEWS.temperature_scale_reverse
        )
        r.loc[missing_score_mask, score] = r.loc[missing_score_mask, value].apply(
            NEWS.temperature_scale
        )

        return SCIData(r)

    def clean_heart_rate(self):
        score, value = "c_NEWS_heart_rate_score", "c_Heart_rate"
        r = self.copy()

        r.loc[r[value] < 0, value] *= 1

        outliers = (r[value] < 25) | (r[value] > 300)
        r.loc[outliers, value] = np.nan

        missing_value_mask, missing_score_mask = (
            r[value].isna() & r[score].notna(),
            r[score].isna() & r[value].notna(),
        )

        r.loc[missing_value_mask, value] = r.loc[missing_value_mask, score].apply(
            NEWS.heartrate_scale_reverse
        )
        r.loc[missing_score_mask, score] = r.loc[missing_score_mask, value].apply(
            NEWS.heartrate_scale
        )

        return SCIData(r)

    def clean_consciousness(self):
        score, value = "c_NEWS_level_of_con_score", "c_Alert"
        r = self.copy()

        # Some erroneous 1.0 values (impossible per the scale)
        r.loc[r[score] == 1.0, score] = np.nan

        missing_score_mask, missing_value_mask = (
            r[value].notna() & r[score].isna(),
            r[score].notna() & r[value].isna(),
        )

        r.loc[missing_score_mask, score] = r.loc[missing_score_mask, value].map(
            {False: 3.0, True: 0.0}
        )

        # Assume the score is more correct
        r[value] = r[score].map({3.0: False, 0.0: True})

        return SCIData(r)

    def clean_O2_flow_rate(self):
        r = self.copy()
        flow, device = "c_Oxygen_flow_rate", "c_Breathing_device"
        lpm_device_mask = r[device].isin(
            ["A - Air", "N - Nasal cannula", "SM - Simple mask"]
        )

        # LPM is:
        lpm = ((r[flow] >= 1) & (r[flow] <= 15)) | ((r[flow] == 0.5) & lpm_device_mask)

        # Anything over 15 that isnt LPM is a decimal expressed as non-decimal i.e. 36.0 instead of 0.36:
        decimal = ~lpm & (r[flow] > 15)
        r.loc[decimal, flow] /= 100

        # Standardise to FiO2
        fio2 = lambda lpm: (0.2 + lpm * 4) / 100
        r.loc[lpm, flow] = r.loc[lpm, flow].apply(fio2)

        # Delete remainder
        r.loc[r[flow] > 1, flow] = np.nan

        return SCIData(r)

    def clean_ae_text(self):
        complaint, diag = "AandEPresentingComplaint", "AandEMainDiagnosis"
        r = self.copy()

        vague = [
            "referral to service (procedure)",
            "generally unwell (finding)",
            "unwell adult",
            "unknown",
            "other",
            "general deterioration",
            "generally unwell",
            "gen unwell",
        ]
        for col in [complaint, diag]:
            r[col] = r[col].str.lower().str.strip(" .?+")
            r.loc[r[col].isin(vague), col] = np.nan

        mask = ~r[complaint].isin(r[complaint].value_counts().head(50).index)
        r.loc[mask, complaint] = np.nan

        r[complaint].fillna("other", inplace=True)

        return SCIData(r)

    def clean_ae_patient_group(self):
        r = self.copy()
        col = "AandEPatientGroupDescription"

        r[col] = r[col].str.lower()
        r[col] = r[col].replace(
            {
                "other than above": "other",
                "falls": "trauma",
                "road traffic accident": "trauma",
                "sports injury": "trauma",
                "knife injuries inflicted": "trauma",
                "assault": "trauma",
                "self harm": "trauma",
                "accident": "trauma",
                "eyes": "other",
                "dental": "other",
                "s/guarding issue/vunerable per": "other",
                "rheumatology": "other",
            }
        )

        return SCIData(r)

    def clean_icd10(self, icd10=None):
        if icd10 is None:
            icd10 = pd.read_hdf("data/icd10.h5", "ICD10_Codes")

        r = self[SCICols.diagnoses].stack()

        # Fix entries matching 'A12.34 D' or 'A12.X'
        mask = r.str.contains(" ") | r.str.endswith(".X")
        r[mask] = r[mask].str[:-2]

        # Fix entries matching 'A12.34D'
        mask = r.str[-1].isin(frozenset(string.ascii_uppercase))
        r[mask] = r.str[:-1]

        # Delete entries not in the external table (only 3 in the entire dataset!)
        mask = ~r.isin(frozenset(icd10.index))
        r[mask] = np.nan
        r = justify(r.unstack())

        result = self.copy()
        result[SCICols.diagnoses] = r

        return SCIData(result)

    def impute_news(self):
        """Fill missing values in the NEWS vital signs with their medians or appropriate static values.
        Assume no assisted breathing. Skips any columns that aren't present.
        """
        r = self.copy()

        # Median (and also mean) of O2 is 97%
        # BP is Systolic: 124 and Diastolic: 70
        # Temperature is 36.7
        # Heart rate is 80
        for _ in SCICols.news_data_raw[:-2] + ["c_BP_Diastolic"]:
            if _ in r.columns:
                r[_].fillna(r[_].median(), inplace=True)

        static_fills = {
            **{col: 0 for col in SCICols.news_data_scored},
            "c_Assisted_breathing": False,
            "c_Breathing_device": "A - Air",
            "c_Oxygen_flow_rate": 0.0,
            "c_Pain": False,
            "c_Alert": False,
            "c_Nausea": False,
            "c_Vomiting_since_last_round": False,
            "c_Lying_down": False,
        }

        for col, val in static_fills.items():
            if col in r.columns:
                r[col].fillna(val, inplace=True)

        return SCIData(r)

    def impute_blood(self):
        r = self.copy()
        for _ in SCICols.blood:
            r[_].fillna(r[_].median(), inplace=True)

        return SCIData(r)

    def omit(self, cols):
        return SCIData(self.drop(cols, errors="ignore", axis=1))

    def omit_news(self):
        return self.omit(SCICols.news_data)

    def omit_blood(self):
        return self.omit(SCICols.blood)

    def omit_vbg(self):
        return self.omit(SCICols.vbg)

    def omit_news_extras(self):
        return self.omit(SCICols.news_data_extras)

    def omit_redundant(self):
        return self.omit(
            SCICols.admin
            + SCICols.duration
            + SCICols.wards
            + SCICols.ward_los
            + SCICols.operations
            + SCICols.hrg
            + SCICols.gp
            + SCICols.news
            + [
                "Area",
                "AandEArrivalTime",
                "AandEDepartureTime",
                "c_NEWS_risk",
                "AssessmentAreaDischarge",
                "Covid",
                "ReadmissionTimespan",
                "ReadmittedTimespan",
                "CriticalReadmitted",
            ]
        )

    def omit_ae(self):
        return self.omit(
            ["AandEMainDiagnosis", "AandEPatientGroupDescription", "AandELocation"]
        )

    def raw_news(self):
        return SCIData(self.drop(SCICols.news_data_scored, axis=1, errors="ignore"))

    def scored_news(self):
        return SCIData(self.drop(SCICols.news_data_raw, axis=1, errors="ignore"))

    def encode_onehot(self, cols, prefix, drop_old=True):
        """Given a set of columns, one-hot encodes them and concatenates to the DataFrame
        :param cols: The columns to encode
        :param prefix: What to call the new columns
        :param drop_old: Whether to drop the original columns after concatenating the encoded ones
        :returns: SCIData instance with the new columns
        """
        if not (set(cols) <= set(self.columns)):
            raise KeyError(
                f"Some of the given columns do not exist: {set(cols)-set(self.columns)}"
            )

        encoded = (
            pd.get_dummies(self[cols].stack(), prefix=prefix).groupby(level=0).any()
        )

        r = pd.concat([self, encoded.astype(int)], axis=1)
        if drop_old:
            r = r.drop(cols, axis=1)

        return SCIData(r)

    def encode_ccs_onehot(self, drop_old=True):
        """Given DataFrame augmented with CCS groupings (CCS, SHMI, or HSMR), one-hot encodes them"""

        r = self.encode_onehot(SCICols.diagnoses, "CCS", drop_old)
        r = r.rename(
            columns={_: str(_) for _ in r.columns if str(_).startswith("CCS_")}
        )

        return SCIData(r)

    def derive_critical_event(self, within=None, col_name="CriticalEvent"):
        """Determines the patients' critical event outcome.
        :param within: Time since admission to consider a critical event. E.g., 1.0 means it occurred within 24 hours, otherwise lived past 24 hours
        :return: New SCIData instance with the new feature added
        """
        temp = self.derive_death_within(within=within).derive_critical_care(
            within=within
        )
        col = temp.DiedWithinThreshold | temp.CriticalCare

        r = self.copy()
        r[col_name] = col

        return SCIData(r)

    def mandate(self, cols):
        return SCIData(
            self.dropna(how="any", subset=set(cols).intersection(self.columns),)
        )

    def mandate_diagnoses(self):
        return self.mandate(SCICols.diagnoses_ccs_encoded + SCICols.diagnoses)

    def mandate_news(self):
        return self.mandate(SCICols.news_data)

    def mandate_blood(self):
        return self.mandate(SCICols.blood)

    def preprocess_from_params(self, **kwargs):
        r = self
        for _ in ["news", "blood"]:
            v = kwargs.get(f"{_}_columns")
            if v is not None:
                r = getattr(r, f"{v}_{_}")()

        v = kwargs.get("news_format")
        if v == "raw":
            r = r.raw_news()
        elif v == "scored":
            r = r.scored_news()

        v, e = kwargs.get("diagnoses_grouping"), kwargs.get("diagnoses_onehot", False)
        if v is not None:
            r = getattr(r, f"augment_{v}")(onehot=e)

        r = r.mandate_diagnoses()
        return r.xy()

    def xy(self, x=[], dtype=None, dropna=False, outcome="DiedDuringStay"):
        X = (
            self[x]
            if len(x)
            else self.drop(
                SCICols.outcome + SCICols.mortality + [outcome], axis=1, errors="ignore"
            )
        )
        if dtype is not None:
            X = X.astype(dtype)
        y = self[outcome].copy()
        if dropna:
            X = X.dropna(how='any')
            y = y[X.index]
        return SCIData(X), y


def justify(df, invalid_val=np.nan, axis=1, side="left"):
    """
    Justifies a 2D array

    Parameters
    ----------
    df : DataFrame
        Input DataFrame to be justified
    axis : int
        Axis along which justification is to be made
    side : str
        Direction of justification. It could be 'left', 'right', 'up', 'down'
        It should be 'left' or 'right' for axis=1 and 'up' or 'down' for axis=0.

    """
    if invalid_val is np.nan:
        mask = df.notna().values
    else:
        mask = df != invalid_val

    justified_mask = np.sort(mask, axis=axis)
    if (side == "up") | (side == "left"):
        justified_mask = np.flip(justified_mask, axis=axis)
    out = np.full(df.shape, invalid_val).astype("O")
    if axis == 1:
        out[justified_mask] = df.values[mask]
    else:
        out.T[justified_mask.T] = df.values.T[mask.T]
    return pd.DataFrame(out, columns=df.columns)


class NEWS:
    @staticmethod
    def risk_scale(score):
        if score <= 4:
            return "Low"
        elif score <= 6:
            return "Medium"
        elif score >= 7:
            return "High"

    @staticmethod
    def SpO2_1_Scale_reverse(score):
        if score == 2:
            return 93
        elif score == 1:
            return 95
        elif score == 0:
            return 98
        elif score == 3:
            return np.nan

    @staticmethod
    def SpO2_1_Scale(sat):
        if sat >= 96:
            return 0
        elif sat >= 94:
            return 1
        elif sat >= 92:
            return 2
        else:
            return 3

    @staticmethod
    def SpO2_2_Scale(sat, assisted):
        if assisted:
            if sat >= 97:
                return 3
            elif sat >= 95:
                return 2
            elif sat >= 93:
                return 1
            elif sat >= 88:
                return 0
        if sat >= 93:
            return 0
        elif sat >= 86:
            return 1
        elif sat >= 84:
            return 2
        else:
            return 3

    @staticmethod
    def systolic_scale(sys):
        if sys >= 220:
            return 3
        elif (sys <= 219) and (sys >= 111):
            return 0
        elif (sys <= 110) and (sys >= 101):
            return 1
        elif (sys <= 100) and (sys >= 91):
            return 2
        elif sys <= 90:
            return 3

    @staticmethod
    def systolic_scale_reverse(score):
        if score == 2:
            return 95
        elif score == 1:
            return 105
        elif score == 0:
            return 165
        else:
            return np.nan

    @staticmethod
    def respiration_scale_reverse(score):
        if score == 1:
            return 10
        elif score == 2:
            return 23
        elif score == 0:
            return 16
        else:
            return np.nan

    @staticmethod
    def respiration_scale(rate):
        if rate >= 25:
            return 3
        elif rate >= 21:
            return 2
        elif rate >= 12:
            return 0
        elif rate >= 9:
            return 1
        else:
            return 3

    @staticmethod
    def temperature_scale(temperature):
        if temperature >= 39.1:
            return 2
        elif temperature >= 38.1:
            return 1
        elif temperature >= 36.1:
            return 0
        elif temperature >= 35.1:
            return 1
        else:
            return 3

    @staticmethod
    def temperature_scale_reverse(score):
        if score == 2:
            return 39.1
        elif score == 0:
            return 37.0
        elif score == 1:
            return 37.0
        elif score == 3:
            return 35.0

    @staticmethod
    def heartrate_scale(rate):
        if rate >= 131:
            return 3
        elif rate >= 111:
            return 2
        elif rate >= 91:
            return 1
        elif rate >= 51:
            return 0
        elif rate >= 41:
            return 1
        else:
            return 3

    @staticmethod
    def heartrate_scale_reverse(score):
        if score == 1:
            return 70
        elif score == 0:
            return 70
        elif score == 2:
            return 120
        else:
            return np.nan


class SCICols:
    redundant = [
        "AdmissionFYMonth",
        "YearAdmit",
        "MonthAdmit",
        "AdmitHour",
        "AdmitWeek",
        "YearDisch",
        "MonthDisch",
        "DischHour",
        "DischWeek",
        "AdmissionFyYear",
        "AdmitDay",
        "DischargeFyYear",
        "DischDay",
        "AdmissionConsultant",
        "LastConsultant",
        "PatientType",
        "IntendedManagement",
        "SpellTariff",
        "WordingAfterAdmission",
        "WordingBeforeAdmission",
        "AdmissionsDate",
        "AdmissionDate",
        "aLTClientGUID",
        "ClientGUID",
        "AESerial",
        "PatientNoSeq",
        "AllCFS.1",
    ]

    admin = [
        "SpellSerial",
        "PatientNumber",
        "SEQ",
    ]

    patient = [
        "Female",
        "Age",
        "AgeBand",
        "Area",
    ]

    duration = [
        "AdmissionDateTime",
        "DischargeDateTime",
        "TotalLOS",
        "LOSBand",
        "Over7Days",
        "Over14Days",
    ]

    admission = [
        "ElectiveAdmission",
        "AdmissionMethodDescription",
        "AssessmentAreaAdmission",
        "AssessmentAreaDischarge",
        "AdmissionSpecialty",
        "LastSpecialty",
    ]

    wards = [
        "AdmitWard",
        "NextWard2",
        "NextWard3",
        "NextWard4",
        "NextWard5",
        "NextWard6",
        "NextWard7",
        "NextWard8",
        "NextWard9",
        "DischargeWard",
    ]

    ward_los = [
        "AdmitWardLOS",
        "NextWard2LOS",
        "NextWard3LOS",
        "NextWard4LOS",
        "NextWard5LOS",
        "NextWard6LOS",
        "NextWard7LOS",
        "NextWard8LOS",
        "NextWard9LOS",
        "DischargeWardLOS",
    ]

    mortality = [
        "DiedDuringStay",
        "DiedWithin30Days",
    ]

    outcome = [
        "CriticalEvent",
        "DiedWithin48h",
        "DiedWithin24h",
        "Readmitted",
        "CriticalCare",
    ]

    ae = [
        "AandEPresentingComplaint",
        "AandEMainDiagnosis",
        "AandEArrivalTime",
        "AandEDepartureTime",
        "AandELocation",
        "AandEPatientGroupDescription",
    ]

    diagnoses = [
        "MainICD10",
        "SecDiag1",
        "SecDiag2",
        "SecDiag3",
        "SecDiag4",
        "SecDiag5",
        "SecDiag6",
    ]

    diagnoses_ccs_encoded = [f"CCS_{_}" for _ in range(1, 18)]

    operations = [
        "MainOPCS4",
        "SecOper1",
        "SecOper2",
        "SecOper3",
        "SecOper4",
        "SecOper5",
        "SecOper6",
    ]

    hrg = [
        "SpellHRG",
        "HRGDesc",
    ]

    gp = [
        "PrimarySpecialtyLocalCode",
        "CareHome",
        "PCT",
        "GPPractice",
    ]

    blood = [
        "Haemoglobin",
        "Urea_serum",
        "Sodium_serum",
        "Potassium_serum",
        "Creatinine",
    ]

    vbg = [
        "PatientTemperatureVenous",
        "pCO2(POC)Venous",
        "pCO2(Tempcorrected)(POC)Venous",
        "PH(POC)Venous",
        "PH(Tempcorrected)(POC)Venous",
        "pO2(POC)Venous",
        "pO2(Tempcorrected)(POC)Venous",
    ]

    news = ["c_NEWS_score", "NewsCreatedWhen", "NewsTouchedWhen", "NewsAuthoredDtm"]

    news_data = [
        "c_Respiration_rate",
        "c_NEWS_resp_rate_score",
        "c_Assisted_breathing",
        "c_Breathing_device",
        "c_NEWS_device_air_score",
        "c_O2_saturation",
        "c_Oxygen_flow_rate",
        "c_NEWS_O2_sat_score",
        "c_Temperature",
        "c_NEWS_temperature_score",
        "c_Lying_down",
        "c_BP_Systolic",
        "c_BP_Diastolic",
        "c_NEWS_BP_score",
        "c_Heart_rate",
        "c_NEWS_heart_rate_score",
        "c_Alert",
        "c_NEWS_level_of_con_score",
        "c_Pain",
        "c_Nausea",
        "c_Vomiting_since_last_round",
    ]

    news_data_raw = [
        "c_Respiration_rate",
        "c_O2_saturation",
        "c_Temperature",
        "c_BP_Systolic",
        "c_Heart_rate",
        "c_Alert",
        "c_Assisted_breathing",
    ]

    news_data_scored = [
        "c_NEWS_resp_rate_score",
        "c_NEWS_device_air_score",
        "c_NEWS_O2_sat_score",
        "c_NEWS_temperature_score",
        "c_NEWS_BP_score",
        "c_NEWS_heart_rate_score",
        "c_NEWS_level_of_con_score",
    ]

    news_data_extras = [
        "c_Vomiting_since_last_round",
        "c_BP_Diastolic",
        "c_Lying_down",
        "c_Pain",
        "c_Oxygen_flow_rate",
        "c_Nausea",
        "c_Breathing_device",
    ]

    icd10_grouping = [
        "Chapter_No",
        "Chapter_Desc",
        "Group_Code",
        "Group_Desc",
        "ICD10_3_Code_Desc",
        # "MainICD10_3_Code",
    ]

    @staticmethod
    def ordered():
        return (
            SCICols.admin
            + SCICols.patient
            + SCICols.duration
            + SCICols.admission
            + SCICols.wards
            + SCICols.ward_los
            + SCICols.mortality
            + SCICols.ae
            + SCICols.diagnoses
            + SCICols.operations
            + SCICols.hrg
            + SCICols.gp
            + SCICols.blood
            + SCICols.vbg
            + SCICols.news
            + SCICols.news_data
        )

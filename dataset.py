import numpy as np
import pandas as pd
from dataclasses import dataclass


class SCIData(pd.DataFrame):
    """ Represents the SCI dataset and related methods to augment or filter it """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def load(cls, file, table="table"):
        """ Loads the specified HDF5 file into Pandas. 
        :param str file: The (local) file path 
        :param str table: The name of the table to load from the HDF5. By default, 'table' is used.
        :return: Instance of SCIData with the DataFrame loaded. 
        """
        return cls(data=pd.read_hdf(file, table))

    def derive_all(self, force=False):
        """ Runs all methods to derive handcrafted features from the raw dataset 
        :return: New SCIData instance with the new features added
        """
        return (
            self.derive_covid_strict(force=force)
            .derive_mortality(force=force)
            .derive_readmission(force=force)
            .derive_criticalcare(force=force)
            .derive_main_icd3code(force=force)
        )

    def derive_covid(
        self,
        covid_codes=["U07.1", "J12.8", "B97.2"],
        start_date="2020-01-01",
        force=False,
    ):
        """ Derives a binary feature indicating whether the patient had COVID-19, based on their coded diagnoses.
        :param covid_codes: The ICD-10 codes to look for. 
        :return: New SCIData instance with the new feature added
        """
        if "Covid" in self and not force:
            return self

        year_mask = self.AdmissionDateTime >= start_date
        covid_mask = self[SCICols.diagnoses].isin(covid_codes).any(axis=1)

        r = self.copy()
        r["Covid"] = year_mask & covid_mask
        return SCIData(data=r)

    def derive_covid_strict(self, covid_codes=["U07.1", "J12.8", "B97.2"], force=False):
        """ Derives a binary feature indicating whether the patient had COVID-19 based on the entire ICD10 code combination.
        :return: New SCIData instance with the new feature added
        """
        if "Covid" in self and not force:
            return self

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
        force=False,
    ):
        """Determines the timespan since the patient's last admission. Bins readmissions into bands.
        :param bins: Bin edges for readmission bands
        :param labels: String labels for the returned bands
        :return: New SCIData instance with the new features added
        """
        if (
            all(
                _ in self
                for _ in ["ReadmissionTimespan", "ReadmissionBand", "Readmission"]
            )
            and not force
        ):
            return self

        bins = [pd.Timedelta(days=_) for _ in bins]

        r = self.copy()
        r["ReadmissionTimespan"] = (
            r.sort_values(["PatientNumber", "AdmissionDateTime"])
            .groupby("PatientNumber")
            .AdmissionDateTime.diff()
            .dropna()
        )
        r["ReadmissionBand"] = (
            pd.cut(r.ReadmissionTimespan, bins, labels=labels)
            .astype(str)
            .replace("nan", np.nan)
        )
        r["Readmission"] = r.ReadmissionTimespan < pd.Timedelta(days=14)

        return SCIData(data=r)

    def derive_mortality(self, force=False):
        """ Determines the patients' mortality outcome. Can be ['DiedDuringStay', 'DiedWithin30Days', 'DidNotDie']
        :return: New SCIData instance with the new feature added
        """
        if "Mortality" in self and not force:
            return self

        m = self[["DiedDuringStay", "DiedWithin30Days"]].copy()
        m["DidNotDie"] = ~m.any(axis=1)

        r = self.copy()
        r["Mortality"] = m.dot(m.columns)
        return SCIData(data=r)

    def derive_criticalcare(self, critical_wards=["CCU", "HH1M"], force=False):
        """ Determines admission to critical care at any point during the spell as indicated by admission to specified wards
        :param critical_wards: The wards to search for. By default, ['CCU', 'HH1M']
        :return: New SCIData instance with the new features added
        """
        if "CriticalCare" in self and not force:
            return self

        r = self.copy()
        r["CriticalCare"] = r[SCICols.wards].isin(critical_wards).any(axis=1)

        wm = r.melt(id_vars="SpellSerial", value_vars=SCICols.wards).drop(
            "variable", axis=1
        )
        wm = wm[wm.value.isin(critical_wards)].drop_duplicates()
        r = r.merge(
            wm[wm.duplicated()],
            how="left",
            indicator=True,
            left_on="SpellSerial",
            right_on="SpellSerial",
        ).rename(columns={"_merge": "CriticalReadmission"})
        r.CriticalReadmission = r.CriticalReadmission.eq("both") | (
            r.Readmission & r.CriticalCare
        )

        return SCIData(data=r.drop("value", axis=1))

    def derive_main_icd3code(self, force=False):
        """ Derives the 3-Code from the main coded diagnosis
        :return: New SCIData instance with the new feature added
        """
        if "MainICD10_3_Code" in self and not force:
            return self

        r = self.copy()
        r["MainICD10_3_Code"] = r.MainICD10.str[:3]

        return SCIData(data=r)

    def filter_vague_diagnoses(self):
        """ Filters out ICD10 codes marked as vague per Hassani15. Remaining diagnoses are shifted to the right.
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

        vague = []
        for letter, start, stop in ranges:
            vague += [f"{letter}{str(num).zfill(2)}" for num in range(start, stop + 1)]

        r = self.copy()
        idx = r.MainICD10.str[:3].isin(vague)
        for _ in range(1, 7):
            r.loc[idx, "MainICD10"] = r.loc[idx, f"SecDiag{_}"]
            r.loc[idx, f"SecDiag{_}"] = np.nan
            idx = r.MainICD10.str[:3].isin(vague)

        r.loc[r.MainICD10.str[:3].isin(vague), "MainICD10"] = np.nan

        return SCIData(data=r)


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
        "AdmissionWardLOS",
        "NextWardLOS2",
        "NextWardLOS3",
        "NextWardLOS4",
        "NextWardLOS5",
        "NextWardLOS6",
        "NextWardLOS7",
        "NextWardLOS8",
        "NextWardLOS9",
        "DischargeWardLOS",
    ]

    outcome = [
        "DiedDuringStay",
        "DiedWithin30Days",
        "DischargeDestination",
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
        "Urea(serum)",
        "Sodium(serum)",
        "Potassium(serum)",
        "Creatinine",
    ]

    vgb = [
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

    @staticmethod
    def ordered():
        return (
            SCICols.admin
            + SCICols.patient
            + SCICols.duration
            + SCICols.admission
            + SCICols.wards
            + SCICols.ward_los
            + SCICols.outcome
            + SCICols.ae
            + SCICols.diagnoses
            + SCICols.operations
            + SCICols.hrg
            + SCICols.gp
            + SCICols.blood
            + SCICols.vgb
            + SCICols.news
            + SCICols.news_data
        )


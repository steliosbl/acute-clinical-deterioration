import numpy as np
import pandas as pd
import logging, argparse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

from dataset import SCICols

def process_SCI(xlsx: pd.DataFrame) -> pd.DataFrame:
    df = xlsx.copy()

    # Remove spaces from column names
    df.columns = df.columns.str.replace(" ", "")

    # Drop redundant columns
    df = df.drop(SCICols.redundant, axis=1)

    col_counts = {col: df[col].value_counts() for col in df.columns}

    # Drop c_ prefixed columns with no values in them
    df = df.drop(
        [
            col
            for col, count in col_counts.items()
            if count.size <= 2 and col.startswith("c_")
        ],
        axis=1,
    )

    # Replace NoNSw2d with np.NAN in all applicable columns
    df = df.replace({"NoNSw2d": np.NAN, "NoObW2D": np.NAN, "Noobw2d": np.NAN})

    df['c_Breathing_device'] = df.c_O2_device_or_air.copy()
    
    # Turn these 2-value string columns into binary
    binarise = {
        "AdmissionType": "Elective",
        "AdmissionArea": "Medical Assessment Area",
        "DischargeArea": "Assessment Area Discharge",
        "c_Nausea": "1 - Nausea present",
        "c_Vomiting": "1 - Vomiting since last round",
        "Gender": "Female",
        "c_O2_device_or_air": "A - Air",
        "c_Patient_Position": "Lying",
        "c_Level_of_consciousness": "A - Alert",
        **{
            _: "Yes"
            for _ in [
                "Over7Days",
                "Over14Days",
                "CareHome",
                "DiedDuringStay",
                "DiedWithin30Days",
            ]
        },
    }

    for col, true in binarise.items():
        df[col] = (
            df[col]
            .apply(true.__eq__)
            .apply(lambda x: np.nan if x == NotImplemented else x)
        )

    df["c_O2_device_or_air"] = df["c_O2_device_or_air"].replace(
        {True: False, False: True}
    )

    # Convert CFS dates
    df.Noofminsafteradmission = df.AdmissionDateTime + pd.to_timedelta(
        df.Noofminsafteradmission, unit="m"
    )
    df.NoMinsBeforeadmission = df.AdmissionDateTime + pd.to_timedelta(
        df.NoMinsBeforeadmission, unit="m"
    )

    # Rename some of the binarised columns for better clarity
    df = df.rename(
        columns={
            "AdmissionType": "AdmittedAfterAEC",
            "AdmissionArea": "AssessmentAreaAdmission",
            "DischargeArea": "AssessmentAreaDischarge",
            "c_Vomiting": "VomitingSinceLastRound",
            "SpellDischargeDate": "DischargeDateTime",
            "Gender": "Female",
            "DischargeDestinationDescription": "DischargeDestination",
            "c_O2_device_or_air": "c_AssistedBreathing",
            "c_Patient_Position": "c_LyingDown",
            "c_Breathing_device": 'c_BreathingDevice',
            "c_Level_of_consciousness": "AVCPU_Alert",
            "c_Heart_rate":'HeartRate',
            'c_BP_Diastolic': 'DiastolicBP',
            'c_BP_Systolic': 'SystolicBP',
            "Noofminsafteradmission": "CFSAfterAdmissionDate",
            "NoMinsBeforeadmission": "CFSBeforeAdmissionDate",
            "CFSBeforeadmission": "CFSBeforeAdmission",
            'AdmissionWardLOS': 'AdmitWardLOS',
            'NextWardLOS2': 'NextWard2LOS',
            'NextWardLOS3': 'NextWard3LOS',
            'NextWardLOS4': 'NextWard4LOS',
            'NextWardLOS5': 'NextWard5LOS',
            'NextWardLOS6': 'NextWard6LOS',
            'NextWardLOS7': 'NextWard7LOS',
            'NextWardLOS8': 'NextWard8LOS',
            'NextWardLOS9': 'NextWard9LOS',
            'DischargeWardLOS': 'DischargeWardLOS',
            "Urea(serum)" : "Urea_serum",
            "Sodium(serum)" : "Sodium_serum",
            "Potassium(serum)" : "Potassium_serum",
        }
    )

    df.columns = [_[2:] if _ .startswith('c_') else _ for _ in df.columns]

    df.Pain = df.Pain.map(
        {
            "0 - No pain": 0,
            "1 - Mild pain": 1,
            "2 - Moderate pain": 2,
            "3 - Severe pain": 3,
        }
    )

    # Convert NEWS dates
    datetimes = ["NewsCreatedWhen", "NewsTouchedWhen", "NewsAuthoredDtm"]
    df[datetimes] = df[datetimes].apply(pd.to_datetime, errors="coerce")

    # Convert blood results
    # Ignore certain non-numeric entries as they make up less than 0.001%
    numeric = [
        "Urea_serum",
        "Sodium_serum",
        "Potassium_serum",
        "Creatinine",
        "pO2(POC)Venous",
    ]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")

    # Drop duplicates based on serial code
    df = df.sort_values("SEQ", ascending=False).drop_duplicates("SpellSerial")

    df = df.replace("nan", np.nan)

    df = df[df.AdmissionDateTime >= pd.Timestamp('2015-01-01')]

    df = df[~df.AdmissionMethodDescription.isin(['TRAUMA ELECTIVE ADM', 'MATERNITY ANTE NATAL', 'WAITING LIST'])]

    return df[SCICols.ordered()].reset_index(drop=True)

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", help="Output filename/path", type=str, default='sci_processed.h5')
parser.add_argument("-t", "--table", help='Name of table to place data into (for HDF5 only)', type=str, default='table')
parser.add_argument("-f", "--format", help="Output format. Must be one of: hdf5, csv. Default: hdf5", type=str, default='hdf5')
parser.add_argument("filename", help="Input filename/path", type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    logging.info(f'Reading file: {args.filename}')
    xlsx = pd.read_excel(args.filename)

    logging.info(f'Processing file')
    df = process_SCI(xlsx)

    if args.format == 'csv':
        logging.info(f'Writing CSV to {args.output}')
        df.to_csv(args.output)
    else:
        if args.format != 'hdf5':
            logging.error(f'Unrecognised output format. Defaulting to HDF5')
        logging.info(f'Writing HDF5 to {args.output}/{args.table}')
        df.to_hdf(args.output, key=args.table)



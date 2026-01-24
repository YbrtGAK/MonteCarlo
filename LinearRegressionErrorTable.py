"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            Convective boiling bench : Linear regression Error table
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""
This script provide a function allowing to select a Keithley channel and get the 
calibration law and the sensor root mean squares error.
"""

# Imports
import pandas as pd

# Generation of udf - table of uncertainties of the measurements
def generateUdf(df_meas : pd.DataFrame) -> pd.DataFrame :
    # Get the excel in a table
    excel_path = r"C:\Users\yberton\OneDrive - INSA Lyon\Expérimental\Acquisition\Etalonnage\Etalonnage.xlsm"
    df_excel_pressure = pd.read_excel(excel_path,sheet_name="capteurs de pression")
    df_excel_thermocouple = pd.read_excel(excel_path, sheet_name="thermocouples")
    udf = pd.DataFrame(index = df_meas.index, columns = df_meas.columns)


    # Get matching thermocouples 

    dict_canal_thermocouple = {}
    for i in range(len(df_excel_thermocouple)):
        for j in range(len(df_meas.columns)) : 
            if str(df_excel_thermocouple['n° canal'][i]) in df_meas.columns[j] : 
                    udf[df_meas.columns[j]] = [df_excel_thermocouple['RMSE [°C]'][i] for k in range(len(df_meas))]

    return(udf)
     
if __name__ == "__main__":
     
     from utilities.data.lvm import lvm_to_df
     df_meas = lvm_to_df(r".\exemples\PPh_359.lvm")
     udf = generateUdf(df_meas) 
     print('final')
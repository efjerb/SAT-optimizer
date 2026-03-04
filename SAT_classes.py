import logging
from enum import Enum
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import plotly.express as px
from bs4 import BeautifulSoup
from numba import njit
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.population import Population
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from scipy.optimize import OptimizeResult, differential_evolution, minimize
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


class Method(Enum):
    EVOLUTION = 1
    GBM = 2
    GENETIC = 3
    PSO = 4
    BAYESIAN = 5

class Zone:
    def __init__(self, name):
        self.name = name
        self.data = pd.DataFrame()
        self.area = 0
        self.volume = 0

    def set_area(self, area):
        self.area = area

    def set_volume(self, volume):
        self.volume = volume

    def add_data(self, data: pd.DataFrame):
        self.data = pd.concat([self.data, data], axis=1)
        self.fix_columns()
        self.aggregate_columns()

    def check_data_point(self):
        required_types = [
            "brick:Supply_Air_Flow_Sensor",
            "brick:Damper_Position_Command",
            "brick:Zone_Air_Cooling_Temperature_Setpoint",
            "brick:Zone_Air_Temperature_Sensor",
            "brick:Min_Supply_Air_Flow_Setpoint_Limit",
            "brick:Max_Supply_Air_Flow_Setpoint_Limit",
            "brick:CO2_part"
        ]

        if not all([req in self.data.columns for req in required_types]):
            response = False
        else:
            response = True

        return response

    def fix_temperatures(self):
        """
        This function fixes errors with temperatures. E.g., some temperature sensors are always 0, and in some periods temperature sensors are fixed at 30 °C.
        """
        # Remove temperature columns, where the temperature is 0 for more than X% of the time:
        threshold = 0.2
        if "brick:Zone_Air_Temperature_Sensor" not in self.data.columns:
            logging.debug(f"Missing brick:Zone_Air_Temperature_Sensor in {self.name}")
            return
        if type(self.data["brick:Zone_Air_Temperature_Sensor"]) is pd.Series:
            zero_prevalence = self.data["brick:Zone_Air_Temperature_Sensor"].value_counts(normalize=True).get(0, 0)
            if zero_prevalence > threshold:
                logging.info("Removing brick:Zone_Air_Temperature_Sensor due to too many 0 °C values")
                self.data.drop(columns=["brick:Zone_Air_Temperature_Sensor"], inplace=True)
        else:
            for col in self.data["brick:Zone_Air_Temperature_Sensor"].columns:
                zero_prevalence = self.data["brick:Zone_Air_Temperature_Sensor"][col].value_counts(normalize=True).get(0, 0)
                if zero_prevalence > threshold:
                    logging.info(f"Removing {col} due to too many 0 °C values")
                    self.data.drop(columns=[("brick:Zone_Air_Temperature_Sensor", col)],inplace=True)

        # Mask periods with T >= 30:
        if type(self.data["brick:Zone_Air_Temperature_Sensor"]) is pd.Series:
            mask = self.data["brick:Zone_Air_Temperature_Sensor"] >= 30
            if mask.sum() > 0:
                logging.info(f"Masking {mask.sum()} periods with T = 30 °C")
                self.data.loc[mask, "brick:Zone_Air_Temperature_Sensor"] = np.nan

        else:
            for col in self.data["brick:Zone_Air_Temperature_Sensor"].columns:
                mask = self.data["brick:Zone_Air_Temperature_Sensor"][col] >= 30
                if mask.sum() > 0:
                    logging.info(f"Masking {mask.sum()} periods with T = 30 °C")
                    self.data.loc[mask, ("brick:Zone_Air_Temperature_Sensor",col)] = np.nan

    def fix_columns(self):
        """
        This function fixes errors with columns. E.g., some temperature sensors are always 0, and in some periods temperature sensors are fixed at 30 °C.
        
        The function is generic, so any hotfix to any column type could be added.
        """
        self.fix_temperatures()
        
    def aggregate_columns(self):
        # Remove the second column level, if present:
        if self.data.columns.nlevels == 2:
            self.data.columns = self.data.columns.droplevel(1)

        for col in self.data.columns.get_level_values(0):

            if type(self.data[col]) is pd.Series:
                continue
            elif col == "brick:Supply_Air_Flow_Setpoint":
                agg = self.data[col].sum(axis=1)
            elif col == "brick:Supply_Air_Flow_Sensor":
                agg = self.data[col].sum(axis=1)
            elif col == "brick:Min_Supply_Air_Flow_Setpoint_Limit":
                agg = self.data[col].sum(axis=1)
            elif col == "brick:Max_Supply_Air_Flow_Setpoint_Limit":
                agg = self.data[col].sum(axis=1)
            elif col == "brick:Damper_Position_Command":
                agg = self.data[col].mean(axis=1)
            elif col == "brick:CO2_part":
                agg = self.data[col].mean(axis=1)
            else:
                agg = self.data[col].mean(axis=1)
            
            # Remove the original column:
            self.data.drop(columns=[col], inplace=True)
            self.data[col] = agg

    def calculate_cooling_part(self):
        if not self.check_data_point():
            return
        cooling_part = self.data["brick:Damper_Position_Command"] - self.data["brick:CO2_part"]
        self.data.loc[:, "cooling_part"] = np.nan
        self.data.loc[cooling_part>=0, "cooling_part"] = self.data["brick:Damper_Position_Command"] - self.data["brick:CO2_part"]
        self.data.loc[cooling_part<0, "cooling_part"] = 0

        self.data.loc[:, "cooling_flow"] = np.nan
        self.data.loc[cooling_part>=0, "cooling_flow"] = self.data["brick:Min_Supply_Air_Flow_Setpoint_Limit"] + (self.data["brick:Max_Supply_Air_Flow_Setpoint_Limit"] - self.data["brick:Min_Supply_Air_Flow_Setpoint_Limit"])*self.data["cooling_part"]/100

        self.data.loc[:, "co2_flow"] = np.nan
        self.data.loc[cooling_part>=0, "co2_flow"] = self.data["brick:Min_Supply_Air_Flow_Setpoint_Limit"] + (self.data["brick:Max_Supply_Air_Flow_Setpoint_Limit"] - self.data["brick:Min_Supply_Air_Flow_Setpoint_Limit"])*self.data["brick:CO2_part"]/100

    def calculate_SAT_curve(self, points: list):
        out_points = [-5, 0, 5, 10]
        
        if len(points) != len(out_points):
            raise ValueError(f"The points should be a list of {len(out_points)} values")

        # Interpolate the points
        sat_curve = np.interp(self.data["brick:Outside_Air_Temperature_Sensor"], out_points, points)
        
        return sat_curve

    def find_mode_of_row(self, row):
        if row["brick:Supply_Air_Flow_Sensor"] < 1: # m3/h
            "inactive"
        elif row["brick:Damper_Position_Command"] < 1: # Control signal (0-100%)
            return "min"
        elif row["brick:CO2_part"] > 99: # Control signal (0-100%)
            return "CO2-max"
        elif abs(row["brick:CO2_part"] - row["brick:Damper_Position_Command"]) < 2:
            return "CO2"
        elif row["brick:Damper_Position_Command"] > 99:
            return "temp-max"
        elif not np.isnan(row["brick:Damper_Position_Command"]) and row["brick:Damper_Position_Command"] is not None:
            return "temp"
        else:
            return np.nan

    def find_mode_of_row_new(self, row):

        max_airflow = row["brick:Max_Supply_Air_Flow_Setpoint_Limit"]
        # Minimum flow from CO2 control
        co2_airflow = (row["brick:Min_Supply_Air_Flow_Setpoint_Limit"] + (row["brick:Max_Supply_Air_Flow_Setpoint_Limit"] - row["brick:Min_Supply_Air_Flow_Setpoint_Limit"])*row["brick:CO2_part"]/100)
        
        try:
            if row["brick:Supply_Air_Flow_Sensor"] < 0.01: # m3/h
                "inactive"
            elif row["brick:Supply_Air_Flow_Sensor"] <= row["brick:Min_Supply_Air_Flow_Setpoint_Limit"]:
                return "min"
            elif row["brick:CO2_part"] > 99: # Control signal (0-100%)
                return "CO2-max"
            elif abs(co2_airflow - row["brick:Supply_Air_Flow_Sensor"])/row["brick:Supply_Air_Flow_Sensor"] < 0.05:
                return "CO2"
            elif abs(row["brick:Supply_Air_Flow_Sensor"] - max_airflow)/max_airflow < 0.05:
                return "temp-max"
            elif not np.isnan(row["brick:Supply_Air_Flow_Sensor"]) and row["brick:Supply_Air_Flow_Sensor"] is not None:
                return "temp"
            else:
                return np.nan
        except Exception:
            return np.nan

    def find_modes(self):
        if any(x not in self.data.columns for x in ["brick:Damper_Position_Command", "brick:CO2_part"]):
            logging.debug(f"Missing columns in {self.name}")
            return
        # self.data["mode"] = self.data.apply(self.find_mode_of_row, axis=1)
        self.data["mode"] = self.data.apply(self.find_mode_of_row_new, axis=1)

    def summarize_modes(self, period=None):
        """
        Counts the amount of timesteps where 'mode' is in each category
        If period is specified, only counts the modes in that period.
        
        Args:
            period (list): A list of two timestamps [start, end] to filter the data. If None, the entire data is used.
        Returns:
            pd.Series: A Series containing the counts of each mode.
        """
        if period is None:
            mode_counts = self.data["mode"].value_counts()
            mode_distribution = 100*(mode_counts / self.data["mode"].notna().sum())
        else:
            mode_counts = self.data.loc[period[0]:period[1], "mode"].value_counts()
            mode_distribution = 100*(mode_counts / self.data.loc[period[0]:period[1], "mode"].notna().sum())
    
        all_modes = [
            "inactive",
            "min",
            "CO2-max",
            "CO2",
            "temp-max",
            "temp"
        ]
        for mode in all_modes:
            if mode not in mode_distribution.keys():
                mode_distribution[mode] = 0

        return mode_distribution

    def calculate_setpoint_difference(self):
        if "brick:Zone_Air_Cooling_Temperature_Setpoint" in self.data.columns and "brick:Zone_Air_Temperature_Sensor" in self.data.columns:
            self.data.loc[:, "sp_diff"] = self.data["brick:Zone_Air_Temperature_Sensor"] - self.data["brick:Zone_Air_Cooling_Temperature_Setpoint"]

    def calculate_heat_load(self):
        if "brick:Supply_Air_Flow_Sensor" not in self.data.columns or \
           "brick:Zone_Air_Temperature_Sensor" not in self.data.columns or \
           "brick:Supply_Air_Temperature_Sensor" not in self.data.columns:
            logging.debug(f"Missing columns in {self.name}")
            return
        
        
        

        
        c_p = 1006.0  # Specific heat capacity of air in J/(kg*K)
        rho = 1.2 # Density of air in kg/m3

        m_flow = rho * self.data["brick:Supply_Air_Flow_Sensor"] / 3600  # kg/s from m3/h

        # We look forward in the data
        dT = -self.data["brick:Zone_Air_Temperature_Sensor"].diff(-1)
        dt = -self.data.index.to_series().diff(-1).dt.total_seconds()
        V = self.volume
        self.data["Q_vent"] = m_flow * c_p * (self.data["brick:Supply_Air_Temperature_Sensor"] - self.data["brick:Zone_Air_Temperature_Sensor"])
        self.data["Q_xs"] = rho * V * c_p * dT / dt # Excess heat, stored in the thermal mass (only air is considered)
        self.data["Q"] = self.data["Q_xs"] - self.data["Q_vent"]
        self.data["m_flow"] = m_flow

    def calculate_alt_SAT_optimized(self):
        temperatures = np.full(len(self.data), np.nan)
        airflows = np.full(len(self.data), np.nan)
        Q_shs = np.zeros(len(self.data))

        temperatures[0] = self.data.iloc[0]["brick:Zone_Air_Temperature_Sensor"]
        airflows[0] = self.data.iloc[0]['brick:Supply_Air_Flow_Sensor']

        c_p = 1006.0
        rho = 1.2

        delta_t = -self.data.index.to_series().diff(-1).dt.total_seconds().values

        m_flow_max = (self.data["brick:Max_Supply_Air_Flow_Setpoint_Limit"] * rho / 3600).values
        m_flow_min = ((self.data["brick:Min_Supply_Air_Flow_Setpoint_Limit"] +
                       (self.data["brick:Max_Supply_Air_Flow_Setpoint_Limit"] - 
                        self.data["brick:Min_Supply_Air_Flow_Setpoint_Limit"]) * 
                        self.data["brick:CO2_part"] / 100) * rho / 3600).values

        SAT_old_list = self.data['brick:Supply_Air_Temperature_Sensor'].values
        SAT_alt_list = self.data['SAT_alt'].values
        Q_list = self.data['Q'].values
        Q_vent_list = self.data['Q_vent'].values
        T_stp_cool_list = self.data['brick:Zone_Air_Cooling_Temperature_Setpoint'].values
        T_stp_heat_list = self.data['brick:Zone_Air_Heating_Temperature_Setpoint'].values
        m_flow_old_list = (self.data['brick:Supply_Air_Flow_Sensor'] * rho / 3600).values

        for i in range(1, len(self.data)):

            prev_temp = temperatures[i-1]

            # If the previous temperature or any column in the data contains na values, skip:
            if pd.isna(prev_temp) or self.data.iloc[i].isna().any():
                temperatures[i] = self.data.iloc[i]['brick:Zone_Air_Temperature_Sensor']
                airflows[i] = self.data.iloc[i]['brick:Supply_Air_Flow_Sensor']
                continue

            SAT = SAT_alt_list[i-1] # New SAT
            SAT_old = SAT_old_list[i-1] # Original SAT
            Q = Q_list[i-1] # Current heat gain
            Q_vent = Q_vent_list[i-1] # Ventilation heat gain (original) - usually negative
            T_stp_cool = T_stp_cool_list[i-1] # Cooling setpoint
            T_stp_heat = T_stp_heat_list[i-1] # Heating setpoint
            m_flow_old = m_flow_old_list[i] # Original mass flow
            # QUESTION: Should T_stp be for this or the previous timestep?
            
            Q_sh = 0

            # Handle SAT = prev_temp, to avoid division by zero:
            if SAT != prev_temp:
                # Initially, we aim to maintain Q_vent
                m_flow_alt = Q_vent / ( c_p * (SAT - prev_temp) )

            else:
                m_flow_alt = m_flow_old

            # Cap the flow at the minimum (from CO2 control) and maximum (max. damper setting)
            m_flow_alt = min(m_flow_alt, m_flow_max[i])
            m_flow_alt = max(m_flow_alt, m_flow_min[i])

            # If the SAT is below the original SAT, the airflow should never be higher than the original
            if SAT < SAT_old:
                m_flow_alt = min(m_flow_alt, m_flow_old)

            # There is a risk that the mass flow has been capped, and we need to recalculate Q_vent:
            Q_vent_new = m_flow_alt * c_p * (SAT - prev_temp)

            # We calculate the new temperature
            T_new = prev_temp + (delta_t[i] / (rho*self.volume*c_p)) * (Q + Q_vent_new) # TODO: Apply euler method on Q_vent

            if T_new < T_stp_heat:
                # If the new temperature is below the heating setpoint, we need to "heat" it back to the setpoint, and use the additional heating in the cost function
                Q_sh = rho * self.volume * c_p * (T_stp_heat - T_new) / delta_t[i]
                T_new = T_stp_heat

            elif T_new <= T_stp_cool:
                # If the new temperature is below (or equal to) the cooling setpoint, we're happy:
                pass

            elif m_flow_alt == m_flow_max[i]:
                # If the new temperature is above the setpoint, but the airflow is maxed out, theres nothing we can do:
                pass

            else:
                # If the new temperature is above the setpoint, we need to adjust the airflow to meet the setpoint:
                if SAT >= prev_temp:
                    # With no cooling potential, the controls will max out
                    m_flow_alt = m_flow_max[i]
                else:
                    Q_extra = -c_p * (T_new - T_stp_cool) * self.volume / delta_t[i]
                    m_flow_alt = (Q_vent_new + Q_extra) / ( c_p * (SAT - prev_temp) )

                # Cap the flow at the minimum (from CO2 control) and maximum (max. damper setting)
                m_flow_alt = min(m_flow_alt, m_flow_max[i])
                m_flow_alt = max(m_flow_alt, m_flow_min[i])

                Q_vent_new = m_flow_alt * c_p * (SAT - prev_temp)
                T_new = prev_temp + (delta_t[i] / (rho*self.volume*c_p)) * (Q + Q_vent_new) # TODO: Apply euler method on Q_vent

            temperatures[i] = T_new
            airflows[i] = m_flow_alt * 3600 / rho
            Q_shs[i] = Q_sh

        self.data["airflow_alt"] = airflows
        self.data["temp_alt"] = temperatures
        self.data["Q_sh"] = Q_shs
    
    def calculate_alt_SAT(self):        
        if "brick:Supply_Air_Flow_Sensor" not in self.data.columns or \
           "brick:Zone_Air_Temperature_Sensor" not in self.data.columns or \
           "brick:Supply_Air_Temperature_Sensor" not in self.data.columns:
            logging.debug(f"Missing columns in {self.name}")
            return
        
        temperatures = self.data["brick:Zone_Air_Temperature_Sensor"].copy().values
        airflows = self.data["brick:Supply_Air_Flow_Sensor"].copy().values
        Q_shs = np.zeros(len(self.data))

        c_p = 1006.0
        rho = 1.2

        delta_t = -self.data.index.to_series().diff(-1).dt.total_seconds().values

        m_flow_max = (self.data["brick:Max_Supply_Air_Flow_Setpoint_Limit"] * rho / 3600).values
        m_flow_min = ((self.data["brick:Min_Supply_Air_Flow_Setpoint_Limit"] +
                       (self.data["brick:Max_Supply_Air_Flow_Setpoint_Limit"] - 
                        self.data["brick:Min_Supply_Air_Flow_Setpoint_Limit"]) * 
                        self.data["brick:CO2_part"] / 100) * rho / 3600).values

        SAT_old_list = self.data['brick:Supply_Air_Temperature_Sensor'].values
        SAT_alt_list = self.data['SAT_alt'].values
        Q_list = self.data['Q'].values
        Q_vent_list = self.data['Q_vent'].values
        T_stp_cool_list = self.data['brick:Zone_Air_Cooling_Temperature_Setpoint'].values
        T_stp_heat_list = self.data['brick:Zone_Air_Heating_Temperature_Setpoint'].values
        m_flow_old_list = (self.data['brick:Supply_Air_Flow_Sensor'] * rho / 3600).values
        
        temperatures, airflows, Q_shs = numba_loop(self.volume, temperatures, airflows, Q_shs, c_p, rho, delta_t, m_flow_max, m_flow_min, SAT_old_list, SAT_alt_list, Q_list, Q_vent_list, T_stp_cool_list, T_stp_heat_list, m_flow_old_list)

        self.data["airflow_alt"] = airflows
        self.data["temp_alt"] = temperatures
        self.data["Q_sh"] = Q_shs

    def find_alt_mode_of_row(self, row):

        max_airflow = row["brick:Max_Supply_Air_Flow_Setpoint_Limit"]
        # Minimum flow from CO2 control
        co2_airflow = (row["brick:Min_Supply_Air_Flow_Setpoint_Limit"] + (row["brick:Max_Supply_Air_Flow_Setpoint_Limit"] - row["brick:Min_Supply_Air_Flow_Setpoint_Limit"])*row["brick:CO2_part"]/100)
        try:
            if row["airflow_alt"] < 0.01:
                # Protection for div by 0
                return "inactive"
            elif row["airflow_alt"] <= row["brick:Min_Supply_Air_Flow_Setpoint_Limit"]:
                return "min"
            elif row["brick:CO2_part"] > 99: # Control signal (0-100%)
                return "CO2-max"
            elif abs(co2_airflow - row["airflow_alt"])/row["airflow_alt"] < 0.05:
                return "CO2"
            elif abs(row["airflow_alt"] - max_airflow)/max_airflow < 0.05:
                return "temp-max"
            elif not np.isnan(row["airflow_alt"]) and row["airflow_alt"] is not None:
                return "temp"
            else:
                return np.nan
        except Exception:
            return np.nan
        
    def find_alt_modes(self):
        if any(x not in self.data.columns for x in ["brick:Damper_Position_Command", "brick:CO2_part", "airflow_alt"]):
            logging.debug(f"Missing columns in {self.name}")
            return
        self.data["alt_mode"] = self.data.apply(self.find_alt_mode_of_row, axis=1)

    def summarize_alt_modes(self, period=None):
        """
        Counts the amount of timesteps where 'mode' is in each category
        If period is specified, only counts the modes in that period.
        
        Args:
            period (list): A list of two timestamps [start, end] to filter the data. If None, the entire data is used.
        Returns:
            pd.Series: A Series containing the counts of each mode.
        """
        if period is None:
            mode_counts = self.data["alt_mode"].value_counts()
            mode_distribution = 100*(mode_counts / self.data["alt_mode"].notna().sum())
        else:
            mode_counts = self.data.loc[period[0]:period[1], "alt_mode"].value_counts()
            mode_distribution = 100*(mode_counts / self.data.loc[period[0]:period[1], "alt_mode"].notna().sum())
            

        all_modes = [
            "inactive",
            "min",
            "CO2-max",
            "CO2",
            "temp-max",
            "temp"
        ]
        for mode in all_modes:
            if mode not in mode_distribution.keys():
                mode_distribution[mode] = 0

        return mode_distribution

class AHU:
    def __init__(self, name=None, fan_intercept=False):
        self.data = pd.DataFrame()
        self.rho = 1.2
        self.c_p = 1006
        self.hr_eps = 0.73 # Efficiency of heat exchanger
        self.cool_COP = 3
        self.PEF_electricity = 1.9
        self.PEF_heating = 0.85
        self.name = name
        self.fan_intercept = fan_intercept

    def add_data(self, data: pd.DataFrame | pd.Series):
        self.data = pd.concat([self.data, data], axis=1)
        self.aggregate_columns()

    def aggregate_columns(self):
        # Remove the second column level, if present:
        if self.data.columns.nlevels == 2:
            self.data.columns = self.data.columns.droplevel(0)

        for col in self.data.columns.get_level_values(0):

            if type(self.data[col]) is pd.Series:
                continue
            elif col == "brick:Supply_Air_Flow_Sensor":
                agg = self.data[col].sum(axis=1)
            elif col == "brick:Return_Air_Flow_Sensor":
                agg = self.data[col].sum(axis=1)
            elif col == "brick:Electric_Power_Sensor":
                agg = self.data[col].sum(axis=1)
            elif col == "brick:Heating_Thermal_Power_Sensor":
                agg = self.data[col].sum(axis=1)
            elif col == "brick:Cooling_Thermal_Power_Sensor":
                agg = self.data[col].sum(axis=1)
            elif col == "brick:Hot_Water_Flow_Sensor":
                agg = self.data[col].sum(axis=1)
            else:
                agg = self.data[col].mean(axis=1)
            
            # Remove the original column:
            self.data.drop(columns=[col], inplace=True)
            self.data[col] = agg

    def calculate_Q_heat(self, SAT: pd.Series, airflow: pd.Series):
        T_ret = self.data["brick:Return_Air_Temperature_Sensor"]
        T_out = self.data["brick:Outside_Air_Temperature_Sensor"]

        # Temperature after heat recovery:
        T_pre = T_out + self.hr_eps * (T_ret - T_out)
        T_pre = np.maximum(T_pre, self.data["brick:Preheat_Supply_Air_Temperature_Sensor"])
        T_pre = self.data["brick:Preheat_Supply_Air_Temperature_Sensor"]

        rho = 1.2
        c_p = 1006

        Q_heat = airflow / 3600 * rho * c_p * ( SAT - T_pre ) # Assuming airflow unit is m3/h

        Q_heat[Q_heat<0] = 0
        self.data["T_pre"] = T_pre
        self.data["Q_heat"] = Q_heat

    def calculate_Q_cool(self, SAT: pd.Series, airflow: pd.Series):
        T_out = self.data["brick:Outside_Air_Temperature_Sensor"]
        Q_cool = airflow / 3600 * self.rho * self.c_p * ( T_out - SAT ) # Assuming airflow unit is m3/h

        Q_cool[Q_cool<0] = 0

        self.data["Q_cool"] = Q_cool/self.cool_COP
    
    def create_fan_model(self):
        if "brick:Supply_Air_Flow_Sensor" not in self.data.columns or \
           "brick:Electric_Power_Sensor" not in self.data.columns:
            logging.error(f"Missing columns in AHU {self.name}.")
        self.fan = Fan(self.data[["brick:Supply_Air_Flow_Sensor", "brick:Electric_Power_Sensor"]], intercept=self.fan_intercept)
        self.fan.create_model()


    def calculate_fan(self, airflow: pd.Series):
        # If the fan_model has not yet been created, create it:
        if not hasattr(self, "fan"):
            self.create_fan_model()

        fan_power = self.fan.predict_power(airflow)
        self.data["airflow_alt"] = airflow
        self.data["Q_fan"] = fan_power

class Fan():
    def __init__(self, data: pd.DataFrame, intercept = False):
        if "brick:Supply_Air_Flow_Sensor" not in data.columns or \
           "brick:Electric_Power_Sensor" not in data.columns:
            raise ValueError("brick:Supply_Air_Flow_Sensor and brick:Electric_Power_Sensor are required for defining the fan but were not found in the data!")
        
        self.data = data
        self.order = 3
        self.intercept = intercept

    def create_model(self):
        
        mask = (self.data["brick:Supply_Air_Flow_Sensor"].notna() & self.data["brick:Electric_Power_Sensor"].notna() & (self.data["brick:Electric_Power_Sensor"] > 0))

        self.X = self.data["brick:Supply_Air_Flow_Sensor"][mask].values.reshape(-1,1)
        self.y = (self.data["brick:Electric_Power_Sensor"][mask].values)*1000

        self.fan_model = make_pipeline(PolynomialFeatures(self.order), LinearRegression(fit_intercept=self.intercept))

        self.fan_model.fit(self.X, self.y)
    
    def predict_power(self, airflow: pd.Series|float|int):
        if not isinstance(airflow, pd.Series):
            fan_power = self.fan_model.predict([[airflow]])
            fan_power = fan_power[0]
        else:
            fan_power = pd.Series(index=airflow.index, data=np.nan)
            fan_power[airflow.notna()] = self.fan_model.predict(airflow[airflow.notna()].values.reshape(-1,1))
        return fan_power

    def plot_fit(self): 

        X_pred = np.linspace(0,self.X.max(),100).reshape(-1,1)
        y_pred = self.fan_model.predict(X_pred)

        fig = px.scatter(x=self.X.reshape(-1), y=self.y)
        fig.add_scatter(x=X_pred.flatten(), y=y_pred, mode="lines", name="Prediction")

        # Add the regression expression to the plot
        coefficients = self.fan_model.named_steps["linearregression"].coef_
        intercept = self.fan_model.named_steps["linearregression"].intercept_

        terms = " + ".join([f"{coefficients[i]:.2e}x^{i}" for i in range(1, self.order + 1)])
        expression = f"y = {intercept:.2f} + {terms}"
        fig.add_annotation(
            x=0.5,
            y=0.95,
            xref="paper",
            yref="paper",
            text=expression,
            showarrow=False,
            font=dict(size=12)
        )
        # Add r_squared to the plot
        r_squared = self.fan_model.score(self.X, self.y)
        fig.add_annotation(
            x=0.5,
            y=0.9,

            xref="paper",
            yref="paper",
            text=f"R² = {r_squared:.3f}",
            showarrow=False,
            font=dict(size=12)
        )
        fig.update_layout(title="Fan Power vs. Airflow for HTRX_VEN01", xaxis_title="Airflow [m³/h]", yaxis_title="Fan Power [W]")
        fig.show()


class Fan_Pressure(Fan):
    """
    A modification of the Fan model, which takes the static pressure into account.
    """

    def __init__(self, data: pd.DataFrame):
        if (
            "brick:Supply_Air_Flow_Sensor" not in data.columns
            or "brick:Supply_Fan_Electric_Power_Sensor" not in data.columns
            or "brick:Supply_Air_Static_Pressure_Sensor" not in data.columns
        ):
            raise ValueError(
                "brick:Supply_Air_Flow_Sensor, brick:Supply_Fan_Electric_Power_Sensor and brick:Supply_Air_Static_Pressure_Sensor are required for defining the fan but were not found in the data!"
            )
        
        self.power_type = "brick:Supply_Fan_Electric_Power_Sensor"

        self.data = data
        self.order = 3

    def create_model(self):
        mask = (
            self.data["brick:Supply_Air_Flow_Sensor"].notna()
            & self.data[self.power_type].notna()
            & self.data["brick:Supply_Air_Static_Pressure_Sensor"].notna()
            & (self.data[self.power_type] > 0)
        )
        
        self.X = np.vstack([
            1                                                         * self.data["brick:Supply_Air_Static_Pressure_Sensor"][mask].values,
            self.data["brick:Supply_Air_Flow_Sensor"][mask].values    * self.data["brick:Supply_Air_Static_Pressure_Sensor"][mask].values,
            self.data["brick:Supply_Air_Flow_Sensor"][mask].values**2 * self.data["brick:Supply_Air_Static_Pressure_Sensor"][mask].values,
            self.data["brick:Supply_Air_Flow_Sensor"][mask].values**3 * self.data["brick:Supply_Air_Static_Pressure_Sensor"][mask].values,
            self.data["brick:Supply_Air_Flow_Sensor"][mask].values**4 * self.data["brick:Supply_Air_Static_Pressure_Sensor"][mask].values
        ]).T  # Transpose to get shape (n_samples, n_features)

        self.y = (self.data[self.power_type][mask].values) * 1000

        self.fan_model = LinearRegression(fit_intercept=False)

        self.fan_model.fit(self.X, self.y)

        print(self.fan_model.score(self.X, self.y))
    
    def predict_power(self, airflow: pd.Series|float|int):
        if not isinstance(airflow, pd.Series):
            fan_power = self.fan_model.predict([[airflow]])
            fan_power = fan_power[0]
        else:
            fan_power = pd.Series(index=airflow.index, data=np.nan)
            fan_power[airflow.notna()] = self.fan_model.predict(airflow[airflow.notna()].values.reshape(-1,1))
        return fan_power

    def plot_fit(self): 
        mask = (
            self.data["brick:Supply_Air_Flow_Sensor"].notna()
            & self.data[self.power_type].notna()
            & self.data["brick:Supply_Air_Static_Pressure_Sensor"].notna()
            & (self.data[self.power_type] > 0)
        )

        X_pred = np.vstack([
            1                                                         * self.data["brick:Supply_Air_Static_Pressure_Sensor"][mask].values,
            self.data["brick:Supply_Air_Flow_Sensor"][mask].values    * self.data["brick:Supply_Air_Static_Pressure_Sensor"][mask].values,
            self.data["brick:Supply_Air_Flow_Sensor"][mask].values**2 * self.data["brick:Supply_Air_Static_Pressure_Sensor"][mask].values,
            self.data["brick:Supply_Air_Flow_Sensor"][mask].values**3 * self.data["brick:Supply_Air_Static_Pressure_Sensor"][mask].values,
            self.data["brick:Supply_Air_Flow_Sensor"][mask].values**4 * self.data["brick:Supply_Air_Static_Pressure_Sensor"][mask].values
        ]).T
        y_pred = self.fan_model.predict(X_pred)

        fig = px.scatter(
            x=self.data["brick:Supply_Air_Flow_Sensor"].values,
            y=self.data[self.power_type] * 1000,
            color=self.data["brick:Supply_Air_Static_Pressure_Sensor"].values,

            
        )
        fig.add_scatter(
            x=self.data["brick:Supply_Air_Flow_Sensor"][mask].values,
            y=y_pred,
            name="Prediction",
            mode="markers",
            marker=dict(
                symbol="x",
                # size=16,
                # cmax=39,
                # cmin=0,
                color=self.data["brick:Supply_Air_Static_Pressure_Sensor"][mask].values,
                colorbar=dict(title=dict(text="Colorbar")),
                colorscale="Viridis",
            ),
            hovertemplate="Airflow: %{x}<br>Power: %{y}<br>Pressure: %{marker.color}"

        )

        # Add the regression expression to the plot
        # coefficients = self.fan_model.named_steps["linearregression"].coef_
        # intercept = self.fan_model.named_steps["linearregression"].intercept_

        # terms = " + ".join([f"{coefficients[i]:.2e}x^{i}" for i in range(1, self.order + 1)])
        # expression = f"y = {intercept:.2f} + {terms}"
        # fig.add_annotation(
        #     x=0.5,
        #     y=0.95,
        #     xref="paper",
        #     yref="paper",
        #     text=expression,
        #     showarrow=False,
        #     font=dict(size=12)
        # )
        # Add r_squared to the plot
        r_squared = self.fan_model.score(self.X, self.y)
        fig.add_annotation(
            x=0.5,
            y=0.9,

            xref="paper",
            yref="paper",
            text=f"R² = {r_squared:.3f}",
            showarrow=False,
            font=dict(size=12)
        )
        fig.update_layout(title="Fan Power vs. Airflow for HTRX_VEN01", xaxis_title="Airflow [m³/h]", yaxis_title="Fan Power [W]")
        fig.show()


@njit
def numba_loop(volume, temperatures, airflows, Q_shs, c_p, rho, delta_t, m_flow_max, m_flow_min, SAT_old_list, SAT_alt_list, Q_list, Q_vent_list, T_stp_cool_list, T_stp_heat_list, m_flow_old_list):
    for i in range(1, len(temperatures)):

        prev_temp = temperatures[i-1]

        # If the previous temperature or any column in the data contains na values, skip:
        if prev_temp is None or np.isnan(prev_temp) or np.isnan(SAT_alt_list[i-1]) or np.isnan(Q_list[i-1]) or np.isnan(Q_vent_list[i-1]) or np.isnan(T_stp_cool_list[i-1]) or np.isnan(T_stp_heat_list[i-1]) or np.isnan(m_flow_old_list[i]):
            continue

        SAT = SAT_alt_list[i-1] # New SAT
        SAT_old = SAT_old_list[i-1] # Original SAT
        Q = Q_list[i-1] # Current heat gain
        Q_vent = Q_vent_list[i-1] # Ventilation heat gain (original) - usually negative
        T_stp_cool = T_stp_cool_list[i] # Cooling setpoint
        T_stp_heat = T_stp_heat_list[i] # Heating setpoint
        m_flow_old = m_flow_old_list[i] # Original mass flow
        # QUESTION: Should T_stp be for this or the previous timestep?
        
        T_new, m_flow_alt, Q_sh = numba_timestep(volume, c_p, rho, delta_t[i], m_flow_max[i], m_flow_min[i], SAT_old, SAT, Q, Q_vent, T_stp_cool, T_stp_heat, m_flow_old, prev_temp)

        

        temperatures[i] = T_new
        if m_flow_alt is None:
            airflows[i] = np.nan
        else:    
            airflows[i] = m_flow_alt * 3600 / rho
        Q_shs[i] = Q_sh
    
    return temperatures, airflows, Q_shs

@njit
def numba_timestep(volume, c_p, rho, delta_t, m_flow_max, m_flow_min, SAT_old, SAT, Q, Q_vent, T_stp_cool, T_stp_heat, m_flow_old, prev_temp):
    # If the previous temperature or any column in the data contains na values, skip:
    if prev_temp is None or np.isnan(prev_temp) or SAT is None or np.isnan(SAT) or Q is None or np.isnan(Q) or Q_vent is None or np.isnan(Q_vent) or T_stp_cool is None or np.isnan(T_stp_cool) or T_stp_heat is None or np.isnan(T_stp_heat) or m_flow_old is None or np.isnan(m_flow_old):
        return np.nan, np.nan, np.nan

    if m_flow_old < 0:
        print("m_flow_old is negative!")
    if m_flow_max < 0:
        print("m_flow_max is negative!")
    if m_flow_min < 0:
        print("m_flow_min is negative!")

    Q_sh_add = 0 # Additional space heating [W]

    # Handle SAT = prev_temp, to avoid division by zero:
    if SAT != prev_temp:
        # Initially, we aim to maintain Q_vent
        m_flow_alt = Q_vent / ( c_p * (SAT - prev_temp) )

    else:
        m_flow_alt = m_flow_old
    
    if m_flow_alt < 0:
        m_flow_alt = m_flow_min

    # Cap the flow at the minimum (from CO2 control) and maximum (max. damper setting)
    m_flow_alt = min(m_flow_alt, m_flow_max)
    m_flow_alt = max(m_flow_alt, m_flow_min)

    # If the SAT is below the original SAT, the airflow should never be higher than the original
    if SAT < SAT_old:
        m_flow_alt = min(m_flow_alt, m_flow_old)

    # There is a risk that the mass flow has been capped, and we need to recalculate Q_vent:
    Q_vent_new = m_flow_alt * c_p * (SAT - prev_temp)

    # We calculate the new temperature
    T_new = prev_temp + (delta_t / (rho*volume*c_p)) * (Q + Q_vent_new) # TODO: Apply euler method on Q_vent

    if T_new < T_stp_heat:
        # If the new temperature is below the heating setpoint, we need to "heat" it back to the setpoint, and use the additional heating in the cost function
        Q_sh_add = rho * volume * c_p * (T_stp_heat - T_new) / delta_t
        T_new = T_stp_heat

    elif T_new <= T_stp_cool:
        # If the new temperature is below (or equal to) the cooling setpoint, we're happy:
        pass

    elif m_flow_alt == m_flow_max:
        # If the new temperature is above the setpoint, but the airflow is maxed out, theres nothing we can do:
        pass

    else:
        # If the new temperature is above the setpoint, we need to adjust the airflow to meet the setpoint:
        if SAT >= prev_temp:
            # With no cooling potential, the controls will max out
            m_flow_alt = m_flow_max
        else:
            Q_extra = -c_p * (T_new - T_stp_cool) * volume / delta_t
            m_flow_alt = (Q_vent_new + Q_extra) / ( c_p * (SAT - prev_temp) )

        # Cap the flow at the minimum (from CO2 control) and maximum (max. damper setting)
        m_flow_alt = min(m_flow_alt, m_flow_max)
        m_flow_alt = max(m_flow_alt, m_flow_min)

        Q_vent_new = m_flow_alt * c_p * (SAT - prev_temp)
        T_new = prev_temp + (delta_t / (rho*volume*c_p)) * (Q + Q_vent_new) # TODO: Apply euler method on Q_vent
    
    # Make sure that m_flow_alt is not negative
    return T_new, m_flow_alt, Q_sh_add

class SATOptimizer:
    def __init__(self, zones: list[Zone], ahus: list[AHU], workers: float, outdoor_temp: pd.Series, fixed_SAT=False, tol=1e-1, x_res = 0.1, pop_size=50):
        '''
        """
        Initialize the SATOptimizer class.

        Args:
            zones (list[Zone]): List of Zone objects to optimize.
            ahu (AHU): AHU object containing data and methods for air handling unit.
            workers (float): Number of workers for parallel processing.
            outdoor_temp (pd.Series): Outdoor temperature data.
            fixed_SAT (bool, optional): Whether to use a fixed SAT value. Defaults to False.
            tol (float, optional): Tolerance for optimization. Defaults to 1e-1.
            x_res (float, optional): Resolution for SAT optimization. Defaults to 0.1.
            pop_size (int, optional): Population size for optimization. Defaults to 30.
        """
        '''
        self.zones: list[Zone] = zones
        self.ahus: list[AHU] = ahus
        self.k1 = 1
        self.k2 = 1
        self.PEF_electricity = 1.9
        self.PEF_heating = 0.85
        self.fixed_SAT = fixed_SAT
        self.outdoor_temp = outdoor_temp
        self.workers = workers
        self.tol = tol
        self.results = []
        self.x_res = x_res
        self.pop_size = pop_size

        # Make sure that indexes in the ahu and zones match:
        for ahu in self.ahus:
            ahu.data = ahu.data.reindex(self.zones[0].data.index)

    def set_parameters(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def set_primary_energy_factors(self, PEF_electricity, PEF_heating):
        self.PEF_electricity = PEF_electricity
        self.PEF_heating = PEF_heating

    def calculate_costs(self, params):
        logger = logging.getLogger(__name__)
        logger.debug(f"Current SAT_alt: {params}")
        if len(params) == 1:
            SAT_alt = params[0]
        else:
            SAT_alt = self.calculate_SAT_curve(params)
        time_above_setpoint = 0
        Q_sh = 0
        Q_heat_coil = 0
        Q_cool_coil = 0
        airflow_alt = pd.Series(index=self.ahus[0].data.index, data=0)
        total_area = 0
        for zone in self.zones:
            if len(SAT_alt) < len(zone.data.index):
                SAT_alt = np.append(SAT_alt, [SAT_alt[-1]] * (len(zone.data.index) - len(SAT_alt))) # FIXME: length of SAT_alt should always be correct!!
            zone.data["SAT_alt"] = SAT_alt  # Adjust SAT_alt for each zone
            zone.calculate_alt_SAT()

            if "temp_alt" not in zone.data.columns:
                logger.debug(f"temp_alt not found in {zone.name}")
                continue

            freq = pd.Timedelta(zone.data.index.freq).total_seconds() / 3600 # Frequency in hours

            T_above_setpoint = zone.data["temp_alt"] - zone.data["brick:Zone_Air_Cooling_Temperature_Setpoint"]
            T_above_setpoint.clip(lower=0, inplace=True)

            time_above_setpoint += (T_above_setpoint*freq).sum()*zone.area # K*h*m2
            total_area += zone.area

            Q_sh += zone.data["Q_sh"].sum()*self.PEF_heating

            airflow_alt = airflow_alt + zone.data["airflow_alt"]
        Q_fan = 0
        Q_heat_coil = 0
        Q_cool_coil = 0
        for ahu in self.ahus:
            ahu.calculate_fan(airflow_alt/len(self.ahus))
            Q_fan += ahu.data["Q_fan"].sum()*self.PEF_electricity/total_area

            ahu.calculate_Q_heat(SAT_alt, airflow_alt/len(self.ahus))
            Q_heat_coil += ahu.data["Q_heat"].sum()*self.PEF_heating/total_area

            ahu.calculate_Q_cool(SAT_alt, airflow_alt/len(self.ahus))
            Q_cool_coil += ahu.data["Q_cool"].sum()*self.PEF_electricity/total_area

        Q_sh = Q_sh/total_area

        time_above_setpoint = time_above_setpoint/total_area

        Q_tot_PE = Q_sh + Q_heat_coil + Q_cool_coil + Q_fan

        total_cost = self.k1 * time_above_setpoint + self.k2 * Q_tot_PE

        logger.debug(f"\tDegreehours above setpoint (area weighted): {time_above_setpoint},\n\tQ_sh: {Q_sh},\n\tQ_heat_coil: {Q_heat_coil},\n\tQ_cool_coil: {Q_cool_coil}")
        logger.debug(f"\tTotal cost (k1 = {self.k1}, k2 = {self.k2}): {total_cost}")

        return total_cost, Q_tot_PE, time_above_setpoint, Q_sh, Q_heat_coil, Q_cool_coil, Q_fan

    def calculate_SAT_curve(self, points: list):
        out_points = [-5, 0, 5, 10]

        if len(points) != len(out_points):
            raise ValueError(f"The points should be a list of {len(out_points)} values")

        # Interpolate the points
        sat_curve = np.interp(self.outdoor_temp, out_points, points)

        return sat_curve

    def cost_function(self, params):
        return self.calculate_costs(params)[0]

    def optimize_SAT(self, method = Method.EVOLUTION):
        if self.fixed_SAT:
            initial_guess = [22]
            bounds = [(16, 24.0)]
        else:
            initial_guess = [22, 19, 18, 16]
            bounds = [(1, 24.0),(1, 24.0),(1, 24.0),(1, 24.0)]
        logger = logging.getLogger(__name__)
        logger.info("Starting optimization")

        if self.workers > 1:
            updating='deferred'
        else:
            updating='immediate'

        if method == Method.EVOLUTION:
            result = differential_evolution(
                self.cost_function,
                bounds=bounds,
                workers=self.workers,
                tol=self.tol,
                updating=updating,
                callback=self.callback
            )
        elif method == Method.GBM:
            result = minimize(
                self.cost_function,
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds,
                tol=0.001,
                callback=self.callback_GBM
            )
        elif method == Method.PSO:
            from pyswarm import pso

            lb = [b[0] for b in bounds]
            ub = [b[1] for b in bounds]
            result, fopt = pso(
                self.cost_function,
                lb,
                ub,
                swarmsize=100,
                maxiter=50,
                minstep=self.tol,
                debug=True
            )

            return result
        else:
            raise NotImplementedError("Only differential evolution (EVOLUTION), gradient-based methods (GBM) and particle swarm optimization (PSO) allowed!")

        logger.info(f"Optimization result: {result.message}. {result.nfev} evaluations of cost function. ")

        if self.fixed_SAT:
            return result.x[0]
        else:
            return result.x

    def callback(self,x, convergence):
        cost = self.cost_function(x)
        self.results.append(x)
        logger = logging.getLogger(__name__)
        logger.info(f"Current best solution: {x}, Cost: {cost}")

    def callback_GBM(self, intermediate_result: OptimizeResult):
        cost = intermediate_result.fun
        self.results.append(intermediate_result.x)
        logger = logging.getLogger(__name__)
        logger.info(f"Current best solution: {intermediate_result.x}, Cost: {cost}")

    def optimize_SAT_MO(self):
        '''
        Multi-objective optimization of SAT setpoints using pymoo
        '''
        
        termination = HypervolumeTermination(
            period=10,
            n_max_gen=50,
            n_skip=5,
            rtol=1e-10
        )

        pool = ThreadPool(self.workers)
        runner = StarmapParallelization(pool.starmap)

        if self.fixed_SAT:
            n_var = 1
            xl = np.array([16])
            xu = np.array([24])
        else:
            n_var = 4
            xl = np.array([10, 10, 10, 10])
            xu = np.array([24, 24, 24, 24])

        
        if self.x_res is None:
            algorithm = NSGA2(pop_size=self.pop_size, seed=1)
        else:
            xl = xl/self.x_res
            xu = xu/self.x_res
            algorithm = NSGA2(pop_size=self.pop_size,
                              sampling=IntegerRandomSampling(),
                              crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                              mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                              eliminate_duplicates=True,
                              seed = 1
                              )
            

        problem = SATProblem(sat_optimizer=self, x_res = self.x_res, n_var=n_var, n_obj=2, n_ieq_constr=n_var-1, xl=xl, xu=xu, elementwise_runner=runner)
        
        res = pymoo_minimize(
            problem,
            algorithm,
            termination=termination,
            seed=1,
            # verbose=True,
            callback=MOCallback(),
        )
        return res

class SATProblem(ElementwiseProblem):
    def __init__(self, sat_optimizer: SATOptimizer, x_res: float = None, **kwargs):
        self.sat_optimizer = sat_optimizer
        self.x_res = x_res
        super().__init__(elementwise=True, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        if self.x_res is not None:
            x = x*self.x_res
        total_cost, Q_tot_PE, time_above_setpoint, Q_sh, Q_heat_coil, Q_cool_coil, Q_fan = self.sat_optimizer.calculate_costs(x)
        out["F"] = np.array([time_above_setpoint, Q_tot_PE])
        # Constrain the optimization so that x is descending
        
        g = []
        for i in range(len(x)-1):
            g.append(x[i+1] - x[i])
            
        out["G"] = g


class MOCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []
        self.old_result = None
        self.old_result_gen = None
        self.old_nds = None
        self.archive = Archive()

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        
        self.archive.add(algorithm.pop)
        
        # Calculate the best objective function value
        best_value = F.min()
        
        # Store the best value
        self.data["best"].append(best_value)
        
        # Plot the Pareto front
        from IPython.display import clear_output
        clear_output()
        self.plot_pareto_front(F, show=True)
        self.plot_convergence()
    
    def plot_pareto_front(self, F, show=False):
        nds = self.archive.nds

        F = np.array([nd.F for nd in nds])

        if self.old_nds is None:
            fig = px.scatter(x=F[:,0], y=F[:,1], labels={"x": "Degreehours above setpoint (area weighted)", "y": "Total primary energy"}, title=f"Pareto front (gen. {len(self.data['best'])}) - {len(F)} Non-Dominated Solutions").update_traces(marker=dict(color='blue'))
            fig['data'][0]['showlegend']=True
            fig['data'][0]['name']='Current front'

        else:
            old_F = np.array([nd.F for nd in self.old_nds])
            fig = px.scatter(x=old_F[:,0], y=old_F[:,1], labels={"x": "Degreehours above setpoint (area weighted)", "y": "Total primary energy"}, title=f"Pareto front (gen. {len(self.data['best'])}) - {len(F)} Non-Dominated Solutions").update_traces(marker=dict(color='grey'))
            fig['data'][0]['showlegend']=True
            fig['data'][0]['name']=f'Previous front (gen. {self.old_result_gen})'

            fig.add_scatter(x=F[:,0], y=F[:,1], mode="markers", marker=dict(color="blue"), name="Current front", showlegend=True)
            # if len(F) == len(self.old_result) and (F == self.old_result).all():
            #     fig.add_annotation(
            #         x=0.5,
            #         y=0.5,
            #         xref="paper",
            #         yref="paper",
            #         text=f"No improvement since generation {self.old_result_gen}",
            #         showarrow=False,
            #         font=dict(size=12)
            #     )
        if self.old_result is None or len(F) != len(old_F) or (F != old_F).any():
            self.old_result = F
            self.old_nds = nds
            self.old_result_gen = len(self.data["best"])
        
        html = fig.to_html()
        
        html = self.add_auto_refresh_script(html, 15)
        with open("figs/pareto_front.html", "w", encoding="UTF-8") as f:
            f.write(html)        

        if show:
            fig.show()
    
    def plot_convergence(self, show=True):
        """
        Plot the evolution of hypervolume to investigate convergence
        """

        hypervolumes = self.archive.hypervolumes

        # Define y as the relative change in hypervolume pr. generation:
        y = 100*(np.diff(hypervolumes)/hypervolumes[:-1])

        # Handle len(hypervolumes) == 1:
        if len(hypervolumes) == 1:
            return

        fig = px.line(x=np.arange(len(hypervolumes))+1, y=hypervolumes)
        fig.update_layout(title="Hypervolume convergence", xaxis_title="Generation", yaxis_title="Hypervolume")
        if show:
            fig.show()
        return fig

    def add_auto_refresh_script(self, html: str, seconds: int) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        head = soup.head
        
        if head:
            script_tag = soup.new_tag("script")
            script_tag.string = f"setInterval(function() {{ location.reload(); }}, {seconds*1000});"
            head.append(script_tag)
        
        return str(soup)
    

class Archive:

    def __init__(self) -> None:
        super().__init__()
        self.nds = Population()
        self.hypervolumes = []
        self.ref_point = np.array([1.2e+10, 1.2e+10])

    def add(self, pop):
        # merged = Population.merge(self.nds, pop)
        # I = NonDominatedSorting().do(merged.get("F"), only_non_dominated_front=True)
        # self.nds = merged[I]
        self.nds = pop

        F_nds = np.array([nd.F for nd in self.nds])

        # Calculate hypervolume of nds
        if self.hypervolumes == []:
            # Check that all F values are "beneath" the reference points. If not, pick a new point that works:
            if not (F_nds <= self.ref_point).all():
                logging.info("Reference point is too low. Adjusting...")
                self.ref_point = np.max(F_nds, axis=0)*1.1
        
        ind = HV(ref_point=self.ref_point, nds=False) # NDS = False, because we have already sorted the results
        
        F_nds = np.array([nd.F for nd in self.nds])

        self.hypervolumes.append(ind(F_nds)/len(F_nds))


class HypervolumeTermination(DefaultMultiObjectiveTermination):
    """
    Termination criterion based on the relative change in hypervolume of the Pareto front.
    """    
    def __init__(self, rtol=0.001, xtol=0.0005, cvtol=1e-8, ftol=0.005, n_skip=5, period=50, **kwargs):
        super().__init__(xtol, cvtol, ftol, n_skip, period, **kwargs)
        self.period = period
        self.n_skip = n_skip
        self.rtol=rtol
        self.archive = Archive()

    def _update(self, algorithm):
        self.archive.add(algorithm.pop)
        if len(self.archive.hypervolumes) > self.period and len(self.archive.hypervolumes) % self.n_skip == 0:
            hv = self.archive.hypervolumes
            # Relative hv changes:
            hv_rel = np.diff(hv[-(self.period+1):])/hv[-(self.period+1):-1]

            abs_hv_rel = np.abs(hv_rel)

            if np.all(abs_hv_rel < self.rtol):
                print("Hypervolume convergence reached!")
                return 1
            else:
                return super()._update(algorithm)
        else:
            return super()._update(algorithm)

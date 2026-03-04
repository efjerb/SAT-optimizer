import configparser

import pandas as pd
import psycopg2

import main.functions as f


def get_timescale_connection():
    config = configparser.ConfigParser()
    config.read('main/config.ini')

    # Get the username and password from the config file
    usr = config['DEFAULT']['usr']
    pwd = config['DEFAULT']['pwd']
    host = config['DEFAULT']['host']
    port = config['TIMESCALE']['port']
    dbname = config['TIMESCALE']['dbname']

    conn = psycopg2.connect(
        dbname=dbname,
        user=usr,
        password=pwd,
        host=host,
        port=port,
    )
    return conn


def query_timescale(conn, query):
    cur = conn.cursor()
    cur.execute(query)
    res = cur.fetchall()
    cur.close()
    return res


def query_points(conn, point_names, start, end, pivot=False, sampling=None, get_units=True, filter=None):
    '''
    This function queries points defined with the point_names variable.
    NB: The output time format is UTC-formatted to avoid double entries at shifts between DST.
    '''
    if type(point_names) == str:
        point_names = f"('{point_names}')"
    elif type(point_names) == list and len(point_names) == 1:
        point_names = f"('{point_names[0]}')"
    elif type(point_names) == list:
        point_names = tuple(point_names)
    
    query = f"""
    SELECT time, name, value, text_value, unit
    FROM measurements
    WHERE name IN {point_names} AND time >= '{start}' AND time <= '{end}'
    """

    if filter:
        query += f"AND {filter}"

    res = query_timescale(conn, query)
    
    if len(res) == 0:
        return None

    df = pd.DataFrame(res, columns=["time", "name", "value", "text_value", "unit"])

    # Remove duplicates over the time and name columns
    df = df[~df.duplicated(subset=["time", "name"])]

    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    df.index = pd.to_datetime(df.index,utc=True)

    # Handle text values by extracting them into a separate dataframe for pivoting
    if "text_value" in df.columns and df["text_value"].notnull().any():
        text_df = df[["name", "text_value"]].dropna()
        text_df = text_df.pivot(columns="name", values="text_value")
        text_df = text_df.resample(sampling).last()  # Use first non-null value for resampling
        # Forward fill NA values
        text_df.ffill(inplace=True)
        # Convert data type to string for text_df
        text_df = text_df.astype(str)
        # Drop the text_value column in df
        df = df.drop(columns=["text_value"])
    else:
        text_df = None

    df = df.dropna(subset=["value"])

    if pivot and get_units:
        # FIXME: THIS IS FUCKED!
        df = df.pivot(columns=["name", "unit"], values="value")
    elif pivot:
        df = df.pivot(columns="name", values="value")
    
    if pivot and sampling:
        df = df.resample(sampling).mean()
    
    if text_df is not None:
        df = df.join(text_df, how='outer')

    return df

def query_points_external_sampling(conn, point_names, start, end, pivot=False, sampling="15m", get_units=True, filter=None):
    '''
    This function queries points defined with the point_names variable.
    NB: The output time format is UTC-formatted to avoid double entries at shifts between DST.
    '''
    if type(point_names) == str:
        point_names = f"('{point_names}')"
    elif type(point_names) == list and len(point_names) == 1:
        point_names = f"('{point_names[0]}')"
    elif type(point_names) == list:
        point_names = tuple(point_names)
    
    if "min" in sampling:
        sampling.replace("min", "m")

    query = f"""
    SELECT time_bucket_gapfill(\'{sampling}\', time) as time,
        name,
        AVG(value){', unit' if get_units else ""}
        
    FROM measurements
    WHERE name IN {point_names} AND time >= '{start}' AND time <= '{end}'
    """

    if filter:
        query += f"AND {filter}"

    query += f"""
    GROUP BY time_bucket_gapfill(\'{sampling}\', time), name, unit
    ORDER BY time
    """

    res = query_timescale(conn, query)
    
    if len(res) == 0:
        return None

    if get_units:
        columns = ["time", "name", "value", "unit"]
    else:
        columns = ["time", "name", "value"]

    df = pd.DataFrame(res, columns=columns)
    # Remove duplicates by retrieving the mean values (in favor of the built-in drop-duplicates function, which doesn't account for NAN values)
    df = df.groupby(['time','name']).mean().reset_index()
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    df.index = pd.to_datetime(df.index,utc=True)


    if pivot and get_units:
        df = df.pivot(columns=["name", "unit"], values="value")
    elif pivot:
        df = df.pivot(columns="name", values="value")

    return df


if __name__ == "__main__":
    from pprint import pprint

    import matplotlib.pyplot as plt

    conn = get_timescale_connection()
    # df = query_points(
    #     conn,
    #     ["R04_07_N_CO201", "R04_07_V_CO201"],
    #     "2024-01-01",
    #     "2024-02-01",
    #     pivot=True,
    #     sampling="15min",
    # )

    start = "2024-01-01"
    stop = "2024-02-01"

    res = f.get_vav_opening_ids_in_zones("HTR-extra")
    df = pd.DataFrame()
    for space, dampers in res.items():
        space_name = space.split("#")[-1]
        damper_names = [damper["posId"] for damper in dampers]
        
        res_df = query_points(
            conn,
            damper_names,
            start,
            stop,
            pivot=True,
            sampling="15min"
            )
        
        if res_df is None:
            continue
        
        # res_df.columns = pd.MultiIndex.from_product([[space_name], res_df.columns])
        res_df.columns = pd.MultiIndex.from_tuples(
            [(space_name, col) for col in res_df.columns]
            )
        df = pd.concat([df, res_df], axis=1)
    pprint(df)
    # Plot the R04.01_S VAV opening
    # fig = df["R04.01"].plot()
    # plt.show()

    # Get maximum damper opening between 06 and 16 for all dampers
    print(df.between_time("06:00", "16:00").max())

    # print(df["R04_07_N_CO201"]["unit"])

    

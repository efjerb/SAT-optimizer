import json
import logging
import urllib.parse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.express as px
import requests

import main.plot_functions as pf
import main.timescaledb_connection as tdb
from SAT_classes import Zone


def sparql_to_df(g, q):
    res = g.query(q)
    df = pd.DataFrame.from_records(list(res))
    df = df.applymap(str)
    df.drop_duplicates(inplace=True)
    return df

def get_token(username, password, base_url = "http://localhost:7200"):
    auth_endpoint = urllib.parse.urljoin(base_url, "rest/login")

    # Authenticate with the repository
    payload = json.dumps({
    "username": username,
    "password": password
    })
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Send the request
    response = requests.post(auth_endpoint, data=payload, headers=headers)
    token = response.headers["Authorization"]

    return token

def run_query(repository , query, base_url = "http://localhost:7200", token = None, accept = "application/sparql-results+json"):
    ### Add authentication if needed
    sparql_endpoint = urllib.parse.urljoin(base_url, f"repositories/{repository}")
    param = {"query": query}
    if token is not None:
        headers = {"Accept": accept, "Authorization": token}
    else:
        headers = {"Accept": accept}

    res = requests.get(sparql_endpoint, params=param, headers=headers)
    return res

def fix_res(res, group_key, values):
    res = res.json()["results"]["bindings"]
    for item in res:
        for key in item.keys():
            item[key] = item[key]["value"]
    new_res = {}
    for item in res.copy():
        new_item = {}
        for key in item.keys():
            if key in values:
                new_item[key] = item[key]

        if item[group_key] in new_res.keys():
            new_res[item[group_key]].append(new_item)
        else:
            new_res[item[group_key]] = [new_item]
    return new_res

def get_radiator_output_in_space(space_tag):
    res = get_radiator_output_in_spaces()
    res = res[space_tag]
    return res

def get_radiator_output_in_spaces():
    query = f"""PREFIX ex: <https://example.com/ex#>
PREFIX fso: <https://w3id.org/fso#>
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX bot: <https://w3id.org/bot#>
PREFIX ref: <https://brickschema.org/schema/Brick/ref#>
PREFIX inst: <https://example.com/inst#>

select ?spaceId ?pointName where {{ 
    ?space a bot:Space .
    ?space ex:revitID ?spaceId .
    ?radiator fso:transfersHeatTo ?space .
    ?radiator brick:hasPoint [a brick:Heating_Thermal_Power_Sensor;
        ref:hasExternalReference [ref:hasTimeseriesId ?pointName;
        ref:storedAt ?db]
    ] . }} """
    
    res = run_query(query)
    
    res = fix_res(res, "spaceId", "pointName")

    return res

def get_damper_opening_ids_in_zones(repository):
    # The query finds all spaces that have fluid supplied from a VAV damper
    query = """
PREFIX ex: <https://example.com/ex#>
PREFIX fso: <https://w3id.org/fso#>
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX bot: <https://w3id.org/bot#>
PREFIX ref: <https://brickschema.org/schema/Brick/ref#>
PREFIX inst: <https://example.com/inst#>

SELECT ?spaceId ?posId WHERE {
    ?space a bot:Space .
    ?space ex:revitID ?spaceId .
    ?space fso:hasFluidFedBy* ?damper .
    ?damper a fso:MotorizedDamper .
    ?damper brick:hasPoint [
        a brick:Position_Sensor;
        ref:hasExternalReference [ref:hasTimeseriesId ?posId]
    	]
}
"""
    
    res = run_query(repository, query)
    
    res = fix_res(res, "spaceId", "posId")

    return res

def get_vav_opening_ids_in_zones(repository):
    # The query finds all spaces that have fluid supplied from a VAV damper
    query = """
PREFIX bot: <https://w3id.org/bot#>
PREFIX ex: <https://example.com/ex#>
PREFIX fso: <https://w3id.org/fso#>
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX ref: <https://brickschema.org/schema/ref#>

SELECT ?space ?posId WHERE {
    ?damper a brick:VAV .
    ?damper brick:hasLocation ?space .
    ?space a bot:Space .
    ?damper brick:hasPoint [
        a brick:Damper_Position_Sensor ;
        ref:hasExternalReference [ref:hasTimeseriesId ?posId]
    	]
}
"""
    
    res = run_query(repository, query)
    
    res = fix_res(res, "space", "posId")

    return res

def get_zone_max_damper_opening():
    damper_names = get_damper_opening_ids_in_zones()

def get_columns_above_below(df, cutoff, above=True):
    if above:
        return df.loc[:, (df > cutoff).any(axis=0)]
    else:
        return df.loc[:, (df < cutoff).any(axis=0)]

def get_room_temperature_ids(repository):
    query = """
PREFIX bot: <https://w3id.org/bot#>
PREFIX ex: <https://example.com/ex#>
PREFIX fso: <https://w3id.org/fso#>
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX ref: <https://brickschema.org/schema/ref#>
SELECT ?space ?pointType ?pointId WHERE {
	?space a bot:Space .
    ?space brick:hasPoint ?point .
    ?point a ?pointType .
    VALUES ?pointType {brick:Zone_Air_Temperature_Sensor brick:Zone_Air_Temperature_Setpoint brick:Zone_Air_Cooling_Temperature_Setpoint brick:Zone_Air_Heating_Temperature_Setpoint}
    ?point ref:hasExternalReference [ref:hasTimeseriesId ?pointId]
}
"""
    res = run_query(repository, query)
    res = fix_res(res, "space", ["pointType", "pointId"])
    return res

def plot_room_points(space, point_types, start, stop, repository = "HTR-extra", sampling = "15min", base_url = "http://localhost:7200", token = None):
    
    res = get_room_points(space, point_types, repository, base_url, token)

    res = res.json()["results"]["bindings"]

    point_names = [point["pointId"]["value"] for point in res]
    if "spaceName" in res[0].keys():
        space_name = space + ": " + res[0]["spaceName"]["value"]
    else:
        space_name = space

    conn = tdb.get_timescale_connection()

    res_df = tdb.query_points(
        conn,
        point_names,
        start,
        stop,
        pivot=True,
        sampling=sampling,
        get_units=False
        )
    res_df.interpolate(method="time", inplace=True)
    conn.close()
    px.line(res_df,title=space_name).show()
    return

def get_room_points(space, point_types, repository = "HTR-extra", base_url = "http://localhost:7200", token = None):
    graphdb_query = f"""
PREFIX bot: <https://w3id.org/bot#>
PREFIX ex: <https://example.com/ex#>
PREFIX fso: <https://w3id.org/fso#>
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX ref: <https://brickschema.org/schema/ref#>
PREFIX inst: <https://example.com/inst#>

SELECT ?spaceName ?pointId ?pointType WHERE {{
    
    inst:{space} brick:hasPoint [
        a ?pointType ;
        ref:hasExternalReference [ref:hasTimeseriesId ?pointId]
        ]
    OPTIONAL {{inst:{space} rdfs:label ?spaceName}}  
    VALUES ?pointType {{ {" ".join(point_types)} }}
}}
    """
    # print(graphdb_query)
    res = run_query(repository, graphdb_query, base_url=base_url, token=token)
    return res

def get_point_ids_from_all_rooms(point_types, repository = "HTR-extra", base_url = "http://localhost:7200", token = None):

    query = f"""
    PREFIX bot: <https://w3id.org/bot#>
    PREFIX ex: <https://example.com/ex#>
    PREFIX fso: <https://w3id.org/fso#>
    PREFIX brick: <https://brickschema.org/schema/Brick#>
    PREFIX ref: <https://brickschema.org/schema/ref#>
    SELECT ?space ?pointType ?pointId WHERE {{
        ?space a bot:Space .
        ?space brick:hasPoint ?point .
        ?point a ?pointType .
        VALUES ?pointType {{{" ".join(point_types)}}}
        ?point ref:hasExternalReference [ref:hasTimeseriesId ?pointId]
    }}
    """
    # Run the query
    res = run_query(repository, query, base_url=base_url, token=token)

    # Format the results to json
    res = fix_res(res, "space", ["pointType", "pointId"])

    return res

def get_timeseries_data_from_multiple(point_ids, start, stop, sampling="15min", include_names=True, external_sampling=False, print_progress=True, tz="UTC"):
    conn = tdb.get_timescale_connection()

    df = pd.DataFrame()
    total_points = sum([len(points) for points in point_ids.values()])
    progress = 0
    for object, points in point_ids.items():
        object_name = object.split("#")[-1]
        point_names = [point["pointId"] for point in points]
        point_types = ["brick:"+point["pointType"].split("#")[-1] for point in points]
        
        name_type_map = dict(zip(point_names, point_types))
        
        if external_sampling:
            res_df = tdb.query_points_external_sampling(
                conn,
                point_names,
                start,
                stop,
                get_units=False,
                pivot=True,
                sampling=sampling
                )
        else:
            res_df = tdb.query_points(
                conn,
                point_names,
                start,
                stop,
                get_units=False,
                pivot=True,
                sampling=sampling
                )
        
        if res_df is None:
            continue
        
        # res_df.columns = [name_type_map[col] for col in res_df.columns]
        if include_names:
            res_df.columns = pd.MultiIndex.from_tuples(
                [(object_name,name_type_map[col], col) for col in res_df.columns]
                )
        else:
            res_df.columns = pd.MultiIndex.from_tuples(
                [(object_name,name_type_map[col]) for col in res_df.columns]
                )
        df = pd.concat([df, res_df], axis=1)
        # Print progress
        progress += len(point_names)
        if print_progress:
            print(f"Progress: {progress}/{total_points} ({(progress/total_points)*100:.2f}%)", end="\r")
    if print_progress:
        print(f"Progress: {progress}/{total_points} ({(progress/total_points)*100:.2f}%)\n")
    conn.close()

    if tz.lower() != 'utc':
        df.index = df.index.tz_convert("Europe/Copenhagen")

    return df

def plot_modes(zones: list[Zone], filename, plot_alt_modes=False, max_rooms_pr_row=20, period=None):
    data = {}
    for zone in zones:
        if plot_alt_modes:
            zone.find_alt_modes()
        modes = zone.summarize_modes(period)

        if plot_alt_modes:
            alt_modes = zone.summarize_alt_modes(period)
            data[zone.name] = {"modes": modes, "alt_modes": alt_modes}

        else:
            data[zone.name] = {"modes": modes}

    # Plot the modes
    
    rows = int((len(zones) / max_rooms_pr_row).__ceil__())
    
    # print(f"rooms {len(zones)}")

    # print(f"rows: {rows}")
    # print(f"rooms_pr_row: {(len(zones) / rows).__ceil__()}")

    rooms_pr_row = (len(zones) / rows).__ceil__()

    # Define the mode names and colors
    mode_names = ["inactive", "min", "CO2", "CO2-max", "temp", "temp-max"]
    colors = ["#A3A3A3", "#666666", "#67D0E0", "#2292A4", "#F4F750", "#BDBF09"]

    # Create a figure and axis
    if rows == 1:
        row_height = 8
    else:
        row_height = 6
        
    fig, axes = plt.subplots(nrows=rows, figsize=pf.get_figure_size(height=row_height*rows))
    if rows == 1:
        axes = [axes]
    fig: plt.Figure
    axes: list[plt.Axes]
    # Define the width of the bars
    if plot_alt_modes:
        bar_width = 0.35
        bar_offset = bar_width*1/2
    else:
        bar_width = 0.7
        bar_offset = 0
    for i in range(rows):
        # Define the positions of the bars
        row_names = [zone for zone in data.keys()][i*rooms_pr_row:(i+1)*rooms_pr_row]
        row_data = {name: data[name] for name in row_names}
        index = np.arange(len(row_data))

        # Initialize the bottom position for the stacked bars
        bottom = np.zeros(len(row_data))
        alt_bottom = np.zeros(len(row_data))

        # Plot each mode
        for mode, color in zip(mode_names, colors):
            mode_values = [row_data[zone]["modes"].get(mode, 0) for zone in row_data.keys()]
            axes[i].bar(index - bar_offset, mode_values, bar_width, label=mode, color=color, bottom=bottom)
            bottom += mode_values
            if plot_alt_modes:
                alt_mode_values = [row_data[zone]["alt_modes"].get(mode, 0) for zone in row_data.keys()]
                axes[i].bar(index + bar_offset, alt_mode_values, bar_width, color=color, bottom=alt_bottom)
                alt_bottom += alt_mode_values
            

        # Set the labels and title
        axes[i].set_ylabel('Mode Distribution [\%]')
        # axes[i].set_title('Modes for each zone')
        axes[i].set_xticks(index)
        axes[i].set_xticklabels(row_data.keys(), rotation=90)
        axes[i].grid(False)
        if i == 0:
            handles, labels = axes[i].get_legend_handles_labels()
            axes[i].legend(handles[::-1], labels[::-1], loc='upper center', ncol=len(mode_names), bbox_to_anchor=(0.5, 1.25))
        if i == rows-1:
            axes[i].set_xlabel('Zones')

        axes[i].set_ylim(0,100)

    # fig.suptitle(f"Modes for each zone with SAT = {' '.join([f'{x:.2f}' for x in opt_SAT_MO])}", size=16)
    # Show the plot
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    fig.savefig(f"figs/{filename}.pdf", dpi=300)
    plt.show()

def compare_modes(zones1: list[Zone], zones2: list[Zone], filename, max_rooms_pr_row=20, period1=None, period2=None, hatch='//', labels=["period1","period2"]):
    
    # Initially, check that all zones have the same names
    if len(zones1) != len(zones2):
        raise ValueError("The two lists of zones must have the same length.")
    
    names1 = [zone.name for zone in zones1]
    names2 = [zone.name for zone in zones2]
    
    if names1 != names2:
        raise ValueError("The two lists of zones must have the same names.")
    
    # Create a dictionary to hold the mode data for each zone
    data = {}
    
    for i in range(len(zones1)):
        zone1 = zones1[i]
        zone2 = zones2[i]
        modes1 = zone1.summarize_modes(period1)
        modes2 = zone2.summarize_modes(period2)
        data[zone1.name] = {"modes1": modes1, "modes2": modes2}

    # Plot the modes
    rows = int((len(zones1) / max_rooms_pr_row).__ceil__())
    
    # print(f"rooms {len(zones)}")

    # print(f"rows: {rows}")
    # print(f"rooms_pr_row: {(len(zones) / rows).__ceil__()}")

    rooms_pr_row = (len(zones1) / rows).__ceil__()

    # Define the mode names and colors
    mode_names = ["inactive", "min", "CO2", "CO2-max", "temp", "temp-max"]
    new_mode_names = ["Inactive", "Min", "CO2", "Max. CO2", "Cooling", "Max. cooling"]

    colors = ["#A3A3A3", "#666666", "#67D0E0", "#2292A4", "#F4F750", "#BDBF09"]

    # Create a figure and axis
    if rows == 1:
        row_height = 8
    else:
        row_height = 6
        
    fig, axes = plt.subplots(nrows=rows, figsize=pf.get_figure_size(height=row_height*rows))
    
    if rows == 1:
        axes = [axes]
    
    fig: plt.Figure
    axes: list[plt.Axes]
    
    # Define the width of the bars
    bar_width = 0.35
    bar_offset = bar_width*1/2
    patches = []
    alt_patches = []
    for i in range(rows):
        # Define the positions of the bars
        row_names = [zone for zone in data.keys()][i*rooms_pr_row:(i+1)*rooms_pr_row]
        row_data = {name: data[name] for name in row_names}
        index = np.arange(len(row_data))

        # Initialize the bottom position for the stacked bars
        bottom = np.zeros(len(row_data))
        alt_bottom = np.zeros(len(row_data))

        # Plot each mode
        for mode, color, new_mode in zip(mode_names, colors, new_mode_names):
            if i == 0:
                patches.append(mpatches.Patch(facecolor=color, label=new_mode))
                alt_patches.append(mpatches.Patch(facecolor=color, hatch=hatch, edgecolor='black', label=new_mode))

            mode_values = [row_data[zone]["modes1"].get(mode, 0) for zone in row_data.keys()]
            axes[i].bar(index, mode_values, -bar_width, label=new_mode, color=color, bottom=bottom, align="edge", edgecolor="grey")
            bottom += mode_values
        
            alt_mode_values = [row_data[zone]["modes2"].get(mode, 0) for zone in row_data.keys()]
            axes[i].bar(index, alt_mode_values, bar_width, color=color, bottom=alt_bottom, align="edge", edgecolor="grey", hatch=hatch)
            alt_bottom += alt_mode_values
            

        # Set the labels and title
        axes[i].set_ylabel('Mode Distribution [\%]')
        # axes[i].set_title('Modes for each zone')
        axes[i].set_xticks(index)
        axes[i].set_xticklabels(row_data.keys(), rotation=90)
        axes[i].grid(False)
        # if i == 0:
            # handles, labels = axes[i].get_legend_handles_labels()
            # first_legend = axes[i].legend(handles[::-1], labels[::-1], loc='upper center', ncol=len(mode_names), bbox_to_anchor=(0.5, 1.25))
            # axes[i].add_artist(first_legend)

        if i == rows-1:
            axes[i].set_xlabel('Zones')

        axes[i].set_ylim(0,100)
    
    
    # After plotting all bars (but before plt.show()):
    solid_patch = mpatches.Patch(facecolor='0.9', label=labels[0], edgecolor='grey')
    hatched_patch = mpatches.Patch(facecolor='0.9', hatch=hatch+hatch[0], edgecolor='grey', label=labels[1])

    # axes[0].legend(handles=[solid_patch, hatched_patch], loc='upper right', bbox_to_anchor=(1.15, 1))


    first_legend = axes[0].legend(handles=patches, loc='upper left', ncol=1, bbox_to_anchor=(1, 1))
    axes[0].add_artist(first_legend)
    # Get the lower edge of first_legend
    new_legend_top = first_legend.get_window_extent().transformed(axes[0].transAxes.inverted()).extents[1]
    axes[0].legend(handles=[solid_patch, hatched_patch], loc='upper left', ncol=1, bbox_to_anchor=(1, new_legend_top-0.1))
    
    

    # fig.suptitle(f"Modes for each zone with SAT = {' '.join([f'{x:.2f}' for x in opt_SAT_MO])}", size=16)
    # Show the plot
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    # fig.tight_layout()
    fig.savefig(f"figs/{filename}.pdf", dpi=300)
    plt.show()
    

def get_unit_from_id(point_ids, repository, base_url, token):
    if isinstance(point_ids, str):
        point_ids = [point_ids]

    formatted_point_ids = [f'"{point_id}"' for point_id in point_ids]
    
    query = f"""
PREFIX ref: <https://brickschema.org/schema/ref#>
PREFIX brick: <https://brickschema.org/schema/Brick#>
PREFIX sesame: <http://www.openrdf.org/schema/sesame#>
PREFIX inst: <https://example.com/inst#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
select ?pointId ?unit where {{ 
    ?point ref:hasExternalReference/ref:hasTimeseriesId ?pointId .
    VALUES ?pointId {{ {' '.join(formatted_point_ids)} }}
    ?point brick:hasUnit ?unit .
}}
    """

    res = run_query(repository=repository, query=query, base_url=base_url, token=token)
    
    units = {}

    for binding in res.json()["results"]["bindings"]:
        point_id = binding["pointId"]["value"]
        unit_uri = binding["unit"]["value"]
        units[point_id] = unit_uri

    for point_id in point_ids:
        if point_id not in units:
            units[point_id] = "Unknown"
            logging.warning(f"Unit not found for point {point_id}. Assigning 'Unknown' as unit.")
    return units

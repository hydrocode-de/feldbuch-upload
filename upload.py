from typing import List, Dict, Any
import os
from collections import defaultdict
from datetime import datetime as dt
import json

import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# create the GROUP lookups
valid_years = ['2022', '2023', '2024', '2025']
G1 = ['height', 'length', 'diameter', 'mortality', 'browse', 'cause', 'lost', 'height_notes']
G2 = ['solno', 'dsf', 'isf', 'tsf', 'openess', 'sol_notes']

@st.cache_resource
def client() -> Client:
    client = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
    return client


@st.cache_data
def get_lookup(tablename: str) -> dict:
    supabase = client()
    data = supabase.table(tablename).select('*').execute()
    return {d['short_name']: int(d['id']) for d in data.data}

@st.cache_data
def read_file(f) -> pd.DataFrame:
    if f.type == 'text/csv':
        df = pd.read_csv(f, sep=';')
        return df.copy()
    elif 'spreadsheet' in f.type:
        df = pd.read_excel(f)
        return df.copy()
    else:
        st.error(f"The file type {f.type} is not supported.")
        st.stop()


def upload_new_file() -> pd.DataFrame:
    f = st.file_uploader('Base data Excel file', accept_multiple_files=False)

    if f is not None:
        return read_file(f)
    else:
        st.stop()


def record_to_datasets(record: Dict[str, Any]) -> List[dict]:
    groups = defaultdict(lambda: defaultdict(lambda: {}))
    
    # get the user id from the session
    user_id = st.session_state.user_id

    # extract the information
    for col in record.keys():
        for gid, g in zip((1,2,), (G1, G2)):
            if any([col.lower().startswith(key)  for key in g]):
                chunks = col.split('_')
                if chunks[-1] not in valid_years:
                    continue
                else:
                    year = int(chunks[-1])
                    if len(chunks[:-1]) > 1 and chunks[1] == 'notes':
                        name = 'notes'
                    else:
                        name = '_'.join(chunks[:-1])

                # change the value                
                try:
                    value = int(record[col])
                except Exception:
                    value = record[col]
                    if isinstance(value, str):
                        value = value.replace('dms', 'ms dead')
                    elif record[col] is None or record[col] == 'NaN' or np.isnan(record[col]):
                        continue
                
                groups[year][gid][name] = value
    
    # make the datasets
    datasets = []
    for year in groups:
        measurement_time = dt(int(year), 3, 31, 12, 00).isoformat()
        for gid in groups[year]:
            datasets.append({
                'measurement_time': measurement_time,
                'data': json.dumps({k: v for k, v in groups[year][gid].items() if v}),
                'group_id': gid,
                'user_id': user_id
            })
    return datasets

def df_to_records(df: pd.DataFrame) -> List[dict]:
    out = []
    
    sites = get_lookup('sites')
    species = get_lookup('species')
    treatments = get_lookup('treatments')
    structures = get_lookup('structures')

    for rec in df.to_dict(orient='records'):
        out.append(dict(
            number=rec['number'],
            individual=rec['individual'],
            species_id=species[rec['species'][:2]] if rec['species'][:2] in species else None,
            structure_id=structures.get(rec['structure']),
            treatment_id=treatments.get(rec['treatment']),
            site_id=sites[rec['site']],
            pm_replaced=bool(rec['pm_replaced']),
            datasets=record_to_datasets(rec)
        ))
    
    return out


def main():
    st.set_page_config(layout='wide', page_title='Feldbuch upload')
    st.title("Feldbuch data upload")

    # get the upload data
    df = upload_new_file()

    # load emails
    supabase = client()
    emails = {r['user_id']: r['email'] for r in supabase.table('email_lookup').select('*').execute().data}
    st.selectbox('Select the uploading user', options=list(emails.keys()), format_func=lambda k: emails.get(k), key='user_id')

    with st.expander('Raw uploaded data'):
        st.dataframe(df)
#    with st.expander('DEVELOPMENT'):
#        st.json(df.to_dict(orient='records')[:10])
    with st.expander(f'JSON for database upload [Preview first 5'):
        with st.spinner('Transforming data...'):
            records = df_to_records(df)
            st.write(f"Parsed {len(records)} Plot records")
        st.json(records[:5])

    # check how many sites are available
    sites = []
    for r in records:
        if r['site_id'] not in sites:
            sites.append(r['site_id'])
    upload_site = st.selectbox('Select site for upload', options=sites)

    st.warning('You can either **relpace** the whole database with the uploaded table, or **add** to the exising instance. In any case **make sure the JSON above is correct!!!**')
    left, right = st.columns(2)
    do_replace = left.button('REPALCE FULL DATABASE')
    do_add = right.button('ADD TO DATABASE')
    
    if do_replace:
        with st.spinner('Deleting database data...'):
            supabase.table('datasets').delete().gte('id', 0).execute()
            supabase.table('plots').delete().gte('id', 0).execute()
    elif do_add:
        st.info('Keeping old data. Just adding...')
    else:
        st.stop()
    
    # do add
    upload_records = [r for r in records if r['site_id'] == upload_site]
    with st.spinner(f"Uploading {len(upload_records)} records of site_id={upload_site}..."):
        bar = st.progress(0.0)
        for i, rec in enumerate(upload_records):
            # add the plot
            added = supabase.table('plots').insert({k: v for k, v in rec.items() if k != 'datasets'}, returning='representation').execute()
            supabase.table('datasets').insert([{**data, 'plot_id': added.data[0]['id']} for data in rec['datasets']]).execute()
            
            bar.progress((i + 1) / len(upload_records), text=f"[{i + 1} / {len(upload_records)}]")
    
    st.success('Done')

if __name__ == '__main__':
    main()

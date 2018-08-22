"""
Parses the data from pvoutput.org into .json files in the form of python list.

Check input: all_dates, all_urls, dir_str
"""

from html.parser import HTMLParser
import urllib.request
import requests
from bs4 import BeautifulSoup
from pprint import pprint as pp
import pandas as pd
import sys
from datetime import *
import copy
import numpy as np
import json
import os


class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Encountered a start tag: " + str(tag) + ". With attributes: ", str(attrs))
        attrs_dict = dict(attrs)
        if tag == "table" and attrs_dict['id'] == 'tb':
            print("Found it!")

    def handle_endtag(self, tag):
        print("Encountered an end tag :", tag)

    def handle_data(self, data):
        print("Encountered some data  :", data)


def find_between(s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""


def convert_date(str):

    t1 = datetime.strptime(str, '%H:%M')
    # t1 = datetime.strptime(str, '%I:%M%p')
    t0 = datetime(1900, 1, 1)
    return (t1-t0).total_seconds() / 60.0


def convert_normalised(str):
    try:
        value = float(str.replace("kW/kW", ""))
        return value
    except ValueError:
        return 0.0


def time_extend(pd, tcolname, tval, start_stamp):

    full_time = np.arange(0, 24*60, tval, dtype=float)

    full_list = []
    a = 0
    for t in full_time:
        select = pd.loc[pd[tcolname] == t]

        if not select.empty:
            a = float(select['NormalisedFloat'])
            listtoadd = [start_stamp + t, a, a]
        else:
            listtoadd = [start_stamp + t, a, a]

        full_list.append(listtoadd)

    full_list_np = np.array(full_list)
    full_list_np_sorted = full_list_np[full_list_np[:, 0].argsort()]
    full_list_sorted = full_list_np_sorted.tolist()

    return full_list_sorted


def pvoutput_org(basic_url, all_dates):

    s = requests.session()
    login_url = "https://pvoutput.org"
    payload = {
        'login': 'igorsowa9',
        'password': 'Zurawski9',
        'remember': '1'
    }
    s.post(login_url, data=payload, headers=dict(referer=login_url))

    tval = 5

    start_stamp = 0
    full_list_sorted_all = np.empty([1, 3])
    sh = []
    for date in all_dates:
        print(date)
        url = basic_url + date

        page = s.get(url, headers=dict(referer=url))
        soup = BeautifulSoup(page.text, 'html.parser')

        only_table = soup.find_all("table")
        listof_frames = pd.read_html(str(only_table[0]))
        frame = listof_frames[0]

        date = frame.iloc[2, 0]
        date_str = str(date).replace("/", "_")
        name = str(frame.iloc[0, 0]).replace(" ", "_")

        # cut off not necessary ones
        mod = frame.drop([11], axis=1)
        first_row = list(mod.iloc[1])
        mod = mod.drop([0, 1])
        mod.columns = first_row
        mod = mod.reset_index(drop=True)

        # take only necessary data
        mod2 = mod[['Time', 'Normalised']]
        mod3 = copy.deepcopy(mod2)
        # substitite dates with minutes and cut off units
        mod3['TimeSeconds'] = mod2.Time.apply(convert_date)
        mod3['NormalisedFloat'] = mod2.Normalised.apply(convert_normalised)
        mod3['Forecast'] = mod2.Normalised.apply(convert_normalised)
        final = mod3.drop(['Time', 'Normalised'], axis=1)

        full_list_sorted = time_extend(final, 'TimeSeconds', tval, start_stamp)
        sh.append(len(full_list_sorted))

        full_list_sorted = np.array(full_list_sorted)

        full_list_sorted_all = np.vstack([full_list_sorted_all, full_list_sorted])
        start_stamp += 24 * 60

    file_name = name + "_" + all_dates[0] + "_" + all_dates[-1]
    dir_str = "/home/iso/PycharmProjects/vpp/data/pvoutput_org/min5/" + str(name)
    path = "/home/iso/PycharmProjects/vpp/data/pvoutput_org/min5/" + str(name) + \
           "/" + str(file_name) + '.json'
    if not os.path.exists(dir_str):
        os.makedirs(dir_str)

    full_list_sorted_all = full_list_sorted_all[1:, :].tolist()
    with open(path, 'w') as outfile:
        json.dump(full_list_sorted_all, outfile)

    print("Saving list to json (length(s): " + str(sh) + ", period: "+str(tval)+" min) to the file: \n" + path)


all_dates = ["20170901", "20170902", "20170903", "20170904", "20170905", "20170906", "20170907"]#,
# all_dates = ["20180625", "20180626", "20180627", "20180628", "20180629", "20180630", "20180701"]#,
             # "20180625", "20180626", "20180627", "20180628", "20180629", "20180630", "20180701",
             # "20180702", "20180703", "20180704", "20180705", "20180706", "20180707", "20180708",
             # "20180709", "20180710", "20180711", "20180712", "20180713", "20180714", "20180715",
             # "20180716", "20180717"]

all_urls = [
    "https://pvoutput.org/intraday.jsp?id=33196&sid=30411&dt="  # GfB mbH - Westnetz GmbH 29.610kW
    # "https://pvoutput.org/intraday.jsp?id=6197&sid=4947&dt="    # HLB Sunnyfarm 19.500kW
    # "https://pvoutput.org/intraday.jsp?id=56577&sid=51073&dt="  # race|result 92.750kW
    # "https://pvoutput.org/intraday.jsp?id=59027&sid=52525&dt="  # Michiels Wegberg 27.900kW
    # "https://pvoutput.org/intraday.jsp?id=49073&sid=46276&dt="  # Freakycat 20.100kW
    # "https://pvoutput.org/intraday.jsp?id=26995&sid=24693&dt="  # WohnhausA1 20.240kW
    # "https://pvoutput.org/intraday.jsp?id=42740&sid=39044&dt="  # PV-Anlage dahoam 23.000kW
    # "https://pvoutput.org/intraday.jsp?id=66425&sid=59062&dt="   # SGjuk_12KW 29.000kW
    ]

count = 0
for url in all_urls:
    count += 1

    pvoutput_org(url, all_dates)

    # for date in all_dates:
    #     print("Parsing "+str(count)+"/"+str(len(all_urls))+" from date: " + date)
    #     pvoutput_org(url + date)

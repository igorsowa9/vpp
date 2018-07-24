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

# parser = MyHTMLParser()
# parser.feed(mystr)

def find_between(s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""


def convert_date(str):
    t1 = datetime.strptime(str, '%I:%M%p')
    t0 = datetime(1900, 1, 1)
    return (t1-t0).total_seconds() / 60.0


def convert_normalised(str):
    try:
        value = float(str.replace("kW/kW", ""))
        return value
    except ValueError:
        return 0.0


def time_extend(pd, tcolname, tval):

    full_time = np.arange(0, 24*60, tval, dtype=float)

    full_list = []

    for t in full_time:
        select = pd.loc[pd[tcolname] == t]

        if select.empty:
            listtoadd = [t, 0, 0]
        else:
            listtoadd = [t, 0, float(select['NormalisedFloat'])]

        # print(listtoadd)
        full_list.append(listtoadd)

    full_list_np = np.array(full_list)
    full_list_np_sorted = full_list_np[full_list_np[:, 0].argsort()]
    full_list_sorted = full_list_np_sorted.tolist()

    return full_list_sorted


def pvoutput_org(url):

    login = ""
    password = ""
    sess = requests.Session()
    sauce = sess.get(url)
    soup = BeautifulSoup(sauce.text, "html.parser")

    # page = requests.get(url)
    # soup = BeautifulSoup(page.text, 'html.parser')
    print(soup.prettify())
    only_table = soup.find("table", {"id": "tb"})
    print(only_table)

    sys.exit()

    opener = urllib.request.urlopen(url)
    mybytes = opener.read()
    mystr = mybytes.decode("utf8")
    opener.close()
    only_table = find_between(mystr, "<div id='content'>", "</div>")
    print(only_table)
    listof_frames = pd.read_html(only_table)
    frame = listof_frames[0]

    date = frame.iloc[2,0]
    date_str = str(date).replace("/", "_")
    name = str(frame.iloc[0, 0]).replace(" ", "_")
    file_nane = name + "_" + date_str

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
    mod3['Forecast'] = 0
    final = mod3.drop(['Time', 'Normalised'], axis=1)

    tval = 5
    full_list_sorted = time_extend(final, 'TimeSeconds', tval)
    sh = len(full_list_sorted)

    path = "/home/iso/PycharmProjects/vpp/data/original_elia/pvoutput_org/min5/" + str(file_nane) + '.json'
    with open(path, 'w') as outfile:
        json.dump(full_list_sorted, outfile)

    print("Saving list to json (length: " + str(sh) + ", period: "+str(tval)+" min) to the file: \n" + path)


url = "https://pvoutput.org/intraday.jsp?id=33196&sid=30411&dt=20180629"
pvoutput_org(url)
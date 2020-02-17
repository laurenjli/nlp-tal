'''
Katy Koenig

Feb. 2020

Functions to Scrape Descriptions/Prologues of "This American Life" Episodes

'''
import os
import csv
import re
import json
import multiprocessing as mp
import glob
import urllib.parse
import requests
import bs4
import pandas as pd


ARCHIVE_URL = 'https://www.thisamericanlife.org/archive'
test_yr = 'https://www.thisamericanlife.org/archive?year=2018'


def make_soup(url):
    '''
    Send request for a url and if successful request, makes a soup object.

    Inputs:
        url: webpage string
        
    Outputs: a soup data structure for a webpage
    '''
    response = requests.get(url)
    if response.status_code == 200:
        return bs4.BeautifulSoup(response.text, 'html.parser')
    else:
        return None


def convert_and_check(new_url, base_url=ARCHIVE_URL[:-8],
                      limiting_domain=ARCHIVE_URL[12:-8]):
    '''
    Attempts to determine whether new_url is a relative URL and if so,
    use base_url to determine the path and create a new absolute
    URL
    Inputs:
        base_url: absolute URL
        new_url: updated URL that is a full link
    Outputs:
        new absolute URL or None
    '''
    if new_url[-4:] not in [".edu", ".org", ".com", ".net"]:
        new_url = urllib.parse.urljoin(base_url, new_url)
    if limiting_domain in new_url:
        return new_url
    return False


def get_one_ep_info(url, output):
    '''
    Gets information regarding one TAL episode and appends this info to
    an existing csv
    
    Inputs:
        url(str): url for one TAL episode
        output(str): name of csv to be appended to

    Outputs: None except changed csv with episode's info
    '''
    soup = make_soup(url)
    title = soup.find('h1').text
    ep_num = soup.find('div', class_='field-item even').text
    pub_date = soup.find('meta',
                         property='article:published_time')['content']
    date = re.search('[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]',
                     pub_date).group(0)
    summary = soup.find('div', class_='field field-name-body ' + \
                        'field-type-text-with-summary field-label-hidden').text
    if soup.find_all('div', class_='field field-name-field-acts ' + \
                              'field-type-entityreference field-label-hidden'):
        acts_soup = soup.find_all('div', class_='field field-name-field-acts ' + \
                                  'field-type-entityreference field-label-hidden') \
                        [0].find_all('div', class_='content')
        with open(output, 'a') as csvfile:
            outputwriter = csv.writer(csvfile, delimiter=',')
            for act in acts_soup:
                if act.find('a'):
                    act_link = convert_and_check(act.find('a')['href'])
                    act_soup = make_soup(act_link)
                    act_descript = act_soup.find('p').text
                    row_lst = [ep_num, title, date, url, summary, act_descript]
                    outputwriter.writerow(row_lst)
                else:
                    print(f'No href in acts for {title}')
        csvfile.close()
    else:
        print(f"Coundn't find acts for {title}")


def write_yr_csv(yr_base_url):
    '''
    Writes csv for an individual year of TAL episodes
    (Initializes csv and then it is written in get_one_ep_info fn)

    Inputs:
        yr_base_url(str): string for the url for a year of TAL eps

    Outputs: None but prints when completes the csv for the year
    '''
    # initialize csv for each year
    output = 'tal_ep_descript_' + yr_base_url[-4:] + '.csv'
    cols = ['ep_num', 'title', 'date', 'url', 'ep_summary', 'act_descript']
    with open(output, 'w') as csvfile:
        outputwriter = csv.writer(csvfile, delimiter=',')
        outputwriter.writerow(cols)
    csvfile.close()
    yr_soup = make_soup(yr_base_url)
    ep_links = yr_soup.find_all('a', href=True, class_="thumbnail " + \
                                "goto goto-episode")
    for ep in ep_links:
        full_ep_link = convert_and_check(ep['href'])
        get_one_ep_info(full_ep_link, output)
    print(f'Finished for {yr_base_url}')


def do_all():
    '''
    This does the same as the main function at the bottom without
    the parallelization

    Specificially, it creates a csv for each year with the episode
    and act descriptions for episodes from 1995 through 2019
    '''
    year_urls = [ARCHIVE_URL + '?year=' + str(x) for x in range(1995, 2020)]
    for year in year_urls:
        write_yr_csv(year)


def merge_yr_csv(folder=os.getcwd(), key='tal'):
    '''
    Combines all the year TAL description csvs into one big pandas df

    Inputs:
        folder(str): folder in which csv files are located
        key(str): keyword in all the csvs

    Outputs: a pandas df with info for all episodes
    '''
    os.chdir(folder)
    to_combine = [pd.read_csv(x) for x in glob.glob(key + "*.csv")]
    return pd.concat(to_combine)


if __name__ == '__main__':
    pool = mp.Pool(processes=os.cpu_count()-1)
    year_urls = [ARCHIVE_URL + '?year=' + str(x) for x in range(1995, 2020)]
    pool.map_async(write_yr_csv, year_urls)
    pool.close()
    pool.join()

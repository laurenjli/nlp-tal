'''
Katy Koenig

Feb. 2020

Functions to Scrape Descriptions/Prologues of "This American Life" Episodes

'''
import os
from sys import argv
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



#### General Helper Functions ####

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


def write_yr_csv(yr_base_url, csv_type):
    '''
    Writes csv for an individual year of TAL episodes

    Inputs:
        yr_base_url(str): string for the url for a year of TAL eps
        csv_type(str): whether want just episode descriptions or transcripts

    Outputs: None but prints when completes the csv for the year
    '''
    # initialize csv for each year
    
    if csv_type == 'descript':
        output = 'tal_ep_descript_' + yr_base_url[-4:] + '.csv'
        cols = ['ep_num', 'title', 'date', 'url', 'ep_summary', 'act_descript']
    if csv_type == 'transcript':
        output = 'trans_tal_ep_' + yr_base_url[-4:] + '.csv'
        cols = ['ep_num', 'ep_title', 'date_published', 'act_name',
                'text', 'url']
    initial_csv(output, cols)
    yr_soup = make_soup(yr_base_url)
    ep_links = yr_soup.find_all('a', href=True, class_="thumbnail " + \
                                "goto goto-episode")
    for ep in ep_links:
        full_ep_link = convert_and_check(ep['href'])
        if csv_type == 'descript':
            get_one_ep_descript(full_ep_link, output)
        if csv_type == 'transcript':
            trans_link = get_transcript_link(full_ep_link)
            get_one_ep_transcript(trans_link, output)
    print(f'Finished for {yr_base_url}')


def initial_csv(output, cols):
    '''
    Initializes a csv using the name and columns given

    Inputs:
        output(str): csv name
        cols: list of column names

    Outputs: None (a csv is created)
    '''
    with open(output, 'w') as csvfile:
        outputwriter = csv.writer(csvfile, delimiter=',')
        outputwriter.writerow(cols)
    csvfile.close()


def combine_csvs(folder=os.getcwd(), key='tal'):
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


def do_all(scrape_type='descript'):
    '''
    This does the same as the main function at the bottom without
    the parallelization

    Specificially, it creates a csv for each year with the episode
    and act descriptions or transcripts for episodes from 1995 through 2019

    Inputs:
        scrape_type(str): must be either "descript" or "transcript"
    '''
    year_urls = [ARCHIVE_URL + '?year=' + str(x) for x in range(1995, 2020)]
    for year in year_urls:
        write_yr_csv(year, scrape_type)


### Specific Functions for Description Scraping ####

def get_one_ep_descript(url, output):
    '''
    Gets information regarding one TAL episode and appends this info to
    an existing csv
    
    Inputs:
        url(str): url for one TAL episode
        output(str): name of csv to which this episode's info will be appended

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


### Specific Functions for Rescraping Transcripts ####

def get_one_ep_transcript(url, output):
    '''
    Scrapes url of transcript of a "This American Life" episode;
    extracts relevant data, cleans transcript of episode
    
    Inputs:
        url: page for an episode's transcript
        output_csv(str): csv to which this episode's info will be appended
    
    Outputs: updated output_csv
    '''
    soup = make_soup(url)
    ep_num = url[33:-11]
    ep_name = soup.find('h1').text
    pub_date = soup.find('meta', property='article:published_time')['content']
    date = re.search('[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]',
                     pub_date).group(0)
    acts = soup.find_all('div', class_='act')
    # Get episode host name
    with open(output, 'a') as csvfile:
        outputwriter = csv.writer(csvfile, delimiter=',')
        for act in acts:
            # Get act name
            act_name = act.find('h3').text
            act_text = ''
            for paragraph in act.find_all('div', class_=True):
                if len(paragraph['class']) > 0: 
                    if paragraph['class'][0] in ['host', 'subject', 'interviewer']:
                        for chunk in paragraph.find_all('p', begin=True):
                            if chunk.text:
                                act_text += chunk.text
            row_lst = [ep_num, ep_name, pub_date, act_name,
                       act_text, url]
            outputwriter.writerow(row_lst)
    csvfile.close()


def get_transcript_link(url):
    '''
    Takes an episode's main webpage and outputs the corresponding transcript
    url for that episode

    Inputs:
        url(str): episode's main webpage

    Outputs:
        trans_link(str): episode's transcript page
    '''
    ep_soup = make_soup(url)
    transcript_links = ep_soup.find_all('a', href=True)
    for link in transcript_links:
        if 'transcript' in link['href']:
            trans_link = convert_and_check(link['href'])
    return trans_link


### ####


if __name__ == '__main__':
    pool = mp.Pool(processes=os.cpu_count()-1)
    year_urls = [(ARCHIVE_URL + '?year=' + str(x), argv[1])for \
                 x in range(1995, 2020)]
    pool.starmap_async(write_yr_csv, year_urls)
    pool.close()
    pool.join()

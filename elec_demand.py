"""generating electrical demand profile from elexon profiles
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 18})


def import_elexon_profile():

    # import the unrestricted residential
    # elexon electricity profile from excel sheet
    df = pd.read_excel('electricity_profiles.xlsx')
    df = df.drop(columns=['Profile Class 1'])
    columns = df.iloc[1].values
    data = df.iloc[2:50].values
    p = pd.DataFrame(data, columns=columns)
    p_av = ((p + p.shift(-1)) / 2)[::2]
    p_av = p_av.reset_index(drop=True)
    return p_av


def time_of_year(hour):

    # TIMINGS PROBABLY NOT ACCURATE - ESTIMATIONS MADE

    # function returning time of year for the hour chosen
    # end of winter is last day of march. simplify to end of march
    # 31 march... 90 days between zero hour and end of march 31st
    # 90 * 24 = 2160
    if hour >= 0 and hour < 2160:
        toy = 'Wtr'
    # spring is starting on 31st march... 15th June approx
    # 24 * 77 + 2160 = 4008
    elif hour >= 2160 and hour < 4008:
        toy = 'Spr'
    # summer is between 25th august and ten weeks after
    # 10 * 7 * 24 + 4008 = 5688
    elif hour >= 4008 and hour < 5688:
        toy = 'Smr'
    # high summer is 6 weeks and 2 days
    # (6 * 7 + 2) * 24 + 5688 = 6744
    elif hour >= 5688 and hour < 6744:
        toy = 'Hsr'
    # autumn is the period up to clock change
    # 25 august to 27 october
    # 64 days
    # 64 * 24 + 6744 = 8280
    elif hour >= 6744 and hour < 8280:
        toy = 'Aut'
    # then into winter
    else:
        toy = 'Wtr'

    return toy


def day_of_week(year, hour):

    # first day of year
    # a = datetime.datetime(year, 1, 1)
    # first_day = a.strftime('%A')
    year = str(year)
    data = pd.date_range('1/1/' + year, periods=8760, freq='H')
    day = data[hour].strftime('%A')
    if day == 'Saturday':
        day = 'Sat'
    elif day == 'Sunday':
        day = 'Sun'
    else:
        day = 'Wd'
    return day


def year_time_series(year):

    df = import_elexon_profile()
    demand = []
    for hour in range(8760):
        toy = time_of_year(hour)
        dow = day_of_week(year, hour)
        column_name = toy + ' ' + dow
        hour_day = hour % 24
        demand.append(df[column_name][hour_day])
    # plt.plot(demand)
    # plt.show()
    return demand


def demand_input():

    df = pd.read_excel('demand_inputs.xlsx', sheet_name='Demand gen')

    hot_water = df['Unnamed: 3'][15]

    dem = df.drop(index=[0, 1, 2, 3],
                    columns=['Unnamed: 0', 'Unnamed: 1',
                            'Unnamed: 2', 'Unnamed: 7'])
    list1 = []
    for x in range(14, 36):
        list1.append(x)
    dem = dem.drop(list1)
    dem = dem.rename(columns={'Unnamed: 3': 'Type',
                                'Unnamed: 4': 'Age',
                                'Unnamed: 5': 'Bedrooms',
                                'Unnamed: 6': 'Number of type'})
    dem = dem.reset_index(drop=True)

    return {'hot_water': hot_water, 'dem': dem}


def profile_from_input(year):

    heating = demand_input()['dem']

    standard_profile = year_time_series(year)
    standard_profile = np.array(standard_profile)

    profiles = []
    aggregate = 0
    number_profiles = len(heating)
    for p in range(number_profiles):
        number_of_type = heating['Number of type'][p]
        profiles.append(standard_profile * number_of_type)
        aggregate += standard_profile * number_of_type

    return aggregate


def predicted_demand(year):

    def movingaverage(values, window):
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
        return sma

    y = profile_from_input(year)

    window = 7
    yMA = movingaverage(y, window)
    yMA_max = np.amax(yMA)
    y_max = np.amax(y)
    ratio = y_max / yMA_max
    inserting = np.array(yMA[: window - 1])
    inserting = ratio * inserting
    yMA = np.insert(yMA, 1, inserting)

    np.savetxt('predicted_elec_demand.csv', yMA, delimiter=",", fmt='%.3e')


def plot_demand():

    df = pd.read_csv('predicted_elec_demand.csv', header=None, names=['dem'])

    plt.plot(df['dem'], 'b', LineWidth=1)
    plt.ylabel('Energy (kWh)')
    plt.xlabel('Hour')
    plt.show()

if __name__ == '__main__':
    predicted_demand(2019)
    plot_demand()

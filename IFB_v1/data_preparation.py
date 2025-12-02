import folktables

data_source = folktables.ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")

def load_acs_data():
    acs_data = data_source.get_data(states=["CA", "TX", "NY"], download=True)

load_acs_data()
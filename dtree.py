import pandas
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# enumerition is needed as ML works with numbers
degree_enum = {
    'IIT': 1,
    'BE': 2,
    'MCA': 3,
    'B.Sc': 4,
    'BCA': 5
}

acadamic_major_enum = {
    'CS': 1,
    'Mech': 2,
    'EnC': 3,
    'Aeronatical': 4,
    'IT': 5,
    'EE': 6,
    'Computer Applications': 7
}

internship_experience_enum = {
    'good': 1,
    'average': 2,
    'best': 3,
    'poor': 4
}

soft_skills_enum = {
    'good': 1,
    'average': 2,
    'best': 3,
    'poor': 4,
}

hard_skills_enum = {
    'good': 1,
    'average': 2,
    'best': 3,
    'poor': 4
}

potentials_enum = {
    'Promising': 1,
    'okay': 2,
    'not_good': 3,
}
offered_job_enum = {'yes': 1, 'no': 0}

# read training dataset

data_frame = pandas.read_csv('decisiontree.csv', header=0)
# print (data_frame.head())

data_frame['Degree'] = data_frame['Degree'].map(degree_enum)
data_frame['Academic major'] = 	data_frame['Academic major'].map(acadamic_major_enum)
data_frame['Internship experience']	= data_frame['Internship experience'].map(internship_experience_enum)
data_frame['Soft skills'] = data_frame['Soft skills'].map(soft_skills_enum)
data_frame['Hard skills'] = data_frame['Hard skills'].map(hard_skills_enum)
data_frame['Potentials'] = data_frame['Potentials'].map(potentials_enum)
data_frame['offered_job?'] = data_frame['offered_job?'].map(offered_job_enum)
# print (data_frame.head())

features = list(data_frame.columns[:7])
#print (features)
X = data_frame[features]
Y = data_frame['offered_job?']
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(X,Y)


forest_classifier = RandomForestClassifier(n_estimators=50)
forest_classifier = forest_classifier.fit(X, Y)

read_data = pandas.read_csv('dataset.csv', header=0)
read_data_org = pandas.read_csv('dataset.csv', header=0)

read_data['Degree'] = read_data['Degree'].map(degree_enum)
read_data['Academic major'] = 	read_data['Academic major'].map(acadamic_major_enum)
read_data['Internship experience']	= read_data['Internship experience'].map(internship_experience_enum)
read_data['Soft skills'] = read_data['Soft skills'].map(soft_skills_enum)
read_data['Hard skills'] = read_data['Hard skills'].map(hard_skills_enum)
read_data['Potentials'] = read_data['Potentials'].map(potentials_enum)

offered_job = {1: 'yes', 0: 'no'}

read_data_org['offered_job?'] = forest_classifier.predict(read_data)
read_data_org['offered_job?'] = read_data_org['offered_job?'].map(offered_job)

# print (read_data_org)
read_data_org.to_csv("results.csv")
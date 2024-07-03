from model import load_transformer
import pandas as pd
import joblib

df = pd.DataFrame({
    'date-mutation': ['2020-01-01'],
    'no-voie': [1],
    'b/t/q': ['B'],
    'type-de-voie': ['RTE'],
    'voie': ['DE SAMOENS'],
    'code-postal': [74440],
    'type-local': ['Maison'],
    'surface-terrain': [20],
    'surface-reelle-bati': [100],
    'nombre-pieces-principales': [5]
})

transformer = load_transformer('transformers/auvergne-rhône-alpes.csv')
model = joblib.load('models/auvergne-rhône-alpes.joblib')

X = transformer.transform(df)
y = model.predict(X)
print(y)

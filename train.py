from model import Transformer
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import pandas as pd
import joblib
import tqdm

region_names = {
    0: 'Auvergne-Rhône-Alpes',
    6: 'Bourgogne-Franche-Comté',
    12: 'Bretagne',
    5: 'Centre-Val de Loire',
    10: 'Corse',
    8: 'Grand Est',
    9: 'Hauts-de-France',
    3: 'Île-de-France',
    1: 'Normandie',
    2: 'Nouvelle-Aquitaine',
    4: 'Occitanie',
    11: 'Pays de la Loire',
    7: 'Provence-Alpes-Côte d\'Azur',
    13: 'Guadeloupe',
    14: 'Martinique',
    15: 'Guyane',
    16: 'La Réunion',
    17: 'Mayotte'
}

regions = {
    0: ['1', '15', '26', '3', '38',
        '42', '43', '63', '69', '7',
        '73', '74'],
    6: ['2', '59', '60', '62', '80'],
    12: ['13', '4', '5', '6', '83', '84'],
    5: ['10', '51', '52', '54', '55',
        '57', '67', '68', '8', '88'],
    10: ['11', '12', '30', '31', '32',
         '34', '46', '48', '65', '66',
         '81', '82', '9'],
    8: ['14', '27', '50', '61', '76'],
    9: ['16', '17', '19', '23', '24',
        '33', '40', '47', '64', '79',
        '86', '87'],
    3: ['18', '28', '36', '37', '41', '45'],
    1: ['21', '25', '39', '58', '70',
        '71', '89', '90'],
    2: ['22', '29', '35', '56'],
    4: ['2A', '2B'],
    11: ['44', '49', '53', '72', '85'],
    7: ['75', '77', '78', '91', '92',
        '93', '94', '95'],
    13: ['971'],
    14: ['972'],
    15: ['973'],
    16: ['974'],
    17: ['976']
}


def train(
    name: str,
    data: pd.DataFrame,
    transformers_path: Path,
    models_path: Path,
    score_path: Path
):
    transformer = Transformer()
    transformer.fit(data)
    transformer_path = f'{name}.csv'
    transformer.save(transformers_path / transformer_path)

    model = RandomForestRegressor(
        n_estimators=100,
        n_jobs=8,
        verbose=1
    )
    X, y = transformer.X, transformer.y
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    mse = mean_squared_error(y_test, model.predict(X_test))
    score = pd.DataFrame({
        'name': [name],
        'r2': [r2],
        'mse': [mse]
    })

    if score_path.exists():
        scores = pd.read_csv(score_path)
        scores = pd.concat([scores, score])
    else:
        scores = score

    scores.to_csv(score_path, index=False)

    model_path = f'{name}.joblib'
    joblib.dump(model, models_path / model_path)


def main():
    transformers_path = Path('transformers')
    models_path = Path('models')
    score_path = Path('scores.csv')

    if not transformers_path.exists():
        transformers_path.mkdir(parents=True)

    if not models_path.exists():
        models_path.mkdir(parents=True)

    path = 'valeursfoncieres-2023-v2.csv'
    full = pd.read_csv(path, low_memory=False)
    column = 'code-departement'
    full[column] = full[column].astype('str')

    for region in tqdm.tqdm(regions):
        name = region_names[region].lower().replace(' ', '-')
        if region == 7:
            for code in regions[region]:
                data = full[full[column] == code]
                _name = name+'-'+str(code)
                train(_name, data, transformers_path, models_path, score_path)
        else:
            data = full[full[column].isin(regions[region])]
            train(name, data, transformers_path, models_path, score_path)


if __name__ == '__main__':
    main()

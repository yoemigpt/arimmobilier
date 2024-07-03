from sklearn.ensemble import RandomForestRegressor
import pandas as pd


class Transformer:
    data: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series

    def fit(self, data):
        self.data = pd.DataFrame()
        column = 'date-mutation'

        df = pd.to_datetime(data[column], format='%Y-%m-%d')
        self.data['year'] = df.dt.year
        self.data['month'] = df.dt.month
        self.data['day'] = df.dt.day

        column = 'no-voie'
        self.data[column] = data[column].astype('Int64')

        column = 'b/t/q'
        self.data[column] = data[column].astype('category')

        column = 'type-de-voie'
        self.data[column] = data[column].astype('category')

        column = 'voie'
        self.data[column] = data[column].astype('category')
        # TODO: in the next step we will
        # convert the address to a coordinate
        # two columns: latitude and longitude
        column = 'code-postal'
        self.data[column] = data[column].astype('Int64')

        column = 'type-local'
        self.data[column] = data[column].astype('category')

        column = 'surface-terrain'
        self.data[column] = data[column]

        column = 'surface-reelle-bati'
        self.data[column] = data[column]

        column = 'nombre-pieces-principales'
        self.data[column] = data[column].astype('Int64')

        # Let it here.
        self.X = self.data.copy()
        for column in self.X.columns:
            if self.X[column].dtype.name == 'category':
                self.X[column] = self.X[column].cat.codes
        self.y = data['valeur-fonciere']

        return self

    def transform(self, data):
        X = pd.DataFrame()

        column = 'date-mutation'
        df = pd.to_datetime(data[column], format='%Y-%m-%d')
        X['year'] = df.dt.year
        X['month'] = df.dt.month
        X['day'] = df.dt.day

        column = 'no-voie'
        X[column] = data[column].astype('Int64')

        column = 'b/t/q'
        X[column] = data[column].apply(
            lambda x: self.data[column].cat.categories.get_loc(x)
        )

        column = 'type-de-voie'
        X[column] = data[column].apply(
            lambda x: self.data[column].cat.categories.get_loc(x)
        )

        column = 'voie'
        X[column] = data[column].apply(
            lambda x: self.data[column].cat.categories.get_loc(x)
        )

        column = 'code-postal'
        X[column] = data[column].astype('Int64')

        column = 'type-local'
        X[column] = data[column].apply(
            lambda x: self.data[column].cat.categories.get_loc(x)
        )

        column = 'surface-terrain'
        X[column] = data[column]

        column = 'surface-reelle-bati'
        X[column] = data[column]

        column = 'nombre-pieces-principales'
        X[column] = data[column].astype('Int64')

        return X

    def save(self, path):
        self.data.to_csv(path, index=False)


def load_transformer(path: str) -> Transformer:
    transformer = Transformer()
    data = pd.read_csv(path)
    transformer.data = data

    columns = [
        'b/t/q',
        'type-de-voie',
        'voie',
        'type-local'
    ]
    for column in columns:
        transformer.data[column] = transformer.data[column].astype('category')

    columns = [
        'no-voie',
        'code-postal',
        'nombre-pieces-principales'
    ]
    for column in columns:
        transformer.data[column] = transformer.data[column].astype('Int64')

    return transformer


if __name__ == '__main__':
    path = 'valeursfoncieres-2023-v2.csv'
    data = pd.read_csv(path)
    transformer = Transformer()
    transformer.fit(data)

    print(transformer.data.info())
    model = RandomForestRegressor(n_estimators=100)
    model.fit(transformer.X[:1000], transformer.y[:1000])

    df = pd.DataFrame({
        'date-mutation': ['2023-01-01', '2023-01-02'],
        'no-voie': [1, 2],
        'b/t/q': ['B', 'T'],
        'type-de-voie': ['RUE', 'VOIE'],
        'voie': ['DE LA LIBERATION', 'DE LA REPUBLIQUE'],
        'code-postal': [75000, 75001],
        'type-local': ['Maison', 'Appartement'],
        'surface-terrain': [100, 200],
        'surface-reelle-bati': [100, 200],
        'nombre-pieces-principales': [1, 2]
    })

    X = transformer.transform(df)
    print(model.predict(X))

    transformer.save('transformer.csv')
    transformer = load_transformer('transformer.csv')
    print(transformer.data.info())

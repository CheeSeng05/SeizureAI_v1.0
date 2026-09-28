from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

# This is a dataset you must create through video annotation.
data = pd.read_csv("labelled_movements.csv") # change the name of the csv file

# Prevent a patient appearing in multiple partitions.
assert data.groupby("patient_id")["split"].nunique().max() == 1

region = "eye_36_41"
target = "closed"
inputs = ["aperture", "velocity"]

usable = data[
    (data["region"] == region) & (data["valid"] == 1)
].dropna(subset=inputs + [target])

train = usable[usable["split"] == "train"]
calibration = usable[usable["split"] == "calibration"]
test = usable[usable["split"] == "test"]

# Training and calibration each need both positive and negative examples.
for partition in (train, calibration):
    assert set(partition[target].unique()) == {0, 1}

base = make_pipeline(StandardScaler(), LogisticRegression())
base.fit(train[inputs], train[target])

model = CalibratedClassifierCV(
    FrozenEstimator(base),
    method="sigmoid",
)
model.fit(calibration[inputs], calibration[target])

positive_column = list(model.classes_).index(1)
probabilities = model.predict_proba(test[inputs])[:, positive_column]

print("Estimated P(closed):", probabilities[:5])
print("Brier score:", brier_score_loss(test[target], probabilities))
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px

# -----------------------------
# Featurization Function
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return [
            Descriptors.MolWt(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol)
        ]
    return None

# -----------------------------
# Load & Prepare Dataset
@st.cache_data
def load_data():
    from pytdc.dataset import Tox21
    df = Tox21().get_data()
    df = df[['Drug', 'Y']].rename(columns={'Drug': 'smiles', 'Y': 'target'})
    df.dropna(inplace=True)
    return df

st.set_page_config(layout="wide")
st.title("🧪 Drug Toxicity Prediction Dashboard")

# Load dataset
data = load_data()
X, y, smiles_list = [], [], []

for _, row in data.iterrows():
    feats = featurize(row['smiles'])
    if feats:
        X.append(feats)
        y.append(row['target'])
        smiles_list.append(row['smiles'])

# Train Model
X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
    X, y, smiles_list, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Dashboard Table
X_test_df = pd.DataFrame(X_test, columns=["MolWt", "HDonors", "HAcceptors", "LogP", "TPSA"])
X_test_df["True Label"] = y_test
X_test_df["Predicted"] = y_pred
X_test_df["SMILES"] = smiles_test
X_test_df["Correct"] = X_test_df["True Label"] == X_test_df["Predicted"]

# -----------------------------
# Sidebar: SMILES Input
st.sidebar.header("🔬 Try Your SMILES")
user_smiles = st.sidebar.text_input("Enter a SMILES string:")

if user_smiles:
    mol = Chem.MolFromSmiles(user_smiles)
    feats = featurize(user_smiles)
    if feats:
        pred = model.predict([feats])[0]
        col1, col2 = st.columns(2)
        with col1:
            st.image(Draw.MolToImage(mol), caption="Molecular Structure")
        with col2:
            st.subheader("Prediction Result")
            st.write("### 🧬 Molecular Features")
            st.json({
                "MolWt": round(feats[0], 2),
                "HDonors": feats[1],
                "HAcceptors": feats[2],
                "LogP": round(feats[3], 2),
                "TPSA": round(feats[4], 2),
            })
            st.success("🟢 NON-TOXIC") if pred == 0 else st.error("🔴 TOXIC")
    else:
        st.sidebar.error("❌ Invalid SMILES string")

# -----------------------------
# Show Overview
st.metric("📊 Model Accuracy", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
st.dataframe(X_test_df.head(10))

# -----------------------------
# Visualize Data
fig = px.scatter(
    X_test_df,
    x="MolWt",
    y="LogP",
    color="Predicted",
    hover_data=["TPSA", "HDonors", "HAcceptors", "SMILES"],
    symbol="Predicted",
    title="Toxicity Prediction by Molecular Properties",
    width=1100,
    height=600
)
fig.update_traces(marker=dict(size=8))
st.plotly_chart(fig, use_container_width=True)

# Expandable Toxic Samples
with st.expander("🔍 View All Predicted Toxic Molecules"):
    st.dataframe(X_test_df[X_test_df["Predicted"] == 1])

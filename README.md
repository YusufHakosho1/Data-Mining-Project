# Mall Customers Segmentation & Analysis Dashboard

A fully interactive Streamlit dashboard for customer segmentation and association-rule mining on the Mall_Customers dataset. Includes:

- **Data Explorer** with real-time filters  
- **Clustering**: K-Means & Hierarchical, interactive 3D plots with cluster selection toggles  
- **Association Rules**: quantile binning, Apriori & rule discovery with CSV export  
- **Prediction**: assign a new customer to a cluster  
- Clean UI/UX, configurable parameters, CSV exports  

---

## 📁 Repository Structure

```
.
├── DataSets
│   └── Mall_Customers.csv        # Raw customer data
├── Mall Customer(GUI Advanced).py  # Full-featured Streamlit app
├── Mall Customer(GUI).py           # Simpler Streamlit app
├── Picture Results
│   └── Mall Customer.py            # Script generating static result images
├── README.md                       # ← You are here
└── test.py                         # Unit / smoke tests
```

---

## 🛠️ Prerequisites

- **Python 3.9+**  
- **Git** (optional, for cloning)  
- **VS Code** (or your favorite code editor)  

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YusufHakosho1/Data-Mining-Project.git
cd Data-Mining-Project
```

### 2. Create & Activate a Virtual Environment

#### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Windows (cmd.exe)
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **requirements.txt** should include:  
> ```
> streamlit
> pandas
> numpy
> scikit-learn
> scipy
> plotly
> mlxtend
> kneed
> matplotlib
> ```

---

## 📝 VS Code Setup

1. **Open Folder**: File → Open Folder → select project root.  
2. **Select Interpreter**: Ctrl+Shift+P → “Python: Select Interpreter” → choose `.venv`.  
3. **Install Extensions**:  
   - Python  
   - Pylance  
   - Streamlit Snippets (optional)  
4. **Debug Configuration** *(optional)*:  
   In `.vscode/launch.json`:
   ```jsonc
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Streamlit: Run App",
         "type": "python",
         "request": "launch",
         "program": "${workspaceFolder}/Mall Customer(GUI Advanced).py",
         "args": ["run"],
         "console": "integratedTerminal"
       }
     ]
   }
   ```

---

## 🚀 Running the App

From your activated venv:

```bash
streamlit run "Mall Customer(GUI Advanced).py"
```

Then open your browser at http://localhost:8501.

---

## 📊 Usage

1. **Data Explorer**: filter by genre, age, income, score; download filtered CSV.  
2. **Clustering**:  
   - **K-Means**: choose _k_, toggle clusters via legend or multiselect.  
   - **Hierarchical**: pick linkage method & distance threshold; interact with legend.  
3. **Association Rules**: tune bin counts, support & confidence, generate & export rules.  
4. **Predict Segment**: enter a new customer’s features to assign a cluster.

---



---

## 🤝 Contributing

1. Fork the repo  
2. Create a feature branch (`git checkout -b feat/YourFeature`)  
3. Commit & push (`git commit -m "Add feature"`)  
4. Open a Pull Request  

Please follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
